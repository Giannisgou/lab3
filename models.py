import torch
import numpy as np
from torch import nn


class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BaselineDNN, self).__init__()

        # EX4
        # 1 - define the embedding layer      
        num_embeddings = len(embeddings) 
        dim = len(embeddings[0])
        self.embeddings = nn.Embedding(num_embeddings, dim)
        self.output_size = output_size

        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        # 3 - define if the embedding layer will be frozen or finetuned
        if not trainable_emb:
            self.embeddings = self.embeddings.from_pretrained(torch.Tensor(embeddings), freeze = True)

        # 4 - define a non-linear transformation of the representations
        # EX5
        self.linear = nn.Linear(2 * dim, 1000) #1.1 double the size of the input
        self.relu = nn.ReLU()

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        # EX5
        self.output = nn.Linear(1000, output_size)

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """

        # 1 - embed the words, using the embedding layer
        # EX6
        embeddings = self.embeddings(x)

        # 2 - construct a sentence representation out of the word embeddings
        # EX6
        # calculate the means
        representations = torch.sum(embeddings, dim=1)
        for i in range(lengths.shape[0]): # necessary to skip zeros in mean calculation
            representations[i] = representations[i] / lengths[i]
        
        #1.1 - calculate the max
        representations_max, _ = torch.max(embeddings, dim=1)

        #1.1 - concatenate the means and max
        representations_concat = torch.cat((representations, representations_max), dim=1)
                    
        # 3 - transform the representations to new ones.
        # EX6
        representations = self.relu(self.linear(representations_concat))

        # 4 - project the representations to classes using a linear layer
        # EX6
        logits = self.output(representations)

        return logits


class LSTM(nn.Module):
    def __init__(self, output_size, embeddings, trainable_emb=False, bidirectional=False):

        super(LSTM, self).__init__()
        self.hidden_size = 100
        self.num_layers = 1
        self.bidirectional = bidirectional

        self.representation_size = 2 * \
            self.hidden_size if self.bidirectional else self.hidden_size

        embeddings = np.array(embeddings)
        num_embeddings, dim = embeddings.shape

        self.embeddings = nn.Embedding(num_embeddings, dim)
        self.output_size = output_size

        self.lstm = nn.LSTM(dim, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, bidirectional=self.bidirectional)

        if not trainable_emb:
            self.embeddings = self.embeddings.from_pretrained(
                torch.Tensor(embeddings), freeze=True)

        self.linear = nn.Linear(self.representation_size, output_size)

    def forward(self, x, lengths):
        batch_size, max_length = x.shape
        embeddings = self.embeddings(x)
        X = torch.nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False)

        ht, _ = self.lstm(X)

        # ht is batch_size x max(lengths) x hidden_dim
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)
        
        # pick the output of the lstm corresponding to the last word
        # TODO: Main-Lab-Q2 (Hint: take actual lengths into consideration)
        representations = []
        for i in range(lengths.shape[0]):
            last = lengths[i] - 1 if lengths[i] <= max_length else max_length - 1      
            if self.bidirectional:
                forward_out = ht[i, last, :self.hidden_size]
                backward_out = ht[i, 0, self.hidden_size:]
                rep = torch.cat((forward_out, backward_out), dim=-1)
            else:
                rep = ht[i, last, :]
            representations.append(rep)
        representations = torch.stack(representations, dim=0)  # σχήμα: (batch_size, representation_size)
        logits = self.linear(representations)

        return logits
