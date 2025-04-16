import os
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN, LSTM
from training import train_dataset, eval_dataset, torch_train_val_split
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors
import nltk 
nltk.download('punkt')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score
from early_stopper import EarlyStopper
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50
DATASET = "Semeval2017A"  # options: "MR", "Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# Convert data labels from strings to integers

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

label_encoder.fit(y_train)

y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)
n_classes = label_encoder.classes_.size

print("Τα πρώτα 10 labels (μετατροπή σε ακέραιους):", y_train[:10]) 
print("Αντιστοίχιση label -> ακέραιος:") 
for i, label in enumerate(label_encoder.classes_): print(f"{label} -> {i}")

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)

print("Τα πρώτα 10 παραδείγματα από τα δεδομένα εκπαίδευσης:") 
for i in range(10): 
    original_text = X_train[i] 
    processed_example, label, length = train_set[i] 
    print(f"\nΠαράδειγμα {i}:") 
    print("Αρχικό κείμενο:", original_text) 
    print("Επεξεργασμένη μορφή:", processed_example) 
    print("Label:", label) 
    print("Πραγματικό μήκος:", length)

print("\nΠέντε παραδείγματα από το training set:") 
for i in range(5): 
    original_text = X_train[i] 
    processed_example, label, length = train_set[i] 
    print(f"\nΠαράδειγμα {i}:") 
    print("Αρχικό κείμενο:", original_text) 
    print("Επεξεργασμένη μορφή:", processed_example) 
    print("Label:", label) 
    print("Πραγματικό μήκος:", length)


# EX7 - Define our PyTorch-based DataLoader
# train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

# 2.1###############################################################
train_loader, val_loader = torch_train_val_split(
    dataset=train_set,
    batch_train=BATCH_SIZE, 
    batch_eval=BATCH_SIZE,
    val_size=0.2,      # 20% validation
    shuffle=True,
    seed=42
)
######################################################################

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################
'''
model = BaselineDNN(output_size=3,  # EX8 #################################### 2 for MR, 3 for Semeval2017A
                    embeddings=embeddings,
                    trainable_emb=EMB_TRAINABLE)
'''
model = LSTM(output_size = 3, 
             embeddings=embeddings,
             trainable_emb=EMB_TRAINABLE)
# move the mode weight to cpu or gpu
model.to(DEVICE)
print(model)
early_stopper = EarlyStopper(model = model, save_path= "best_model.pth", patience=5, min_delta=0.0)

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
criterion = torch.nn.CrossEntropyLoss() 
parameters = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.Adam(parameters, lr=0.001)


#############################################################################
# Training Pipeline
#############################################################################
# lists to acummulate train and test losses 
TRAIN_LOSS = []
TEST_LOSS = []
VAL_LOSS = []
for epoch in range(1, EPOCHS + 1):
    '''
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                            model,
                                                            criterion)

    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
                                                         model,
                                                         criterion)
    '''
    # a. Εκπαιδεύουμε το μοντέλο στο training split
    train_dataset(epoch, train_loader, model, criterion, optimizer)
    # b. Υπολογίζουμε το training loss
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader, model, criterion)
    # b. Υπολογίζουμε το validation loss
    val_loss, (val_y_pred, val_y_true) = eval_dataset(val_loader, model, criterion)
    # b. Υπολογίζουμε το test loss
    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader, model, criterion)

    print(f"[Epoch {epoch}] Training loss: {train_loss:.4f} | Validation loss: {val_loss:.4f} | Test loss: {test_loss:.4f}")

    # 3c. Ελέγχουμε αν πρέπει να διακόψουμε πρόωρα
    if early_stopper.early_stop(val_loss):
        print(f"Early stopping at epoch {epoch}. Validation loss has not improved for {early_stopper.patience} epochs.")
        break


    TRAIN_LOSS.append(train_loss)
    TEST_LOSS.append(test_loss)
    VAL_LOSS.append(val_loss)
    # concatenate the predictions and gold labels in lists.
    y_train_true = np.concatenate( y_train_gold, axis=0 )
    y_test_true = np.concatenate( y_test_gold, axis=0 )
    y_train_pred = np.concatenate( y_train_pred, axis=0 )
    y_test_pred = np.concatenate( y_test_pred, axis=0 )
    # compute metrics using sklearn functions
    val_y_true = np.concatenate(val_y_true, axis=0)
    val_y_pred = np.concatenate(val_y_pred, axis=0)
# Compute metrics using sklearn functions:
print("Train loss:" , train_loss)
print("Validation loss:" , val_loss)
print("Test loss:", test_loss)

print("Train accuracy:" , accuracy_score(y_train_true, y_train_pred))
print("Validation accuracy:" , accuracy_score(val_y_true, val_y_pred))
print("Test accuracy:" , accuracy_score(y_test_true, y_test_pred))

print("Train F1 score:", f1_score(y_train_true, y_train_pred, average='macro'))
print("Validation F1 score:", f1_score(val_y_true, val_y_pred, average='macro'))
print("Test F1 score:", f1_score(y_test_true, y_test_pred, average='macro'))

print("Train Recall:", recall_score(y_train_true, y_train_pred, average='macro'))
print("Validation Recall:", recall_score(val_y_true, val_y_pred, average='macro'))
print("Test Recall:", recall_score(y_test_true, y_test_pred, average='macro'))
# plot training and validation loss curves
    
plt.plot(range(1, EPOCHS + 1), TRAIN_LOSS, label='Training Loss')
plt.plot(range(1, EPOCHS + 1), TEST_LOSS, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()
