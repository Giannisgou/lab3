# transfer_pretrained.py

from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm import tqdm
import pandas as pd

from utils.load_datasets import load_MR, load_Semeval2017A

# --- 1. διάλεξε εδώ 3 μοντέλα που θέλεις να δοκιμάσεις ---
PRETRAINED_MODELS = [
    'siebert/sentiment-roberta-large-english',
    'cardiffnlp/twitter-roberta-base-sentiment',
    'nlptown/bert-base-multilingual-uncased-sentiment',
]

# mapping των ετικετών απ' το κάθε μοντέλο στο δικό μας label set
LABELS_MAPPING = {
    'siebert/sentiment-roberta-large-english': {
        'POSITIVE': 'positive', 'NEGATIVE': 'negative'
    },
    'cardiffnlp/twitter-roberta-base-sentiment': {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive'
    },
    'nlptown/bert-base-multilingual-uncased-sentiment': {
        '1 star': 'negative',
        '2 stars': 'negative',
        '3 stars': 'neutral',
        '4 stars': 'positive',
        '5 stars': 'positive',
    },
}

DATASETS = {
    'MR': load_MR,
    'Semeval2017A': load_Semeval2017A
}

if __name__ == '__main__':
    results = []
    print("=== Transfer Learning with Pre-trained Models ===")

    for ds_name, loader in DATASETS.items():
        # 1) φορτώνουμε το dataset
        X_train, y_train, X_test, y_test = loader()

        # 2) ετοιμάζουμε τον LabelEncoder
        #    Για το MR επιβάλλουμε να υπάρχει και η ετικέτα 'neutral'
        unique_labels = set(y_train)
        if ds_name == 'MR':
            unique_labels.add('neutral')
        le = LabelEncoder()
        le.fit(list(unique_labels))

        # 3) κωδικοποιούμε τις χρυσές ετικέτες
        y_test_enc = le.transform(y_test)

        for model_name in PRETRAINED_MODELS:
            print(f"\n==> Dataset: {ds_name}  |  Model: {model_name}")
            sentiment = pipeline("sentiment-analysis", model=model_name)

            # 4) προβλέπουμε πάνω στο test set
            y_pred = []
            for txt in tqdm(X_test, desc="Predicting"):
                out = sentiment(txt)[0]['label']
                mapped = LABELS_MAPPING[model_name][out]
                y_pred.append(mapped)

            # 5) κωδικοποιούμε τις προβλέψεις
            y_pred_enc = le.transform(y_pred)

            # 6) μετρικά
            acc = accuracy_score(y_test_enc, y_pred_enc)
            f1  = f1_score(y_test_enc, y_pred_enc, average='macro', zero_division=0)
            rec = recall_score(y_test_enc, y_pred_enc, average='macro', zero_division=0)

            results.append({
                'dataset': ds_name,
                'model': model_name,
                'accuracy': acc,
                'f1 (macro)': f1,
                'recall (macro)': rec
            })

    # Τυπώνουμε τον συγκριτικό πίνακα
    df = pd.DataFrame(results)
    print("\n=== Σύγκριση Αποτελεσμάτων ===")
    print(
        df
        .pivot(index='model', columns='dataset', values=['accuracy','f1 (macro)','recall (macro)'])
        .round(4)
    )
    df.to_csv("results.csv", index=False)

