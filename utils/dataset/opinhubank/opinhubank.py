import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter


def read_excel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="iso8859_16", index_col=0)
    return df


def preproc(df):
    def aggregate(p):
        freqs = Counter(p).most_common()
        if len(freqs) > 2 and freqs[0][1] == freqs[1][1]:
            return 0
        else:
            assert (-1 <= int(freqs[0][0]) and int(freqs[0][0]) <= 1)
            return int(freqs[0][0])

    annot = zip(df.Annot1, df.Annot2, df.Annot3, df.Annot4, df.Annot5)
    aggr = [aggregate(ann) for ann in annot]
    df["Aggregate"] = aggr
    return df


def split(df):
    train_df, test_df = train_test_split(df[["Entity", "Sentence", "Aggregate"]], test_size=0.2, random_state=1)
    train_df, dev_df = train_test_split(train_df, test_size=0.125, random_state=1)
    return train_df, test_df, dev_df


def read_data(path: str, seed: int = 0, binary=False):
    df = read_excel(path)
    df = preproc(df)
    if binary:
        df = df.loc[df.Aggregate != 0]
    train_df, test_df, dev_df = split(df)
    train_sentence, train_label = train_df['Sentence'].to_numpy(), train_df['Aggregate'].to_numpy()
    test_sentence, test_label = test_df['Sentence'].to_numpy(), test_df['Aggregate'].to_numpy()
    dev_sentence, dev_label = dev_df['Sentence'].to_numpy(), dev_df['Aggregate'].to_numpy()

    train = {
        "sentence": train_sentence.tolist(),
        "labels": (train_label + 1).tolist() if not binary else ((train_label + 1) // 2).tolist()
    }
    test = {
        "sentence": test_sentence.tolist(),
        "labels": (test_label + 1).tolist() if not binary else ((test_label + 1) // 2).tolist()
    }
    dev = {
        "sentence": dev_sentence.tolist(),
        "labels": (dev_label + 1).tolist() if not binary else ((dev_label + 1) // 2).tolist()
    }
    return train, test, dev
