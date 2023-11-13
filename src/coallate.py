from glob import glob
import pandas as pd

from src.utils import load_preprocess_dataframe


def create_split(dir_, split, n_rows=None):
    df = []
    for f in glob(f"{dir_}/*3_{split}.csv"):
        df.append(load_preprocess_dataframe(f))
    df = pd.concat(df)
    return df


def create_dataset(dir_, n_rows=None):
    df_test = create_split(dir_, 'test', n_rows)
    df_train = create_split(dir_, 'train', n_rows)
    df_dev = create_split(dir_, 'dev', n_rows)
    return df_train, df_dev, df_test
