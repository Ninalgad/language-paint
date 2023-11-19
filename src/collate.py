import pandas as pd
from glob import glob
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding

from src.utils import load_preprocess_dataframe


def create_split(dir_, split, n_rows=None):
    df = []
    for f in glob(f"{dir_}/*3_{split}.csv"):
        df.append(load_preprocess_dataframe(f, n_rows=n_rows))
    df = pd.concat(df)
    df = df.reset_index(drop=True)
    return df


def create_dataset(data_dir, tokenizer, input_len, n_rows=None):
    # read csvs
    df_train = create_split(data_dir, 'train', n_rows)
    df_dev = create_split(data_dir, 'dev', n_rows)
    df_test = create_split(data_dir, 'test', n_rows)

    # convert to datasets
    ds_train = Dataset.from_pandas(df_train[['text', 'category']])
    ds_dev = Dataset.from_pandas(df_dev[['text', 'category']])
    ds_test = Dataset.from_pandas(df_test[['text', 'category']])

    ds = DatasetDict()
    ds['train'] = ds_train
    ds['validation'] = ds_dev
    ds['test'] = ds_test

    # tokenize dataset
    def tokenize_function(batch):
      inp = tokenizer(batch['text'], padding="max_length", truncation=True,
                      max_length=input_len)
      inp['category'] = batch['category']
      return inp

    tokenized_datasets = ds.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column('category', 'labels')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return tokenized_datasets, data_collator, df_train, df_dev, df_test
