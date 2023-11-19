import pandas as pd

LABEL2ID = {'Non-anti-LGBT+ content': 0, 'Transphobia': 1, 'Homophobia': 2,
            'Homophobic': 2, 'Transphobic': 1, 'None': 0}

ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def parse_file_name(f):
    f = f.split('/')[-1][:-4]
    f = f.split('_')
    lang = f[1]
    split = f[-1]
    return lang, split


def load_preprocess_dataframe(filename, has_labels=True, n_rows=None):
    df = pd.read_csv(filename, nrows=n_rows).dropna()
    df = df.rename(columns={'Labels': 'category',
                            'text                ': 'text',
                            'text                        ': 'text'})
    if has_labels:
        df = df[['text', 'category']]
        df['category'] = df['category'].map(lambda x: LABEL2ID[x])
    else:
        df = df[['text']]

    lang, split = parse_file_name(filename)
    df['language'] = lang
    df['split'] = split
    return df
