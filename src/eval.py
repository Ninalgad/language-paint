import pandas as pd
from transformers import TextClassificationPipeline
from sklearn.metrics import f1_score

from utils import LABEL2ID


def predict(text, model, tokenizer, input_length=128, batch_size=32):
    model.to('cpu')
    model.eval()

    classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, framework='pt', device=None)

    predictions = []
    for out in classifier(text, batch_size=batch_size, max_length=input_length):
        predictions.append(out)

    predictions = pd.DataFrame(predictions)['label'].map(lambda x: LABEL2ID[x])
    return predictions


def evaluate(model, tokenizer, df, input_length=128, batch_size=32):
    predictions = predict(df['text'].tolist(), model, tokenizer, input_length, batch_size)
    df = df.reset_index(drop=True)

    scores = {'overall': f1_score(df.category, predictions, average='weighted')}
    for lang, g in df.groupby('language'):
        s = f1_score(g.category, predictions.loc[g.index], average='weighted')
        scores[lang] = s

    return scores
