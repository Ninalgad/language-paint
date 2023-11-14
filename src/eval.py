import numpy as np
import torch

from sklearn.metrics import f1_score


def predict(model, val_loader, return_labels=False, return_logits=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    predictions, labels = [], []
    y = None
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if 'labels' in batch:
            y = batch['labels'].detach().cpu().numpy()
            del batch['labels']
        with torch.no_grad():
            logits = model(**batch).logits
        p = logits.detach().cpu().numpy()
        if not return_logits:
            p = np.argmax(p, axis=-1)
        predictions.append(p)
        labels.append(y)

    predictions, labels = np.concatenate(predictions), np.concatenate(labels)
    if return_labels:
        return predictions, labels
    return predictions


def evaluate(model, val_loader, val_lang):
    val_lang = np.array(val_lang)
    model.eval()

    predictions, labels = predict(model, val_loader, return_labels=True)
    scores = dict()
    for lang in sorted(set(val_lang)):
        idx = val_lang == lang
        s = f1_score(predictions[idx], labels[idx], average='weighted')
        scores[lang] = s
    return np.mean(list(scores.values())), scores
