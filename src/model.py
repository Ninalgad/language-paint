import torch
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import ID2LABEL


def create_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, id2label=ID2LABEL)
    return model, tokenizer


def load_model_algorithm(model_config, pt_file):
    model, tokenizer = create_model(model_config)
    optimizer = Adam(model.parameters(), lr=1e-5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model_obj = torch.load(pt_file)
    model.load_state_dict(model_obj['model_state_dict'])
    optimizer.load_state_dict(model_obj['optimizer_state_dict'])

    return model, tokenizer, optimizer
