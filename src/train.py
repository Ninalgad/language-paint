from torch.optim import Adam
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gc
import torch

from src.loss import JSD
from src.dataloader import get_loader
from src.eval import evaluate


def get_model(model_config):
    tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
    model = AutoModelForSequenceClassification.from_pretrained(model_config['name'], num_labels=3)
    return model, tokenizer


def train(model_save_name, x_train, x_dev, y_train, y_dev, df_dev_language,
          model_config, batch_size=16, num_epochs=14,
          model=None, tokenizer=None, optimizer=None, jsd_alpha=1., debug=False):
    if model is None:
        model, tokenizer = get_model(model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_loader = get_loader(tokenizer, model_config['input_len'], x_train, y_train,
                              batch_size=batch_size, shuffle=True)
    val_loader = get_loader(tokenizer, model_config['input_len'], x_dev, y_dev,
                            batch_size=batch_size, shuffle=False)

    if optimizer is None:
        optimizer = Adam(model.parameters(), lr=1e-5)

    def train_step(inp):
        inp = {k: v.to(device) for k, v in inp.items()}
        optimizer.zero_grad()
        labels = inp['labels']
        del inp['labels']
        outputs = model(**inp)
        logits = outputs.logits
        loss_ = JSD(jsd_alpha)(logits, labels)
        loss_.backward()
        optimizer.step()
        return loss_.detach().cpu().numpy()

    best_ep = 0
    best_val = -1
    global_step = 0
    steps_per_epoch = len(train_loader)
    best_details = {}
    if debug:
        num_epochs = 1

    for epoch in range(num_epochs):
        model.train()
        train_loss = []
        for batch in tqdm(train_loader):
            loss = train_step(batch)
            train_loss.append(loss)
            global_step += 1
            if debug:
                break

        gc.collect()
        val, details = evaluate(model, val_loader, df_dev_language, device)
        gc.collect()
        if val > best_val:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                str(model_save_name) + '.pt')
            best_val = val
            best_ep = epoch
            best_details = details

    return best_val, best_details
