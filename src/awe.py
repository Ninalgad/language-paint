import torch
import gc
import numpy as np
from torch.optim import Adam

from src.train import train, get_model
from src.eval import evaluate
from src.dataloader import get_loader


def get_mixed_model(model_config, files):
    model, tokenizer = get_model(model_config)

    weights = {name: [] for name, param in model.named_parameters()}

    with torch.no_grad():
        for f in files:
            model.load_state_dict(torch.load(f)['model_state_dict'])
            for name, param in model.named_parameters():
                weights[name].append(param.data)

    for name, param in model.named_parameters():
        w = weights[name]
        param.data = sum(w) / len(w)

    return model, tokenizer


def load_algo_model(model_config, pt_file):
    model, tokenizer = get_model(model_config)
    optimizer = Adam(model.parameters(), lr=1e-5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        model_obj = torch.load(pt_file)
        model.load_state_dict(model_obj['model_state_dict'])
        optimizer.load_state_dict(model_obj['optimizer_state_dict'])

    return model, tokenizer, optimizer


def finetune_languages(df_train, df_dev, model_save_name_base, pt_file, model_config,
                       jsd_alphas=None, num_epochs=3, debug=False):
    scores = {}
    languages = sorted(df_dev.language.unique())
    if jsd_alphas is None:
        jsd_alphas = [1] * len(languages)
    else:
        assert len(languages) == len(jsd_alphas)

    for i, (lang, alpha) in enumerate(zip(languages, jsd_alphas)):
        train_index = df_train.language == lang
        test_index = df_dev.language == lang

        train_data, dev_data = df_train[train_index], df_dev[test_index]
        gc.collect()

        model, tokenizer, optimizer = load_algo_model(model_config, pt_file)
        s = train(f'{model_save_name_base}-{lang}',
                  train_data.text.values, dev_data.text.values,
                  train_data.category.values, dev_data.category.values,
                  dev_data.language.values,
                  batch_size=16, model_config=model_config,
                  model=model, tokenizer=tokenizer, optimizer=optimizer,
                  jsd_alpha=alpha, num_epochs=num_epochs, debug=debug)
        del model, tokenizer
        gc.collect()
        scores[lang] = s
    return scores


def awe(lang, w0_file, w1_file, text, labels, lang_data, model_config, batch_size=32,
        return_best_model=False, debug=False):
    model, tokenizer = get_model(model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_loader = get_loader(tokenizer, model_config['input_len'],
                             text, labels, batch_size=batch_size, shuffle=False)

    model.load_state_dict(torch.load(w0_file)['model_state_dict'])
    weights0 = {name: param.clone().detach() for name, param in model.named_parameters()}
    model.load_state_dict(torch.load(w1_file)['model_state_dict'])
    weights1 = {name: param.clone().detach() for name, param in model.named_parameters()}

    scores = []
    alphas = np.linspace(0, 1, num=11, endpoint=True)
    for alpha in alphas:
        beta = 1. - alpha
        for name, param in model.named_parameters():
            param.data = alpha * weights0[name] + beta * weights1[name]

        s, meta = evaluate(model, test_loader, lang_data, device)
        scores.append(meta[lang])
        if debug:
            break

    if return_best_model:
        best_alpha = alphas[np.argmax(scores)]
        beta = 1. - best_alpha
        for name, param in model.named_parameters():
            param.data = best_alpha * weights0[name] + beta * weights1[name]
        return scores, best_alpha, model, tokenizer

    return scores, alphas
