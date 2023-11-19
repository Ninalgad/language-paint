from loguru import logger
from pathlib import Path

import typer
import json
import torch
import numpy as np

from src.collate import create_dataset
from src.eval import evaluate
from src.model import create_model


def main(
        output_dir: Path = typer.Option(
            "./models", help="File to the output model weights in npy format"
        ),
        data_path: Path = typer.Option(
            "./data/", help="Path to the raw features"
        ),
        model_name: str = typer.Option(
            "jhu-clsp/bernice", help="Model name of the language model."
        ),
        input_len: int = typer.Option(
            128, help="Number of input tokens fed to the model"
        ),
        batch_size: int = typer.Option(
            16, help="Number of samples fed to the model"
        ),
        debug: bool = typer.Option(
            False, help="Run on a small subset of the data and a two folds for debugging"
        )
):
    n_rows = None
    if debug:
        logger.info("Running in debug mode")
        n_rows = 2
        input_len = 4

    model, tokenizer = create_model(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    logger.info(f"Creating data from {data_path}")
    dataset, collator, df_train, df_dev, df_test = create_dataset(data_path, tokenizer, input_len, n_rows=n_rows)
    multilingual_filename = output_dir / 'multilingual/best-model.pt'
    languages = df_test.language.unique()

    results = dict()
    for i, lang in enumerate(languages):
        lang_filename = output_dir / f'{lang}/best-model.pt'

        # get language subsets from dataframes
        df_test_ = df_test[df_test.language == lang]
        df_dev_ = df_dev[df_dev.language == lang]

        if (len(df_dev_) == 0) or (len(df_test_) == 0):
            logger.warn(f"Not enough sampels for `{lang}`")

        else:
            logger.info(f"Performing AWE on {lang}")
            scores, best_alpha, model, tokenizer = awe(model, tokenizer,
                                                       multilingual_filename, lang_filename, df_dev_, input_len,
                                                       batch_size, debug)

            # f1 score on held-out sets
            dev_score = evaluate(model, tokenizer, df_dev_, input_len, batch_size)['overall']
            test_score = evaluate(model, tokenizer, df_test_, input_len, batch_size)['overall']

            # update results
            lang_result = {
                'scores': scores, 'best_alpha': best_alpha,
                'dev': dev_score, 'test': test_score
            }
            results[lang] = lang_result

    with open(output_dir / 'awe_results.json', 'w') as f:
        json.dump(results, f)
    logger.success(f"Completed AWE; saved results to {output_dir / 'awe_results.json'}")


def awe(model, tokenizer, w0_file, w1_file, df_eval, input_len, batch_size, debug=False):
    model.load_state_dict(torch.load(w0_file))
    weights0 = {name: param.clone().detach() for name, param in model.named_parameters()}
    model.load_state_dict(torch.load(w1_file))
    weights1 = {name: param.clone().detach() for name, param in model.named_parameters()}

    scores = []
    alphas = np.linspace(0, 1, num=11, endpoint=True)
    for alpha in alphas:
        beta = 1. - alpha

        # interpolate weights
        for name, param in model.named_parameters():
            param.data = alpha * weights0[name] + beta * weights1[name]

        # evaluation
        s = evaluate(model, tokenizer, df_eval, input_len, batch_size)['overall']
        scores.append(s)

        if debug:
            break

    best_alpha = alphas[np.argmax(scores)]
    beta = 1. - best_alpha

    # create best model
    for name, param in model.named_parameters():
        param.data = best_alpha * weights0[name] + beta * weights1[name]

    return scores, best_alpha, model, tokenizer


if __name__ == "__main__":
    typer.run(main)
