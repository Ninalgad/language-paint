from loguru import logger
from pathlib import Path
from copy import deepcopy
import typer
import json
import torch

from src.collate import create_dataset
from src.trainer import train
from src.eval import evaluate
from src.model import create_model


def main(
        output_dir: Path = typer.Option(
            "./models", help="File to the output model weights in npy format"
        ),
        data_path: Path = typer.Option(
            "./data/", help="Path to the raw features"
        ),
        trained_mul_model: Path = typer.Option(
            "./models/multilingual/best-model.pt", help="Path to trained model"
        ),
        model_name: str = typer.Option(
            "jhu-clsp/bernice", help="Model name of the language model."
        ),
        num_epochs: int = typer.Option(
            5, help="Number of itraining epochs"
        ),
        lr: float = typer.Option(
            1e-5, help="Learning rate"
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
        num_epochs = 1

    model, tokenizer = create_model(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if trained_mul_model is not None:
        # load model
        logger.info(f"Loading model from {trained_mul_model}")
        model.load_state_dict(torch.load(trained_mul_model, device))

    logger.info(f"Creating data from {data_path}")
    dataset, collator, df_train, df_dev, df_test = create_dataset(data_path, tokenizer, input_len, n_rows=n_rows)

    languages = df_test.language.unique()

    for i, lang in enumerate(languages):

        # get language subsets from dataset
        ds = deepcopy(dataset)
        ds['train'] = ds['train'].select(df_train[df_train.language == lang].index.tolist())
        ds['validation'] = ds['validation'].select(df_dev[df_dev.language == lang].index.tolist())
        df_test_ = df_test[df_test.language == lang]

        if (len(ds['train']) == 0) or (len(ds['validation']) == 0) or (len(df_test_) == 0):
            logger.warning(f"Not enough samples for `{lang}`")

        else:

            # train
            logger.info(f"({i + 1}/{len(languages)}) Training model for `{lang}`")
            train(output_dir / lang, model, tokenizer, ds, collator, num_epochs, batch_size=batch_size, lr=lr,
                  debug=debug)

            # f1 score on held-out sets
            dev_scores = evaluate(model, tokenizer, df_dev[df_dev.language == lang], input_len, batch_size)
            test_scores = evaluate(model, tokenizer, df_test_, input_len, batch_size)

            # save results
            torch.save(model.state_dict(), output_dir / f'{lang}/best-model.pt')
            with open(output_dir / f'{lang}/dev_results.json', 'w') as f:
                json.dump(dev_scores, f)
            with open(output_dir / f'{lang}/test_results.json', 'w') as f:
                json.dump(test_scores, f)
            logger.success(f"Completed {lang}; saved results to {output_dir / lang}")


if __name__ == "__main__":
    typer.run(main)
