from loguru import logger
from pathlib import Path
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

    logger.info(f"Creating data from {data_path}")
    dataset, collator, _, df_dev, df_test = create_dataset(data_path, tokenizer, input_len, n_rows=n_rows)

    # train
    logger.info("Training `multilingual` model")
    train(output_dir / 'multilingual', model, tokenizer, dataset, collator, num_epochs, batch_size=batch_size, lr=lr, debug=debug)

    # f1 score on held-out sets
    logger.info("Evaluating model")
    dev_scores = evaluate(model, tokenizer, df_dev, input_len, batch_size)
    test_scores = evaluate(model, tokenizer, df_test, input_len, batch_size)

    # save results
    torch.save(model.state_dict(), output_dir / 'multilingual/best-model.pt')
    with open(output_dir / 'multilingual/dev_results.json', 'w') as f:
        json.dump(dev_scores, f)
    with open(output_dir / 'multilingual/test_results.json', 'w') as f:
        json.dump(test_scores, f)
    logger.success(f"Completed; saved results to {output_dir / 'multilingual'}")


if __name__ == "__main__":
    typer.run(main)
