from loguru import logger
from pathlib import Path
import typer
import torch
import json

from src.coallate import create_dataset
from src.train import train
from src.awe import awe, finetune_languages
from src.dataloader import get_loader
from src.eval import evaluate


def main(
        model_dir: Path = typer.Option(
            "../models", help="File to the output model weights in npy format"
        ),
        data_path: Path = typer.Option(
            "../data/", help="Path to the raw features"
        ),
        model_name: str = typer.Option(
            "jhu-clsp/bernice", help="Model name of the language model."
        ),
        input_len: int = typer.Option(
            128, help="Number of input tokens fed to the model"
        ),
        debug: bool = typer.Option(
            False, help="Run on a small subset of the data and a two folds for debugging"
        )
):
    n_rows = None
    if debug:
        logger.info("Running in debug mode")
        n_rows = 10
    logger.info(f"Creating data from {data_path}")
    df_train, df_dev, df_test = create_dataset(data_path, n_rows=n_rows)
    languages = sorted(df_dev.language.unique())

    model_config = {
        "name": model_name,
        "input_len": input_len,
    }

    # train
    logger.info("Training base model")
    s, meta = train(model_dir / 'model',
                    df_train.text.values, df_dev.text.values,
                    df_train.category.values, df_dev.category.values,
                    df_dev.language.values,
                    model_config=model_config,
                    batch_size=16, debug=debug)

    logger.info("Finetuning languages")
    finetune_languages(df_train, df_dev, model_dir / 'model-ft',
                       model_dir / 'model.pt', model_config,
                       jsd_alphas=None, num_epochs=3, debug=debug)

    logger.info("Running AWE")
    scores = {lang: [] for lang in languages}
    baseline = {lang: [] for lang in languages}
    alphas = dict()
    for lang in sorted(df_dev.language.unique()):
        awe_scores, best_alpha, awe_model, awe_tokenizer = awe(lang, model_dir / f'model-ft-{lang}.pt',
                                                               model_dir / 'model.pt',
                                                               df_dev.text.values, df_dev.category.values,
                                                               df_dev.language.values, model_config=model_config,
                                                               batch_size=32, return_best_model=True)
        lang_ft_score, mlang_score = awe_scores[0], awe_scores[-1]

        alphas[lang] = best_alpha

        # evaluate on dev_set
        df_eval = df_dev[df_dev.language == lang]
        test_loader = get_loader(awe_tokenizer, model_config['input_len'], df_eval.text.values, df_eval.category.values,
                                 batch_size=32,
                                 shuffle=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        s_dev, meta = evaluate(awe_model, test_loader, df_eval.language.values, device)

        # evaluate on test_set
        df_eval = df_test[df_test.language == lang]
        test_loader = get_loader(awe_tokenizer, model_config['input_len'], df_eval.text.values, df_eval.category.values,
                                 batch_size=32,
                                 shuffle=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        s_test, meta = evaluate(awe_model, test_loader, df_eval.language.values, device)

        del awe_model, awe_tokenizer, df_eval

        baseline[lang].append((lang_ft_score, mlang_score))
        scores[lang].append((s_dev, s_test))

    results = {'baseline': baseline, 'awe': scores}
    with open(model_dir / 'results.json', 'w') as f:
        json.dump(results, f)
    logger.success(f"Completed; saved results to {model_dir / 'results.json'}")


if __name__ == "__main__":
    typer.run(main)
