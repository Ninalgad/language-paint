from loguru import logger
from pathlib import Path
import typer
import json

from src.coallate import create_dataset
from src.train import train
from src.awe import awe, finetune_languages
from src.dataloader import get_loader
from src.eval import evaluate
from model import load_model_algorithm


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
        n_rows = 2
        input_len = 4
    logger.info(f"Creating data from {data_path}")
    df_train, df_dev, df_test = create_dataset(data_path, n_rows=n_rows)
    languages = sorted(df_dev.language.unique())

    model_config = {
        "name": model_name,
        "input_len": input_len,
    }

    # train
    logger.info("Training base model")
    train(
        model_dir / 'model', df_train.text.values, df_dev.text.values, df_train.category.values, df_dev.category.values,
        df_dev.language.values, model_config=model_config, batch_size=16,  debug=debug
    )
    model, tokenizer, _ = load_model_algorithm(model_config, str(model_dir / f'model.pt'))
    test_loader = get_loader(
        tokenizer, model_config['input_len'], df_test.text.values, df_test.category.values,
        batch_size=32, shuffle=False
    )
    _, ml_test_scores = evaluate(model, test_loader, df_test.language.values)

    logger.info("Finetuning languages")
    finetune_languages(
        df_train, df_dev, model_dir / 'model-ft', model_dir / 'model.pt', model_config,
        jsd_alphas=None, num_epochs=3, debug=debug
    )
    ls_test_scores = dict()
    for lang in languages:
        model, _, _ = load_model_algorithm(model_config, str(model_dir / f'model-ft-{lang}.pt'))
        _, ls_scores = evaluate(model, test_loader, df_test.language.values)
        ls_test_scores[lang] = ls_scores[lang]

    logger.info("Running AWE")
    empty = {l: None for l in languages}
    results = {
        'ml': {'test': ml_test_scores, 'dev': empty.copy()},
        'ls': {'test': ls_test_scores, 'dev': empty.copy()},
        'lp': {'test': empty.copy(), 'dev': empty.copy(), 'alpha': empty.copy()}
    }
    for lang in sorted(df_dev.language.unique()):
        awe_scores, best_alpha, awe_model, tokenizer = awe(
            lang, model_dir / f'model-ft-{lang}.pt', model_dir / 'model.pt',
            df_dev.text.values, df_dev.category.values,
            df_dev.language.values, model_config=model_config,
            batch_size=32, return_best_model=True
        )
        ls_dev_score, ml_dev_score = awe_scores[0], awe_scores[-1]

        df_dev_lang = df_dev[df_dev.language == lang]
        dev_loader = get_loader(
            tokenizer, model_config['input_len'], df_dev_lang.text.values, df_dev_lang.category.values,
            batch_size=32, shuffle=False
        )
        df_test_lang = df_test[df_test.language == lang]
        test_loader = get_loader(
            tokenizer, model_config['input_len'], df_test_lang.text.values, df_test_lang.category.values,
            batch_size=32, shuffle=False
        )

        s_dev, _ = evaluate(awe_model, dev_loader, df_dev_lang.language.values)
        s_test, _ = evaluate(awe_model, test_loader, df_test_lang.language.values)

        del awe_model, df_dev_lang, df_test_lang, dev_loader, test_loader

        results['ml']['dev'][lang] = ml_dev_score
        results['ls']['dev'][lang] = ls_dev_score
        results['lp']['dev'][lang] = s_dev
        results['lp']['test'][lang] = s_test
        results['lp']['alpha'][lang] = best_alpha

    with open(model_dir / 'results.json', 'w') as f:
        json.dump(results, f)
    logger.success(f"Completed; saved results to {model_dir / 'results.json'}")


if __name__ == "__main__":
    typer.run(main)
