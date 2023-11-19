import numpy as np
import shutil
from glob import glob
from transformers import TrainingArguments
from transformers import Trainer
import evaluate

from src.eval import evaluate


def compute_metrics(eval_preds):
    metric = evaluate.load("f1")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average='weighted')


def _flush_runs(ckpt_dir):
    for dir_name in glob(str(ckpt_dir / "checkpoint-*")):
        shutil.rmtree(dir_name)
    shutil.rmtree(ckpt_dir / 'runs')


def train(output_dir, model, tokenizer, dataset, collator, num_epochs,
          batch_size=16, lr=1e-5, debug=False):
    log_level = 'debug' if debug else 'passive'

    training_args = TrainingArguments(
        output_dir,
        num_train_epochs = num_epochs,
        overwrite_output_dir=True,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        gradient_accumulation_steps = 1,
        learning_rate = lr,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        log_level = log_level,
        save_total_limit=1,
        disable_tqdm=True
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # train
    trainer.train()

    # clear checkpoints to save memory
    _flush_runs(output_dir)
