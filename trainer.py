import os

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_metric, load_dataset

from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from consts import INTENT_DATASET_DIR, INTENT_MODEL_DIR


use_local_dataset = False

base_model = 'roberta-large'
n_epochs = 15
learning_rate = 5e-6


def run_trainer(dataset, n_labels, do_train=False, do_eval=False, do_predict=False):
    metric = load_metric("accuracy")
    model_name = base_model if do_train else INTENT_MODEL_DIR
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def compute_metrics(p):
        logits, labels = p
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=INTENT_MODEL_DIR,
        learning_rate=learning_rate,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=n_epochs,
        load_best_model_at_end=True,
        save_total_limit=1,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        metric_for_best_model='accuracy',
        seed=123
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if do_train:
        trainer.train()
        trainer.save_model()

    if do_eval:
        return trainer.evaluate()

    if do_predict:
        return trainer.evaluate(tokenized_dataset['test'])


def read_dataset():
    train_file = os.path.join(INTENT_DATASET_DIR, 'train_23.csv')
    dev_file = os.path.join(INTENT_DATASET_DIR, 'dev_23.csv')
    test_file = os.path.join(INTENT_DATASET_DIR, 'test_23.csv')

    def read_file(file):
        df = pd.read_csv(file)
        df = df[['sentence', 'label_idx']]
        df = df.rename(columns={'sentence': 'text', 'label_idx': 'label'})
        return Dataset.from_pandas(df)

    dataset = DatasetDict()
    dataset['train'] = read_file(train_file)
    dataset['validation'] = read_file(dev_file)
    dataset['test'] = read_file(test_file)

    return dataset


def main():
    if use_local_dataset:
        dataset = read_dataset()
    else:
        from upload import dataset_args
        dataset_path = dataset_args['repo_id']
        dataset = load_dataset(dataset_path)

    n_labels = max(dataset['train']['label']) + 1

    # train and eval on the validation set
    eval_result = run_trainer(dataset, n_labels, do_train=True, do_eval=True)
    print(f'Accuracy on the validation set: {eval_result["eval_accuracy"]}')

    # eval on the test set
    eval_result = run_trainer(dataset, n_labels, do_predict=True)
    print(f'Accuracy on the test set: {eval_result["eval_accuracy"]}')


if __name__ == '__main__':
    main()
