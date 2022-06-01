# (c) Copyright IBM Corporation 2020-2022
# SPDX-License-Identifier: Apache2.0

import os

import numpy as np
import pandas as pd
from transformers import AutoTokenizer,  AutoModelForSequenceClassification, TextClassificationPipeline


use_local_model = False

if use_local_model:
    from consts import INTENT_MODEL_DIR
    model_path = INTENT_MODEL_DIR
else:
    from upload import model_args as hf_model_args
    model_path = hf_model_args['repo_url']

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
df = pd.read_csv(os.path.join('resources', 'oracle_supported_intents.csv'))
model_trained_intents = df['intent'].apply(str.strip).to_list()


def get_model_predictions(candidates):
    results = pipeline(candidates)
    intent_scores = np.array([[label['score'] for label in result] for result in results])
    intent_ids = np.argmax(intent_scores, axis=1)
    intent_scores = np.max(intent_scores, axis=1)
    model_intents = [model_trained_intents[intent_id] for intent_id in intent_ids]
    return model_intents, intent_scores.tolist()


if __name__ == '__main__':
    intents, scores = get_model_predictions(['i love it', 'i hate it'])
    print(intents, scores)
