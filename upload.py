# (c) Copyright IBM Corporation 2020-2022
# SPDX-License-Identifier: Apache2.0

import sys

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from trainer import base_model, read_dataset
from consts import INTENT_MODEL_DIR

model_args = {
    'repo_path_or_name': f'{INTENT_MODEL_DIR}_hf',
    'repo_url': f'ibm/{base_model}-vira-intents',
}

dataset_args = {
    'repo_id': f'ibm/vira-intents',
}


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Syntax: {sys.argv[0]} <auth-token>")
        exit(1)

    # add the auth token to the args
    model_args.update({'use_auth_token': sys.argv[1]})
    dataset_args.update({'token': sys.argv[1]})

    # upload the model and the tokenizer
    for mt in [AutoModelForSequenceClassification, AutoTokenizer]:
        mt.from_pretrained(INTENT_MODEL_DIR).push_to_hub(**model_args)

    # upload the dataset
    dataset = read_dataset()
    dataset.push_to_hub(**dataset_args)
