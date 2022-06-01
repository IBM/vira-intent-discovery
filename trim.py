# (c) Copyright IBM Corporation 2020-2022
# SPDX-License-Identifier: Apache2.0

import os

import numpy as np
import pandas as pd

from algorithms import Algorithm, paths


def convert_to_single_none_row_df(none_df):
    agg = none_df.agg({'size': 'sum', 'ids': 'sum'})
    return pd.DataFrame({'intent': ['none'], 'size': [agg['size']], 'ids': [agg['ids']]})


def trim_sqrt_n_for_slot_data(slot_data):
    df = pd.DataFrame(slot_data)
    df = df.groupby('intent').agg(ids=('id', list), size=('id', 'count')).reset_index()
    df = df.sort_values(by='size', ascending=False)

    # calculate number of sqrt_n clusters
    num_sentences = sum(df['size'].tolist())
    sqrt_n_num_sentences = int(np.round(np.sqrt(num_sentences)))

    # split data
    top_df = df.iloc[:sqrt_n_num_sentences]
    bottom_df = df.iloc[sqrt_n_num_sentences:]

    if 'none' in top_df['intent'].tolist():
        bottom_df = bottom_df.append(top_df.loc[top_df['intent'] == 'none', :])

    # convert all clusters below cut-off point to a single none cluster
    bottom_df = convert_to_single_none_row_df(bottom_df)
    none_ids = bottom_df['ids'].tolist()[0]

    # convert the intent to none for the new ids
    for x in slot_data:
        if x['id'] in none_ids:
            x.update({'intent': 'none'})


def trim_results_by_sqrt_n(algorithm):
    directory = paths[algorithm]
    df = pd.read_csv(os.path.join(directory, 'predictions_original.csv'))
    df_slot_intent = df.groupby(['slot', 'intent']).agg(ids=('id', list)).reset_index()
    df = df_slot_intent.groupby('slot').agg(intents=('intent', list), ids=('ids', list)).reset_index()

    data = []

    for _, row in df.iterrows():
        slot = row['slot']
        intents = row['intents']
        ids = row['ids']

        slot_data = []
        for intent, id_list in zip(intents, ids):
            for i in id_list:
                slot_data.append({'slot': slot, 'intent': intent, 'id': i})

        trim_sqrt_n_for_slot_data(slot_data)

        data.extend(slot_data)

    # save the results
    os.makedirs(directory, exist_ok=True)
    pd.DataFrame(data).to_csv(os.path.join(directory, 'predictions.csv'), index=False)


if __name__ == "__main__":
    trim_results_by_sqrt_n(Algorithm.RBC)
