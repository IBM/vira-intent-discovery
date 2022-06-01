# (c) Copyright IBM Corporation 2020-2022
# SPDX-License-Identifier: Apache2.0

import sys

import pandas as pd

from consts import SNAPSHOT_FILE, SNAPSHOT_SPLIT_FILE, SNAPSHOT_TRAIN_FILE, SNAPSHOT_TEST_FILE


MASKED_TOKENS = {'PERSON', 'DATE', 'LOCATION', 'EMAIL', 'PHONE_NUMBER', 'TOXIC'}


def filter_all_data(row):
    text = row['text']
    alpha_numeric_chars = len([c for c in text if c.isalnum() or c == ' '])
    non_masked_tokens_count = len([w for w in text.split() if w not in MASKED_TOKENS])
    return 0 < len(text) < 250 and alpha_numeric_chars / len(text) > 0.75 and not row[
        'is_feedback'] and non_masked_tokens_count > 1


def filter_oracle_train(row):
    dialog_act = row['dialog_act']
    return dialog_act.lower() not in ('negative_reaction', 'appreciation', 'farewell', 'close_discussion', 'agreement',
                                      'disagreement', 'positive_reaction', 'change_subject', 'no_other_concern',
                                      'human_or_bot', 'clarify', 'greeting') and not row['is_profanity']


def main(split_mode):

    date_interval = "1MS"
    date_format = "%Y-%m-01"

    converters = {'dialog_id': int, 'message_id': int, 'side': str,
                  'text': str, 'dialog_act': str, 'intent': str, 'thumbs_up_down': int}

    df = pd.read_csv(SNAPSHOT_FILE, converters=converters, parse_dates=['date'])

    user_dates = df[df['side'] == 'user']['date']

    start_date = user_dates.iloc[0].strftime(date_format)
    end_date = user_dates.iloc[-1].strftime(date_format)

    date_list = pd.date_range(start=start_date, end=end_date, freq=date_interval).to_pydatetime().tolist()
    slot_list = [(date_list[i], date_list[i+1]) for i in range(len(date_list)-1)]

    def assign_to_slot(dates):
        result = []
        d1 = d2 = None
        for d in dates:
            match = False
            for d1, d2 in slot_list:
                if d1 <= d < d2:
                    match = True
                    break
            d = d1 if match else d2
            result.append(d.strftime(date_format))
        return result

    # limit to the user turns
    df = df[df['side'] == 'user']

    # add a slot column
    df = df.assign(slot=lambda x: assign_to_slot(x['date']))

    # filter unuseful texts
    df = df[df.apply(filter_all_data, axis=1)]

    if split_mode:
        # split to train and test
        df_train = df.groupby('slot').sample(frac=0.5, random_state=1024)
        df_test = df.drop(df_train.index)
        df_split = pd.concat([
            pd.DataFrame(data=['train'] * (len(df_train)), index=df_train.index, columns=['set']),
            pd.DataFrame(data=['test'] * (len(df_test)), index=df_test.index, columns=['set']),
        ]).sort_index()
        df_split.to_csv(SNAPSHOT_SPLIT_FILE)
    else:
        df_split = pd.read_csv(SNAPSHOT_SPLIT_FILE, index_col=0)
        if not all(df_split.index == df.index):
            raise ValueError("Snapshot data and split and incompatible")
        df_train = df[df_split['set'] == 'train']
        df_test = df[df_split['set'] == 'test']

    # filter the oracle data further
    df_train = df_train[df_train.apply(filter_oracle_train,  axis=1)]

    # keep only the columns needed for evaluation
    df_train = df_train[['text', 'slot']]
    df_test = df_test[['text', 'slot']]

    # save to csv
    df_train.to_csv(SNAPSHOT_TRAIN_FILE, index_label='id')
    df_test.to_csv(SNAPSHOT_TEST_FILE, index_label='id')


def read_data(set_name):
    set_file = {
        'train': SNAPSHOT_TRAIN_FILE,
        'test': SNAPSHOT_TEST_FILE
    }
    df = pd.read_csv(set_file[set_name], converters={'text': str})
    return df.groupby('slot').agg(texts=('text', list), ids=('id', list)).reset_index()


if __name__ == '__main__':
    split_param = len(sys.argv) == 2 and sys.argv[1] == '-split'
    main(split_param)
