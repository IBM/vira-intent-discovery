# (c) Copyright IBM Corporation 2020-2022
# SPDX-License-Identifier: Apache2.0

from collections import defaultdict
import pandas as pd
import os

from algorithms import Algorithm, paths
from evaluation import matching_threshold
from prepare import read_data
from model import get_model_predictions


def create_silverdata():
	df = read_data('train')
	output_dir = os.path.join('resources', paths[Algorithm.ORACLE])
	run(df, output_dir, induce_silver_labels)


def predict(_, df, output_dir):
	run(df, output_dir, induce_silver_labels)


def run(df, output_dir, trim_func):
	data = []
	for _, row in df.iterrows():
		slot = row['slot']
		texts = row['texts']
		ids = row['ids']

		# get the model predictions
		intents, scores = get_model_predictions(texts)

		# organize the result
		intent_ids = map_intents_to_utterance_ids(intents, scores, ids)
		slot_data = []
		for intent, ids in intent_ids.items():
			for i in ids:
				slot_data.append({'slot': slot, 'intent': intent, 'id': i})

		# mark the largest intents that cover 80% of the data
		trim_func(slot_data)

		data.extend(slot_data)

	# save the results
	os.makedirs(output_dir, exist_ok=True)
	pd.DataFrame(data).to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)


def map_intents_to_utterance_ids(intents, scores, user_utterances_comment_ids):
	intent_to_utterance_ids = defaultdict(list)
	for intent, score, utterance_id in zip(intents, scores, user_utterances_comment_ids):
		if score >= matching_threshold:
			intent_to_utterance_ids[intent].append(utterance_id)
		else:
			intent_to_utterance_ids['none'].append(utterance_id)
	return intent_to_utterance_ids


def induce_silver_labels(slot_data):
	df = pd.DataFrame(slot_data)
	df = df.groupby('intent').agg(ids=('id', list), size=('id', 'count')).reset_index()
	df = df.sort_values(by='size', ascending=False)

	# get the ids in the 'none' cluster
	none_ids = df[df['intent'] == 'none']['ids'].iloc[0]

	# exclude the 'none' cluster for now
	df = df[df['intent'] != 'none']

	#  accumulate the coverage of data
	df['cum_size'] = df['size'].cumsum()
	df['cum_ratio'] = df['cum_size'] / df['size'].sum()

	# select the silver intents
	not_selected_df = df[~((df['cum_ratio'] <= 0.8) & (df['size'] >= 3))]

	# convert the intents that are not selected to none
	none_ids.extend(not_selected_df['ids'].agg(sum))
	for x in slot_data:
		if x['id'] in none_ids:
			x.update({'intent': 'none'})


if __name__ == "__main__":
	create_silverdata()
