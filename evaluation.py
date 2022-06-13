# (c) Copyright IBM Corporation 2020-2022
# SPDX-License-Identifier: Apache2.0

from collections import defaultdict
from enum import Enum

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, v_measure_score

from algorithms import Algorithm, titles
from model import get_model_predictions


intent_matching_model = 'oracle_supported_intents.csv'
matching_threshold = 0.296


class MeasureType(Enum):
    IntentCoverage = 1
    ClusteringQuality = 2


intent_coverage_measures = [
    ('recall',          'Coverage of intents discovered by Oracle'),
    ('precision',       'Precision of algorithm intent prediction'),
    ('coverage_f1',     'F1 Score (Intent Coverage)'),
    ('jensenshannon',   'Jensen-Shannon distance of intent distributions'),
]

clustering_quality_measures = [
    ('ari',             'Adjusted Rand-Index for intent clusters'),
    ('ami',             'Adjusted Mutual-Information for intent clusters'),
    ('quality_f1',      'F1 Score (Clustering Quality)'),
    ('v_measure',       'V-Measure for intent clusters')
]

measures_by_type = {
    MeasureType.IntentCoverage: intent_coverage_measures,
    MeasureType.ClusteringQuality: clustering_quality_measures,
}


def evaluate(df_oracle, df_oracle_predict):
    def process(row):
        slot = row['slot']
        algorithm_intents_to_cluster_size_dict = defaultdict(int)
        for intent, intent_size in zip(row['intents'], row['cluster_size']):
            algorithm_intents_to_cluster_size_dict[intent] += intent_size
        algorithm_intents = list(algorithm_intents_to_cluster_size_dict.keys())

        df_oracle_slot = df_oracle[df_oracle['slot'] == slot]
        oracle_intents = list(df_oracle_slot['intents'].iloc[0])
        oracle_intent_size = df_oracle_slot['cluster_size'].iloc[0]
        oracle_intents_to_cluster_size_dict = dict(zip(oracle_intents, oracle_intent_size))

        # removing "none" from precision / coverage calculations
        if 'none' in algorithm_intents:
            algorithm_intents.remove('none')
        oracle_intents.remove('none')

        row['num_intents_predicted'] = len(algorithm_intents)

        # calculating how much texts the intents cover (in pct. out of all texts)
        row['intent_to_none_ratio'] = sum([algorithm_intents_to_cluster_size_dict[k] for k in algorithm_intents]) / sum(
            algorithm_intents_to_cluster_size_dict.values())

        # run intent matching and calculate the coverage of oracle intents
        model_intents, scores = get_model_predictions(algorithm_intents)
        model_intents_scores = defaultdict(list)
        algorithm_intentss_scores = defaultdict(list)
        for model_intent, score, algorithm_intent in zip(model_intents, scores, algorithm_intents):
            model_intents_scores[model_intent].append((score, algorithm_intent))
            algorithm_intentss_scores[algorithm_intent].append((score, model_intent))
        oracle_intents_set = set(oracle_intents)

        # map every oracle intent to a list of pairs - (score, algorithm_intent) where
        # the score is above the threshold
        covered_oracle_intents = {model_intent: [(score, algorithm_intent)
                                                 for score, algorithm_intent in intent_scores
                                                 if score >= matching_threshold]
                                  for model_intent, intent_scores in model_intents_scores.items()
                                  if model_intent in oracle_intents_set}
        valid_algorithm_intents = [(algorithm_intent, score, model_intent)
                                   for model_intent, intent_scores in covered_oracle_intents.items()
                                   for score, algorithm_intent in intent_scores]
        covered_oracle_intents = {model_intent: max(intent_scores, key=lambda x: x[0])
                                  for model_intent, intent_scores in covered_oracle_intents.items()
                                  if len(intent_scores) > 0}
        row['covered_oracle_intents'] = [(model_intent, score, algorithm_intent)
                                         for model_intent, (score, algorithm_intent)
                                         in covered_oracle_intents.items()]
        row['recall'] = len(covered_oracle_intents) / len(oracle_intents)
        row['precision'] = len(valid_algorithm_intents) / len(algorithm_intents)
        row['coverage_f1'] = 2 * ((row['recall'] * row['precision']) /
                                  (row['recall'] + row['precision']))

        # updating the algorithm intent names with their target match
        # algorithm_intent is removed from dict if it is replaced by a model intent, or not matched at all
        for algorithm_intent, model_intent, score in zip(algorithm_intents, model_intents, scores):
            if algorithm_intent == model_intent and model_intent in oracle_intents_set:
                continue
            else:
                # if matched to oracle, we replace the algorithm intent with the oracle intent
                # else, we remove it and add its sum to "none"
                cluster_size_temp = algorithm_intents_to_cluster_size_dict[algorithm_intent]
                if score >= matching_threshold and model_intent in oracle_intents_set:
                    algorithm_intents_to_cluster_size_dict[model_intent] += cluster_size_temp
                else:
                    algorithm_intents_to_cluster_size_dict['none'] += cluster_size_temp
                algorithm_intents_to_cluster_size_dict.pop(algorithm_intent)

        total_num_sentences = sum(list(algorithm_intents_to_cluster_size_dict.values()))
        row['cluster_ratios'] = {(alg_intent, algorithm_intents_to_cluster_size_dict[alg_intent] / total_num_sentences)
                                 for alg_intent in algorithm_intents_to_cluster_size_dict.keys()}

        # updating the algorithm and cluster size dicts with empty clusters for the correlation calculation
        algorithm_intents_to_cluster_size_dict.update(
            {new_intent: 0 for new_intent in oracle_intents_set.difference(
                set(algorithm_intents_to_cluster_size_dict.keys()))})

        algorithm_vector = [x[1] for x in sorted(algorithm_intents_to_cluster_size_dict.items(), key=lambda y: y[0])]
        oracle_vector = [x[1] for x in sorted(oracle_intents_to_cluster_size_dict.items(), key=lambda y: y[0])]
        row['pearson'] = pearsonr(algorithm_vector, oracle_vector)[0]

        row['ordered_oracle_intents'] = oracle_intents
        row['ordered_algorithm_intents'] = [covered_oracle_intents[oracle_intent][1]
                                            if oracle_intent in covered_oracle_intents else ''
                                            for oracle_intent in oracle_intents]

        sum_algorithm_vector = sum(algorithm_vector)
        sum_oracle_vector = sum(oracle_vector)
        algorithm_vector = [s / sum_algorithm_vector for s in algorithm_vector]
        oracle_vector = [s / sum_oracle_vector for s in oracle_vector]
        row['jensenshannon'] = jensenshannon(algorithm_vector, oracle_vector)

        df_oracle_predict_slot = df_oracle_predict[df_oracle_predict['slot'] == slot]
        oracle_text_ids = df_oracle_predict_slot['text_ids'].iloc[0]
        oracle_labels = []
        text_id_to_index = {}
        for i, intent_text_ids in enumerate(oracle_text_ids):
            for text_id in intent_text_ids:
                oracle_labels.append(i)
                text_id_to_index[text_id] = len(oracle_labels) - 1
        oracle_labels = np.array(oracle_labels)

        algorithm_labels = np.empty_like(oracle_labels)
        algorithm_text_ids = row['text_ids']
        for i, intent_text_ids in enumerate(algorithm_text_ids):
            if len(intent_text_ids) > 0:
                for text_id in intent_text_ids:
                    index = text_id_to_index[text_id]
                    algorithm_labels[index] = i
        row['ari'] = adjusted_rand_score(oracle_labels, algorithm_labels)
        row['ami'] = adjusted_mutual_info_score(oracle_labels, algorithm_labels)
        row['v_measure'] = v_measure_score(oracle_labels, algorithm_labels)
        row['quality_f1'] = 2 * ((row['ari'] * row['ami']) /
                                 (row['ari'] + row['ami']))

        return row

    return process


def evaluate_oracle():
    def process(row):
        oracle_intents = row['intents']
        oracle_intent_size = row['cluster_size']
        oracle_intents_to_cluster_size_dict = dict(zip(oracle_intents, oracle_intent_size))

        row['num_intents_predicted'] = len(oracle_intents)
        row['intent_to_none_ratio'] = sum([oracle_intents_to_cluster_size_dict[k]
                                           for k in oracle_intents if k != 'none']) / sum(
            oracle_intent_size)

        return row

    return process


def evaluate_algorithm(df, algorithm):
    # split into oracle and non-oracle
    df_oracle = df[df['algorithm'] == Algorithm.ORACLE]
    df_algorithms = df[df['algorithm'] != Algorithm.ORACLE]
    df_oracle_predict = df[df['algorithm'] == Algorithm.ORACLE_PREDICT]

    # evaluate the non-oracle against the oracle.
    df_algorithms = df_algorithms[df_algorithms['algorithm'] == algorithm]
    df_algorithms = df_algorithms.apply(evaluate(df_oracle, df_oracle_predict), axis=1)

    # assign weight to every slot
    slot_sizes = df_oracle[['slot', 'intents']].set_index('slot')['intents'].apply(lambda x: len(x))
    slot_weights = slot_sizes / slot_sizes.to_numpy().sum()
    df_algorithms = df_algorithms.apply(assign_weights(slot_weights), axis=1)

    return df_algorithms


def check(ids1, ids2, algorithm1, algorithm2):
    ids1_diff = ids1 - ids2
    if len(ids1_diff) > 0:
        print(f'[{titles[algorithm1]}] has {len(ids1_diff)} id(s) '
              f'that [{titles[algorithm2]}] does not have:')
        print(list(ids1_diff)[:10])
        print(f'{len(ids2)} + {len(ids1_diff)} == {len(ids1)}')
        return True
    return False


def assign_weights(slot_weights):
    def process(row):
        row['weight'] = slot_weights[row['slot']]
        return row
    return process


def prepare_matched_intents(row):
    row['Oracle'] = row['ordered_oracle_intents'][0]
    for algorithm, ordered_algorithm_intents in zip(row['algorithms'],
                                                    row['ordered_algorithm_intents']):
        row[titles[algorithm]] = ordered_algorithm_intents
    return row


def get_intent_coverage(df_algorithms):
    df_intent_coverage = df_algorithms.groupby(['slot']).agg(
        algorithms=('algorithm', list),
        ordered_oracle_intents=('ordered_oracle_intents', list),
        ordered_algorithm_intents=('ordered_algorithm_intents', list)).reset_index(0)
    df_intent_coverage = df_intent_coverage.apply(prepare_matched_intents, axis=1)
    df_intent_coverage = df_intent_coverage.drop(['algorithms',
                                                  'ordered_algorithm_intents',
                                                  'ordered_oracle_intents'], axis=1)
    return df_intent_coverage


def weighted_mean(measures):
    def process(df):
        return pd.DataFrame([{measure: (df[measure] * df['weight']).sum()
                              for measure, _ in measures}])
    return process


def compute_means(df_algorithms, measures):
    df_algorithms_means = df_algorithms.groupby(['algorithm']).apply(weighted_mean(measures))
    df_algorithms_means = df_algorithms_means.droplevel(1).reset_index()
    return df_algorithms_means
