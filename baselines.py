# (c) Copyright IBM Corporation 2020-2022
# SPDX-License-Identifier: Apache2.0

import os
from enum import Enum

import pandas as pd

from algorithms import Algorithm, paths
from clustering import cluster_and_extract_intents
from evaluation import evaluate_algorithm, evaluate_oracle, compute_means, MeasureType, \
    measures_by_type
from prepare import read_data
from oracle import predict as oracle_predict


class RunStatus(Enum):
    Run = 0,
    Skip = 1


def generated_externally():
    pass


# baselines = algorithms bar the oracle
baselines = {
    Algorithm.ORACLE_PREDICT:   (oracle_predict,                RunStatus.Run),
    Algorithm.SIB:              (cluster_and_extract_intents,   RunStatus.Run),
    Algorithm.KMEANS:           (cluster_and_extract_intents,   RunStatus.Run),
    Algorithm.KPA:              (generated_externally,          RunStatus.Skip),
    Algorithm.RBC:              (generated_externally,          RunStatus.Skip),
}


def read_baseline_results(algorithm):
    run_results_file, run_results_means_file = create_result_filenames(algorithm)
    return pd.read_csv(run_results_file), pd.read_csv(run_results_means_file)


def create_result_filenames(algorithm):
    run_path = paths[algorithm]
    run_results_file = os.path.join(run_path, 'scores.csv')
    run_results_means_file = os.path.join(run_path, 'means.csv')
    return run_results_file, run_results_means_file


def read_baseline_raw_files(algorithm, path_suffix):
    # collect the data from all points (slots x algorithms)
    data = []
    oracle_algorithms = [Algorithm.ORACLE, Algorithm.ORACLE_PREDICT]
    alg_set = set([algorithm] + oracle_algorithms) if algorithm != Algorithm.ORACLE else {Algorithm.ORACLE}
    for alg in alg_set:
        directory = paths[alg]
        if path_suffix and alg not in oracle_algorithms:
            directory = os.path.join(directory, path_suffix)
        df = pd.read_csv(os.path.join(directory, 'predictions.csv'))
        df_slot_intent = df.groupby(['slot', 'intent']).agg(ids=('id', list)).reset_index()
        df2 = df_slot_intent.groupby('slot').agg(intents=('intent', list), ids=('ids', list)).reset_index()

        for _, row in df2.iterrows():
            data.append({
                'slot': row['slot'],
                'algorithm': alg,
                'intents': row['intents'],
                'cluster_size': [len(x) for x in row['ids']],
                'text_ids': row['ids'],
            })
    return pd.DataFrame(data)


def main():
    overwrite = True
    skip_generation = False

    # read the data
    df = read_data('test')

    # iterate over the baselines
    for algorithm, (baseline_generator, run_status) in baselines.items():
        run_results_file, run_results_means_file = create_result_filenames(algorithm)
        if run_status == RunStatus.Run and (not os.path.exists(run_results_file) or overwrite):

            # determine the output dir
            output_dir = paths[algorithm]
            os.makedirs(output_dir, exist_ok=True)

            # generate the baseline unless skipped
            if not skip_generation:
                baseline_generator(algorithm, df, output_dir)

            # load the results
            df_slot_intents = read_baseline_raw_files(algorithm, '')

            # evaluate against the oracle predict
            df_algorithm = evaluate_algorithm(df_slot_intents, algorithm)

            # compute the means for our measures, weighted by slot size
            df_algorithm_means = pd.concat([compute_means(df_algorithm, measures_by_type[measure_type])
                                            for measure_type in MeasureType], axis=1)
            df_algorithm_means = df_algorithm_means.loc[:, ~df_algorithm_means.columns.duplicated()]

            # save to disk
            df_algorithm.to_csv(run_results_file, index=False)
            df_algorithm_means.to_csv(run_results_means_file, index=False)

    # load oracle data
    df_slot_oracle = read_baseline_raw_files(Algorithm.ORACLE, None)

    # calculate oracle stats
    df_slot_oracle = df_slot_oracle.apply(evaluate_oracle(), axis=1)

    # save to disk
    oracle_results_file, oracle_results_file_mean = create_result_filenames(Algorithm.ORACLE)
    df_slot_oracle.to_csv(oracle_results_file, index=False)
    df_slot_oracle.to_csv(oracle_results_file_mean, index=False)  # dummy file


if __name__ == '__main__':
    main()
