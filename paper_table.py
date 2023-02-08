# (c) Copyright IBM Corporation 2020-2022
# SPDX-License-Identifier: Apache2.0

import os

import numpy as np
import pandas as pd

from algorithms import paths, titles
from baselines import baselines
from consts import PAPER_DIR
from evaluation import intent_coverage_measures, clustering_quality_measures

# path = paths[Algorithm.ORACLE_PREDICT]
# predictions_df = pd.read_csv(os.path.join(path, "predictions.csv"))
# weights = predictions_df.groupby('slot').count()['id'].to_numpy()

mean_results = []
stdev_results = []

measures = [measure for measure, desc in
            intent_coverage_measures +
            clustering_quality_measures]

for baseline in baselines.keys():
    path = paths[baseline]
    scores_df = pd.read_csv(os.path.join(path, "scores.csv"))

    weights = scores_df['weight'].to_numpy()

    means = np.average(scores_df[measures], weights=weights, axis=0)
    variations = np.average((scores_df[measures]-means)**2, weights=weights, axis=0)
    stdevs = np.sqrt(variations)

    mean_results.append(pd.Series(means, index=measures, name=titles[baseline]))
    stdev_results.append(pd.Series(stdevs, index=measures, name=titles[baseline]))

os.makedirs(PAPER_DIR, exist_ok=True)

for results, filename in zip([mean_results, stdev_results],
                             ['means.csv', 'stdevs.csv']):
    pd.concat(results, axis=1).transpose().to_csv(
        os.path.join(PAPER_DIR, filename), float_format='%.2f')
