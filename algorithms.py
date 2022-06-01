# (c) Copyright IBM Corporation 2020-2022
# SPDX-License-Identifier: Apache2.0

import os
from enum import IntEnum

from consts import PREDICTIONS_DIR, SILVERDATA_DIR


class Algorithm(IntEnum):
    ORACLE = 1,
    ORACLE_PREDICT = 2,
    SIB = 3,
    KMEANS = 4,
    KPA = 5,
    RBC = 6,


titles = {
    Algorithm.ORACLE:           'Oracle',
    Algorithm.ORACLE_PREDICT:   'Oracle-test',
    Algorithm.SIB:              'sIB',
    Algorithm.KMEANS:           'KMeans',
    Algorithm.KPA:              'KPA',
    Algorithm.RBC:              'RBC',
}

paths = {
    Algorithm.ORACLE:           SILVERDATA_DIR,
    Algorithm.ORACLE_PREDICT:   os.path.join(PREDICTIONS_DIR, 'oracle'),
    Algorithm.SIB:              os.path.join(PREDICTIONS_DIR, 'sib'),
    Algorithm.KPA:              os.path.join(PREDICTIONS_DIR, 'kpa'),
    Algorithm.KMEANS:           os.path.join(PREDICTIONS_DIR, 'kmeans'),
    Algorithm.RBC:              os.path.join(PREDICTIONS_DIR, 'rbc'),
}
