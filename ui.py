# (c) Copyright IBM Corporation 2020-2022
# SPDX-License-Identifier: Apache2.0

from ast import literal_eval

import pandas as pd
import streamlit as st

from algorithms import Algorithm, titles
from baselines import read_baseline_results
from evaluation import get_intent_coverage, measures_by_type, MeasureType


# NOTE: This must be the first command in your app, and must be set only once
st.set_page_config(layout="wide")


measures_desc = {
    MeasureType.IntentCoverage: ('coverage_f1', 'Intent Coverage'),
    MeasureType.ClusteringQuality: ('quality_f1', 'Clustering Quality')
}


def get_means(algorithm_results):
    df_means = pd.concat([df.reset_index() for _, df in algorithm_results.values()], axis=0)
    df_means.drop(['index'], axis=1, inplace=True)
    df_means.index = [titles[algorithm] for algorithm in df_means['algorithm']]
    return df_means


def main():

    # read the baseline results
    algorithm_results_with_oracle = {algorithm: read_baseline_results(algorithm) for algorithm in Algorithm}

    algorithm_result = {algorithm: algorithm_results_with_oracle[algorithm] for algorithm in Algorithm if
                        algorithm != Algorithm.ORACLE}

    # prepare the table of n_intents_predicted per algorithm
    df = pd.concat([df_algorithm[['slot', 'num_intents_predicted']].set_index('slot')
                    .rename(columns={'num_intents_predicted': titles[algorithm]})
                    for algorithm, (df_algorithm, _) in algorithm_results_with_oracle.items()], axis=1)
    st.header('Number of intents predicted')
    st.dataframe(df)

    st.header(f"Means")
    df_means = get_means(algorithm_result)

    cols = st.columns(2)
    for measure_type, col in zip(MeasureType, cols):
        measure_key, measure_title = measures_desc[measure_type]
        df_measures = df_means[[name for name, desc in measures_by_type[measure_type]]]
        df_measures = df_measures.sort_values(by=measure_key, ascending=False)
        with col:
            st.header(measure_title)
            st.dataframe(df_measures)

    st.header(f"Analysis")

    # obtain a dict of dataframes, each describing the coverage and quality scores
    # on each slot by one algorithm.
    df_algorithm = {algorithm: df_algorithm
                    for algorithm, (df_algorithm, _) in algorithm_result.items()}

    for measure_type, measures in measures_by_type.items():
        for measure, title in measures:
            df = pd.concat([df[['slot', measure]].set_index('slot')
                           .rename(columns={measure: titles[algorithm]})
                            for algorithm, df in df_algorithm.items()], axis=1)
            if measure_type == MeasureType.ClusteringQuality:
                df = df.drop([titles[Algorithm.ORACLE_PREDICT]], axis=1)
            st.header(title)
            st.line_chart(df)

    df_intent_coverage = get_intent_coverage(pd.concat([df for df in df_algorithm.values()], ignore_index=True))
    st.header('Intent Coverage')
    for name, df_slot in df_intent_coverage.groupby('slot'):
        df_slot = df_slot.drop('slot', axis=1)
        for col in df_slot.columns:
            df_slot[col] = df_slot[col].apply(literal_eval)
        df_slot = df_slot.explode(df_slot.columns.to_list())
        df_slot.set_index('Oracle', inplace=True)
        with st.expander(name):
            st.table(df_slot)

    df = pd.concat([df_algorithm[['slot', 'intent_to_none_ratio']].set_index('slot')
                   .rename(columns={'intent_to_none_ratio': titles[algorithm]})
                    for algorithm, (df_algorithm, _) in algorithm_results_with_oracle.items()], axis=1)
    st.header('Text coverage of intents predicted')
    st.dataframe(df)

    intent = st.selectbox('Select intent to show progress', [
        'Can children get the vaccine?',
        'Will I need a booster shot?',
        'How effective is the vaccine against the Omicron variant?',
        'Can I still get COVID even after being vaccinated?',
        'Does the vaccine impact pregnancy?',
        'Do vaccines work against the mutated strains of COVID-19?',
        'Does the vaccine prevent transmission?',
        'I\'m not sure the vaccine is effective enough',
        'Can my kids go back to school without a vaccine?',
        'There are many reports of severe side effects '
        'or deaths from the vaccine'])
    df = pd.concat([df[['slot', 'cluster_ratios']].set_index('slot')
                   .rename(columns={'cluster_ratios': titles[algorithm]}).apply(set_intent, args=(intent,))
                    for algorithm, df in df_algorithm.items()], axis=1)
    st.header(f'intent progress : {intent}')
    st.line_chart(df)


def set_intent(row, intent):
    cluster_ratio = [[x[1] for x in list(literal_eval(r)) if x[0] == intent] for r in row]
    return [c[0] if len(c) > 0 else 0 for c in cluster_ratio]


if __name__ == '__main__':
    main()
