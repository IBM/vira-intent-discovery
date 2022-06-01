# (c) Copyright IBM Corporation 2020-2022
# SPDX-License-Identifier: Apache2.0

import os
import pickle
from enum import Enum

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sib import SIB
from sib.clustering_utils import get_key_terms, get_clusters, get_key_texts
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

from algorithms import Algorithm
from consts import VECTORS_DIR
from sib_utils import text_processor


class ClusteringAlgorithm(Enum):
    sIB = 1,
    KMeans = 2,


clustering_methods = {
    Algorithm.SIB: ClusteringAlgorithm.sIB,
    Algorithm.KMEANS: ClusteringAlgorithm.KMeans,
}


class SBertVectorizer:

    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def fit_transform(self, texts):
        return self.encoder.encode(texts)


class ClusterAnalysis:

    def __init__(self, algorithm, n_clusters, run_name, ngram_range=(1, 3),
                 max_features=10000, p_value_threshold=0.05, top_k_key_terms=10,
                 top_k_intents=1, use_stemmer=True):
        self.algorithm = algorithm
        self.run_name = run_name
        self.algorithms = {
            ClusteringAlgorithm.sIB: lambda k: SIB(n_clusters=k, random_state=1024),
            ClusteringAlgorithm.KMeans: lambda k: KMeans(n_clusters=k, random_state=1024),
        }
        self.algorithm_preprocessors = {
            ClusteringAlgorithm.sIB: lambda x: self.process_texts_bow(x),
            ClusteringAlgorithm.KMeans: lambda x: x,
        }
        self.algorithm_vectorizers = {
            ClusteringAlgorithm.sIB: lambda: CountVectorizer(ngram_range=ngram_range,
                                                             max_features=max_features,
                                                             min_df=5, max_df=0.5),
            ClusteringAlgorithm.KMeans: lambda: SBertVectorizer(),
        }
        self.ngram_range = ngram_range
        self.n_clusters = n_clusters
        self.p_value_threshold = p_value_threshold
        self.top_k_key_terms = top_k_key_terms
        self.top_k_intents = top_k_intents
        self.use_stemmer = use_stemmer
        self.sib_empty_text_threshold = 1

    def process_texts_bow(self, texts):
        return [text_processor(text, self.use_stemmer) for text in texts]

    def extract_intents(self, texts, ids):
        vectors_path = os.path.join(VECTORS_DIR, f"{self.algorithm.name.upper()}_{self.run_name}.pkl")
        empty_ids = []
        if not os.path.exists(vectors_path):
            texts_to_vectorize = self.algorithm_preprocessors[self.algorithm](texts)
            data_vectors = self.algorithm_vectorizers[self.algorithm]().fit_transform(texts_to_vectorize)
            os.makedirs(VECTORS_DIR, exist_ok=True)
            with open(vectors_path, 'wb') as f:
                pickle.dump(data_vectors, f)
        else:
            with open(vectors_path, 'rb') as f:
                data_vectors = pickle.load(f)

        n_clusters = self.n_clusters

        if len(empty_ids) > 0:
            n_clusters -= 1

        algorithm = self.algorithms[self.algorithm](n_clusters)
        algorithm.fit(data_vectors)
        labels = algorithm.labels_

        if len(empty_ids) > 0:
            for i in empty_ids:
                labels = np.insert(labels, i, n_clusters)
            n_clusters += 1

        clusters = get_clusters(labels)
        intents = []
        intent_size = np.bincount(labels, minlength=n_clusters).tolist()
        intent_ids = []

        processed_texts = self.process_texts_bow(texts)

        vectorizer = CountVectorizer(ngram_range=self.ngram_range)
        vectors = vectorizer.fit_transform(processed_texts)

        key_terms = get_key_terms(vectors, clusters, self.p_value_threshold, self.top_k_key_terms)
        key_texts = get_key_texts(vectors, clusters, key_terms, self.top_k_intents)

        for cluster_id, cluster_key_text_ids in key_texts.items():
            selected_texts = [texts[i] for i in cluster_key_text_ids]
            if len(selected_texts) == 1:
                selected_texts = selected_texts[0]
            intents.append(selected_texts)
            intent_ids.append([ids[i] for i in clusters[cluster_id]])

        return intents, intent_size, intent_ids


def cluster_and_extract_intents(algorithm: Algorithm, df: pd.DataFrame, output_dir: str) -> None:
    clustering_algorithm = clustering_methods[algorithm]
    data = []
    for _, row in df.iterrows():
        slot = row['slot']
        user_utterances = row['texts']
        ids = row['ids']
        n_clusters = int(np.round(np.sqrt(len(user_utterances))))
        short_texts = []
        short_ids = []
        long_texts = []
        long_ids = []
        for text, text_id in zip(user_utterances, ids):
            if len(text.split()) < 5:
                short_texts.append(text)
                short_ids.append(text_id)
            else:
                long_texts.append(text)
                long_ids.append(text_id)

        cluster_analysis = ClusterAnalysis(clustering_algorithm, n_clusters - 1, slot)
        intents, intent_size, intent_ids = cluster_analysis.extract_intents(long_texts, long_ids)
        intents.append('none')
        intent_size.append(len(short_texts))
        intent_ids.append(short_ids)
        for intent, text_ids in zip(intents, intent_ids):
            for text_id in text_ids:
                data.append({'slot': slot, 'intent': intent, 'id': text_id})
    pd.DataFrame(data).to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
