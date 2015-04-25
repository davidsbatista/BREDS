#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'dsbatista'
__email__ = "dsbatista@inesc-id.pt"

import random
import fileinput
import sys

from nltk.corpus import stopwords
from numpy.linalg import norm
from numpy import dot, array, zeros
from gensim import utils, matutils
from gensim.models import Word2Vec

VECTOR_DIM = 200


def read_patterns(data):
    patterns = list()
    for line in fileinput.input(data):
        pos_tags = line.split('Pos-tags:')[1].strip()
        tokens = line.split('Pos-tags:')[0].split('pattern:')[1].strip()
        print tokens
        print pos_tags
    return patterns


def get_words_vector(pattern, model, include_stopwords):
    vector = []
    for word in pattern.split(' '):
        if include_stopwords is False:
            if word not in stopwords.words('english'):
                try:
                    vector.append(model[word])
                except KeyError:
                    pass
        else:
            try:
                vector.append(model[word])
            except KeyError:
                pass
    return vector


def sum_rep(v1, v2):
    vector1 = zeros(VECTOR_DIM)
    vector2 = zeros(VECTOR_DIM)
    for v in v1:
        vector1 += v
    for v in v2:
        vector2 += v
    # normalize vectors, divide by the norm
    vector1 /= norm(vector1)
    vector2 /= norm(vector2)
    return dot(vector1, vector2) / (norm(vector1) * norm(vector2))


def centroid_rep(v1, v2):
    """
    # centroid -  escolher o vector mais próximo da média
    :param v1:
    :param v2:
    :return:
    """
    vector1 = zeros(VECTOR_DIM)
    vector2 = zeros(VECTOR_DIM)
    for v in v1:
        vector1 += v
    for v in v2:
        vector2 += v
    vector1 /= float(len(v1))
    vector2 /= float(len(v2))

    # normalize vectors, divide by the norm
    vector1 /= norm(vector1)
    vector2 /= norm(vector2)

    return dot(vector1, vector2) / (norm(vector1) * norm(vector2))


def mean_rep(v1, v2):
    return dot(matutils.unitvec(array(v1).mean(axis=0)), matutils.unitvec(array(v2).mean(axis=0)))


def compare_patterns(pattern1, pattern2, model):
    # TODO: incluir proposições e stopwords?
    # TODO: proposições fazem sentido quando um verbo está presente
    # TODO: dar um peso maior aos verbos e nouns,

    # sum everything
    v1 = get_words_vector(pattern1, model, include_stopwords=True)
    v2 = get_words_vector(pattern2, model, include_stopwords=True)

    # sum everything except stop words
    v1_no_stopwords = get_words_vector(pattern1, model, include_stopwords=False)
    v2_no_stopwords = get_words_vector(pattern2, model, include_stopwords=False)

    # sum everything except adjectives
    # sum everything except stop words and adjectives
    # sum everything except auxiliary verbs stop words and adjectives

    #acc_everything = sum(v1, v2)
    #acc_no_stop_words = sum_rep(v1_no_stopwords, v2_no_stopwords)
    #print pattern1, pattern2, '\t', sum_rep(v1, v2), sum_rep(v1_no_stopwords, v2_no_stopwords)


def sample(patterns, n):
    samples = list()
    for i in range(0, n):
        e = random.choice(patterns)
        samples.append(e)
        del patterns[patterns.index(e)]
    return samples


def main():
    #word2vec_path=/home/dsbatista/word2vec-read-only/vectors.bin
    #word2vec_path=/home/dsbatista/GoogleNews-vectors-negative300.bin
    #word2vec_path=/home/dsbatista/gigaword/word2vec/afp_apw_vectors.bin
    #word2vec_path=/home/dsbatista/gigaword/word2vec/afp_apw_xing_vectors.bin
    #word2vec_path=/home/dsbatista/gigaword/word2vec/afp_apw_xing200.bin

    #model = Word2Vec.load_word2vec_format(sys.argv[1], binary=True)
    correct = read_patterns(sys.argv[2])
    incorrect = read_patterns(sys.argv[3])

    """
    correct_s = sample(correct, 10)
    incorrect_s = sample(incorrect, 10)

    for p1 in correct_s:
        for p2 in incorrect_s:
            compare_patterns(p1, p2, model)
    """

if __name__ == "__main__":
    main()