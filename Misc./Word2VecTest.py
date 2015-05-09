#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from nltk import PunktWordTokenizer
from gensim.models import Word2Vec
from numpy.linalg import norm

__author__ = 'dsbatista'


def pattern2vector(tokens, word2vec, avg = False):
    pattern_vector = np.zeros(word2vec.layer1_size)
    n_words = 0
    if len(tokens) > 1:
        for t in tokens:
            try:
                vector = word2vec[t.strip()]
                pattern_vector = np.add(pattern_vector,vector)
                n_words += 1
            except KeyError:
                continue
        if avg is True:
            pattern_vector = np.divide(pattern_vector,n_words)
    elif len(tokens) == 1:
        try:
            pattern_vector = word2vec[tokens[0].strip()]
        except KeyError:
            pass
    return pattern_vector


def main():
    print "Loading word2vec model ...\n"
    word2vecmodelpath = "/home/dsbatista/gigaword/word2vec/afp_apw_xing200.bin"
    word2vec = Word2Vec.load_word2vec_format(word2vecmodelpath, binary=True)
    print "Dimensions", word2vec.layer1_size

    pattern_1 = 'founder and ceo'
    pattern_2 = 'co-founder and chairman'
    pattern_3 = 'founder and chief'
    pattern_4 = 'founder and chief executive'
    pattern_5 = 'founder and chief executive officer'
    pattern_6 = 'co-founder and former chairman'
    pattern_7 = 'founder and hedge fund operator'
    pattern_8 = 'founder and owner'
    pattern_9 = 'founder and philanthropist'

    tokens_1 = PunktWordTokenizer().tokenize(pattern_1)
    tokens_2 = PunktWordTokenizer().tokenize(pattern_4)
    print "vec1", tokens_1
    print "vec2", tokens_2

    p1 = pattern2vector(tokens_1, word2vec, False)
    p2 = pattern2vector(tokens_2, word2vec, False)
    print "\nSUM"
    print "dot(vec1,vec2)", np.dot(p1,p2)
    print "norm(p1)", norm(p1)
    print "norm(p2)", norm(p2)
    print "dot((norm)vec1,norm(vec2))", np.dot(norm(p1), norm(p2))
    print "cosine(vec1,vec2)", np.divide(np.dot(p1, p2), np.dot(norm(p1), norm(p2)))
    print "\n"
    print "AVG"
    p1 = pattern2vector(tokens_1, word2vec, True)
    p2 = pattern2vector(tokens_2, word2vec, True)
    print "dot(vec1,vec2)", np.dot(p1,p2)
    print "norm(p1)", norm(p1)
    print "norm(p2)", norm(p2)
    print "dot(norm(vec1),norm(vec2))", np.dot(norm(p1), norm(p2))
    print "cosine(vec1,vec2)", np.divide(np.dot(p1, p2), np.dot(norm(p1), norm(p2)))


if __name__ == "__main__":
    main()