#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim import matutils
from gensim.models import Word2Vec
from numpy import dot

__author__ = 'dsbatista'


def main():
    print "Loading word2vec model ...\n"
    #word2vecmodelpath = "../afp_apw_vectors.bin"
    word2vecmodelpath = "../vectors.bin"
    word2vec = Word2Vec.load_word2vec_format(word2vecmodelpath, binary=True)

    vec1 = word2vec['headquarters']
    vec2 = word2vec['headquarters']

    score = dot(matutils.unitvec(vec1), matutils.unitvec(vec2))
    print score

if __name__ == "__main__":
    main()