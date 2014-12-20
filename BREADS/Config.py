#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import fileinput
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from Seed import Seed


class Config(object):

    def __init__(self, config_file, seeds_file):

        self.seed_tuples = set()
        self.vec_dim = 0
        self.e1_type = None
        self.e2_type = None
        self.stopwords = stopwords.words('english')

        for line in fileinput.input(config_file):
            if line.startswith("#") or len(line) == 1:
                continue

            if line.startswith("wUpdt"):
                self.wUpdt = float(line.split("=")[1])

            if line.startswith("number_iterations"):
                self.number_iterations = int(line.split("=")[1])

            if line.startswith("use_RlogF"):
                self.use_RlogF = bool(line.split("=")[1])

            if line.startswith("min_pattern_support"):
                self.min_pattern_support = int(line.split("=")[1])

            if line.startswith("max_tokens_away"):
                self.max_tokens_away = int(line.split("=")[1])

            if line.startswith("min_tokens_away"):
                self.min_tokens_away = int(line.split("=")[1])

            if line.startswith("context_window_size"):
                self.context_window_size = int(line.split("=")[1])

            if line.startswith("threshold_similarity"):
                self.threshold_similarity = float(line.split("=")[1])

            if line.startswith("instance_confidance"):
                self.instance_confidance = float(line.split("=")[1])

            if line.startswith("single_vector"):
                self.single_vector = line.split("=")[1].strip()

            if line.startswith("similarity"):
                self.similarity = line.split("=")[1].strip()

            if line.startswith("word2vec_path"):
                self.word2vecmodelpath = line.split("=")[1].strip()

        print "Loading word2vec model ...\n"
        self.word2vec = Word2Vec.load_word2vec_format(self.word2vecmodelpath, binary=True)
        self.vec_dim = 200
        self.read_seeds(seeds_file)
        fileinput.close()

        print "Configuration parameters"
        print "========================"
        print "similartiy:", self.similarity
        print "e1 type:", self.e1_type
        print "e2 type:", self.e2_type
        print "instance confience:", self.instance_confidance
        print "min_pattern_support", self.min_pattern_support
        print "iterations: ", self.number_iterations
        print "threshold_similarity: ", self.threshold_similarity
        print "iteration wUpdt:", self.wUpdt
        print "similarity:", self.similarity
        print "word2vecmodel: ", self.word2vecmodelpath
        print "context window:", self.context_window_size
        print "max tokens away:", self.max_tokens_away
        print "min tokens away:", self.min_tokens_away

    def read_seeds(self, seeds_file):
        for line in fileinput.input(seeds_file):
            if line.startswith("#") or len(line) == 1:
                continue
            if line.startswith("e1"):
                self.e1_type = line.split(":")[1].strip()
            elif line.startswith("e2"):
                self.e2_type = line.split(":")[1].strip()
            else:
                e1 = line.split(";")[0].strip()
                e2 = line.split(";")[1].strip()
                seed = Seed(e1, e2)
                self.seed_tuples.add(seed)