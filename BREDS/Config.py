#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import fileinput
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk import WordNetLemmatizer
from Word2VecWrapper import Word2VecWrapper
from Common.Seed import Seed
from Common.ReVerb import Reverb


class Config(object):

    def __init__(self, config_file, seeds_file, negative_seeds, similarity, confidance):

        self.seed_tuples = set()
        self.negative_seed_tuples = set()
        self.vec_dim = 0
        self.e1_type = None
        self.e2_type = None
        self.stopwords = stopwords.words('english')
        self.lmtzr = WordNetLemmatizer()
        self.threshold_similarity = similarity
        self.instance_confidance = confidance
        self.word2vecwrapper = Word2VecWrapper()
        self.reverb = Reverb()

        for line in fileinput.input(config_file):
            if line.startswith("#") or len(line) == 1:
                continue

            if line.startswith("wUpdt"):
                self.wUpdt = float(line.split("=")[1])

            if line.startswith("wUnk"):
                self.wUnk = float(line.split("=")[1])

            if line.startswith("wNeg"):
                self.wNeg = float(line.split("=")[1])

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

            if line.startswith("embeddings"):
                self.embeddings = line.split("=")[1].strip()

            if line.startswith("single_vector"):
                self.single_vector = line.split("=")[1].strip()

            if line.startswith("similarity"):
                self.similarity = line.split("=")[1].strip()

            if line.startswith("word2vec_path"):
                self.word2vecmodelpath = line.split("=")[1].strip()

            if line.startswith("vector"):
                self.vector = line.split("=")[1].strip()

            if line.startswith("alpha"):
                self.alpha = float(line.split("=")[1])

            if line.startswith("beta"):
                self.beta = float(line.split("=")[1])

            if line.startswith("gamma"):
                self.gamma = float(line.split("=")[1])

            if line.startswith("semantic_drift"):
                self.semantic_drift = line.split("=")[1].strip()

        self.read_seeds(seeds_file)
        self.read_negative_seeds(negative_seeds)
        fileinput.close()

        print "Configuration parameters"
        print "========================"

        print "Relationship Representation"
        print "e1 type              :", self.e1_type
        print "e2 type              :", self.e2_type
        print "context window       :", self.context_window_size
        print "max tokens away      :", self.max_tokens_away
        print "min tokens away      :", self.min_tokens_away
        print "Word2Vec Model       :", self.word2vecmodelpath

        print "\nVectors"
        print "embeddings type      :", self.embeddings
        print "alpha                :", self.alpha
        print "beta                 :", self.beta
        print "gamma                :", self.gamma

        print "\nSeeds:"
        print "positive seeds       :", len(self.seed_tuples)
        print "negative seeds       :", len(self.negative_seed_tuples)
        print "negative seeds wNeg  :", self.wNeg
        print "unknown seeds wUnk   :", self.wUnk

        print "\nParameters and Thresholds"
        print "threshold_similarity :", self.threshold_similarity
        print "instance confidence  :", self.instance_confidance
        print "min_pattern_support  :", self.min_pattern_support
        print "iterations           :", self.number_iterations
        print "iteration wUpdt      :", self.wUpdt
        print "semantic drift filter:", self.semantic_drift
        print "\n"
        print "Loading word2vec model ...\n"
        self.word2vec = Word2Vec.load_word2vec_format(self.word2vecmodelpath, binary=True)
        self.vec_dim = self.word2vec.layer1_size

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

    def read_negative_seeds(self, negative_seeds):
        for line in fileinput.input(negative_seeds):
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
                    self.negative_seed_tuples.add(seed)