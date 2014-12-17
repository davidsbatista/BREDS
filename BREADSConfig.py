#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fileinput
import sys
from Sentence import Sentence
from Tuple import Tuple
from Word2VecWrapper import Word2VecWrapper

__author__ = 'dsbatista'


class BREADSConfig(object):

    def __init__(self, config_file):

        self.seed_tuples = list()

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

            if line.startsWith("threshold_similarity"):
                self.threshold_similarity = float(line.split("=")[1])

            if line.startsWith("instance_confidance"):
                self.instance_confidance = float(line.split("=")[1])

            if line.startsWith("single_vector"):
                self.single_vector = line.split("=")[1]

            if line.startsWith("similarity"):
                self.similarity = line.split("=")[1]

            if line.startsWith("word2vec_path"):
                self.Word2VecModelPath = line.split("=")[1]

        self.word2vec = Word2VecWrapper(self.Word2VecModelPath)
        fileinput.close()

    @staticmethod
    def read_seeds(self, seeds_file):
        for line in fileinput.input((seeds_file)):
            if line.startswith("#") or len(line)==1:
                continue
            if line.startsWith("e1"):
                self.e1_type = line.split(":")[1]
            elif line.startsWith("e2"):
                self.e2_type = line.split(":")[1]
            else:
                e1 = line.split(";")[0]
                e2 = line.split(";")[1]
                seed = new Seed(e1, e2)
                seedTuples.add(seed)

