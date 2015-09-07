#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import uuid
from numpy import zeros
from math import log


class Pattern(object):

    def __init__(self, config, t=None):
        self.id = uuid.uuid4()
        self.single_vector = zeros(config.vec_dim)
        self.positive = 0
        self.negative = 0
        self.unknown = 0
        self.confidence = 0
        self.tuples = set()
        self.vectors = list()
        self.patterns_words = set()
        if tuple is not None:
            self.tuples.add(t)

    def __hash__(self):
        return hash((self.patterns_words, self.tuples))

    def __eq__(self, other):
        return (self.tuples, self.patterns_words) == (other.tuples, other.patterns_words)

    def __str__(self):
        return " | ".join([p for p in self.patterns_words]).encode("utf8")

    def __cmp__(self, other):
        if other.confidence > self.confidence:
            return -1
        elif other.confidence < self.confidence:
            return 1
        else:
            return 0

    def update_confidence_2003(self, config):
        if self.positive > 0:
            self.confidence = log(float(self.positive), 2) * (float(self.positive) / float(self.positive + self.unknown * config.wUnk + self.negative * config.wNeg))
        elif self.positive == 0:
            self.confidence = 0

    def update_confidence(self):
        if self.positive > 0 or self.negative > 0:
            self.confidence = float(self.positive) / float(self.positive + self.negative)

    def add_tuple(self, t):
        self.tuples.add(t)

    def merge_patterns(self):
        for t in self.tuples:
            for p in t.patterns_words:
                self.patterns_words.add(p)

    def update_selectivity(self, t, config):
        #TODO: usar seeds em que e1, faz match com varios e2, e alterar a forma
        for s in config.seed_tuples:
            if s.e1 == t.e1 or s.e1.strip() == t.e1.strip():
                if s.e2 == t.e2.strip() or s.e2.strip() == t.e2.strip():
                    self.positive += 1
                else:
                    self.negative += 1
            else:
                for n in config.negative_seed_tuples:
                    if n.e1 == t.e1 or n.e1.strip() == t.e1.strip():
                        if n.e2 == t.e2.strip() or n.e2.strip() == t.e2.strip():
                            self.negative += 1
                self.unknown += 1

        #self.update_confidence()
        self.update_confidence_2003(config)
