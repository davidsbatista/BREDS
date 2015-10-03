#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import uuid


class Pattern(object):

    def __init__(self, t=None):
        self.id = uuid.uuid4()
        self.positive = 0
        self.negative = 0
        self.unknown = 0
        self.confidence = 0
        self.bet_uniques = set()
        self.tuples = set()
        self.tuples_vectors_uniques = set()
        if tuple is not None:
            self.tuples.add(t)

    def __eq__(self, other):
        return self.tuples == other.tuples

    def __cmp__(self, other):
        if other.confidence > self.confidence:
            return -1
        elif other.confidence < self.confidence:
            return 1
        else:
            return 0

    def update_confidence(self, config):
        if self.positive > 0:
            self.confidence = (float(self.positive) / float(self.positive +
                                                            self.unknown * config.wUnk +
                                                            self.negative * config.wNeg))
        elif self.positive == 0:
            self.confidence = 0

    def add_tuple(self, t):
        self.tuples.add(t)

    # put all tuples with the same BEF, BET, and AFT vectors into a set,
    # so that comparision is made more quickier eficicient
    def merge_all_tuples(self):
        # transform numpy array into a tuple so it can be hashed and added into a set
        # represent a tuple as a python tuple(bef_vector,bet_vector,aft_vector)
        self.tuples_vectors_uniques = set()
        vec_bef = None
        vec_bet = None
        vec_aft = None
        for t in self.tuples:
            vec_bef = tuple(t.bef_vector)
            vec_bet = tuple(t.bet_vector)
            vec_aft = tuple(t.aft_vector)
        tuple_vect = tuple((vec_bef, vec_bet, vec_aft))
        self.tuples_vectors_uniques.add(tuple_vect)

    def merge_all_tuples_bet(self):
        for t in self.tuples:
            # transform numpy array into a tuple so it can be hashed and added into a set
            self.bet_uniques.add(tuple(t.bet_vector))

    def update_selectivity(self, t, config):
        for s in config.positive_seed_tuples:
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
                    else:
                        self.unknown += 1