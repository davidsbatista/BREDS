#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fileinput
import sys
import BREADSConfig
from Sentence import Sentence
from Tuple import Tuple

__author__ = 'dsbatista'


class BREADS(object):

    def __init__(self, config_file):
        self.tuples = list()
        self.config = BREADSConfig(config_file)

    def generate_tuples(self, sentences_file):
        """
        Generate tuples instances from a text file with sentences
        where named entities are already tagged
        """
        #TODO: apenas tuplos da relação de interesse: ORG-LOC, ou PER-ORG
        for line in fileinput.input(sentences_file):
            sentence = Sentence(line.strip())
            for rel in sentence.relationships:
                """
                print rel.sentence
                print rel.ent1, rel.arg1type
                print rel.ent2, rel.arg2type
                print rel.before
                print rel.between
                print rel.after
                print "\n"
                """
                t = Tuple(rel.ent1, rel.ent2, rel.sentence, rel.before, rel.between, rel.after)
                self.tuples.append(t)

        fileinput.close()

    @staticmethod
    def calculate_tuple_confidence(self):
        """
        Calculates the confidence of a tuple is: Conf(P_i) * DegreeMatch(P_i)
        """
        pass

    @staticmethod
    def compare_pattern_tuples(self):
        """
        Compute similarity of a tuple with all the extraction patterns
        Compare the relational words from the sentence with every extraction pattern
        """
        pass

    @staticmethod
    def expand_patterns(self):
        """
        Expands the relationship words of a pattern based on similarities
        - For each word part of a pattern
            - Construct a set with all the similar words according to Word2Vec given a threshold t
        - Calculate the intersection of all sets
        """
        pass

    @staticmethod
    def match_seeds_tuples(self):
        """
        checks if an extracted tuple matches seeds tuples
        """

    @staticmethod
    def pattern_drifts(self):
        """
        detects if the patterns drift
        """

    @staticmethod
    def iteration(self):
        """
        starts a bootstrap iteration
        """


def main(senteces_file):

    breads = BREADS()
    breads.generate_tuples(senteces_file)


if __name__ == "__main__":
    main(sys.argv[1])