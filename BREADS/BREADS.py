#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fileinput
import sys
from gensim.models.word2vec import Word2Vec

from BREADS import Config
from BREADS.Pattern import Pattern
from Sentence import Sentence
from Tuple import Tuple


__author__ = 'dsbatista'


class BREADS(object):

    def __init__(self, config_file, seeds_file):
        self.processed_tuples = list()
        self.config = Config(config_file, seeds_file)

    def generate_tuples(self, sentences_file):
        """
        Generate tuples instances from a text file with sentences
        where named entities are already tagged
        """

        print "Generating relationship instances from sentences..."
        for line in fileinput.input(sentences_file):
            sentence = Sentence(line.strip())
            for rel in sentence.relationships:
                if rel.arg1type == self.config.e1_type and rel.arg2type == self.config.e2_type:
                    t = Tuple(rel.ent1, rel.ent2, rel.sentence, rel.before, rel.between, rel.after, self.config)
                    self.processed_tuples.append(t)
        fileinput.close()

        print len(self.processed_tuples), "tuples generated"



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
    def iteration(self):
        """
        starts a bootstrap iteration
        """
        self.iteration = 0
        self.patterns = list()
        while iter <= self.config.number_iterations:

            # Looks for sentences macthing the seed instances
            matched_tuples = self.match_seeds_tuples()

            # Cluster the matched instances to generate patterns
            self.cluster_tuples(matched_tuples)

            # Eliminate patterns supported by less than 'min_pattern_support' tuples

            # Look for sentences with occurrence of seeds semantic types (e.g., ORG - LOC)
            # Measure the similarity of each sentence(Tuple) with each Pattern
            # Matching Tuple objects are used to score a Pattern confidence, based
            # on having extracted a relationship which part of the seed set

            # Update Tuple confidence based on patterns confidence

            # Calculate a new seed set of tuples to use in next iteration, such that:
            # seeds = { T | Conf(T) > min_tuple_confidence }

    @staticmethod
    def cluster_tuples(self, matched_tuples):
        """
        single-pass clustering
        """

        # Initialize: first tuple goes to first cluster
        if len(self.patterns) == 0:
            c1 = Pattern(matched_tuples[0])
            self.patterns.add(c1)

            # Compute the similarity between an instance with each pattern
            # go through all tuples
            for t in matched_tuples:
                max_similarity = 0
                max_similarity_cluster_index = 0

                # go through all patterns
                for i in range(0, len(self.patterns), 1):
                    extraction_pattern = self.patterns[i]
                    similarity = 0

                    # each pattern has one or more vectors representing ReVerb patterns
                    # compute the similarity between the instance vector and each vector from a pattern
                    # in two different ways:
                    #      1 - when then extraction pattern is represented as single vector, sum of all vectors
                    #      2 - compare each vector from the extraction pattern with the tuple vector

                    # 1- similarity calculate with just one vector
                    if self.config.similarity == "single-vector":
                        # TODO: só estou a usar o primeiro pattern, a frase pode ter mais
                        score = similarity(t.pattern_vectors[0], extraction_pattern)

                    elif self.config.similarity == "all":
                        score = similarity(t.pattern_vectors[0], extraction_pattern)





    @staticmethod
    def match_seeds_tuples(self):
        """
        checks if an extracted tuple matches seeds tuples
        """
        matched_tuples = set()
        for t in self.processed_tuples:
            for s in self.config.seed_tuples:
                if t.e1 == s.e1 and t.e2 == s.e2:
                    matched_tuples.add(tuple)

        return matched_tuples


def similarity_sum(sentence_vector, extraction_pattern):
    """
    Cosine similarity between a Cluster/Extraction Pattern represented as a single vector
    and the vector of a ReVerb pattern extracted from a sentence
    """


def similarity_all(sentence_vector, extraction_pattern):
    """
    Cosine similarity between all patterns part of a Cluster/Extraction Pattern
    and the vector of a ReVerb pattern extracted from a sentence
    """


def main():
    configuration = sys.argv[1]
    senteces_file = sys.argv[2]
    seeds_file = sys.argv[3]
    breads = BREADS(configuration, seeds_file)
    breads.generate_tuples(senteces_file)


if __name__ == "__main__":
    main()