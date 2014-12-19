#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import pickle
import fileinput
import sys

from Sentence import Sentence
from Pattern import Pattern
from Config import Config
from Tuple import Tuple
from Word2VecWrapper import Word2VecWrapper
from numpy import dot
from gensim import matutils


class BREADS(object):

    def __init__(self, config_file, seeds_file):
        self.patterns = list()
        self.instances = list()
        self.processed_tuples = list()
        self.config = Config(config_file, seeds_file)

    def generate_tuples(self, sentences_file):
        """
        Generate tuples instances from a text file with sentences
        where named entities are already tagged
        """

        try:
            os.path.isfile("processed_tuples.pkl")
            f = open("processed_tuples.pkl", "r")
            print "Loading processed tuples from disk..."
            self.processed_tuples = pickle.load(f)
            f.close()
            print len(self.processed_tuples), "tuples loaded"
        except Exception, e:
            print e
            print "Generating relationship instances from sentences..."
            for line in fileinput.input(sentences_file):
                sentence = Sentence(line.strip())
                for rel in sentence.relationships:
                    if rel.arg1type == self.config.e1_type and rel.arg2type == self.config.e2_type:
                        t = Tuple(rel.ent1, rel.ent2, rel.sentence, rel.before, rel.between, rel.after, self.config)
                        self.processed_tuples.append(t)
            fileinput.close()

            print len(self.processed_tuples), "tuples generated"

            f = open("processed_tuples.pkl", "wb")
            pickle.dump(self.processed_tuples, f)
            f.close()

    def start(self):
        """
        starts a bootstrap iteration
        """
        i = 0
        print "\nStarting", self.config.number_iterations, "iterations"
        while i <= self.config.number_iterations:
            print "Looking for seed matches:"
            for s in self.config.seed_tuples:
                print s.e1, '\t', s.e2

            # Looks for sentences macthing the seed instances
            count_matches, matched_tuples = self.match_seeds_tuples(self)

            if len(matched_tuples) == 0:
                print "\nNo seed matches found"
                sys.exit(0)

            else:
                print "\nMatches found"
                for t in count_matches.keys():
                    print t.e1, '\t', t.e2, '\t', count_matches[t]

                # Cluster the matched instances: generate patterns/update patterns
                print "\nClustering matched instances to generate patterns"
                self.cluster_tuples(self, matched_tuples)

                # Eliminate patterns supported by less than 'min_pattern_support' tuples
                new_patterns = [p for p in self.patterns if len(p.tuples) >= 2]
                self.patterns = new_patterns
                print "Patterns generated"
                print self.patterns

                # Look for sentences with occurrence of seeds semantic types (e.g., ORG - LOC)
                # Measure the similarity of each sentence(Tuple) with each Pattern
                # Matching Tuple objects are used to score a Pattern confidence, based
                # on having extracted a relationship which part of the seed set

                # Update Tuple confidence based on patterns confidence

                # Calculate a new seed set of tuples to use in next iteration, such that:
                # seeds = { T | Conf(T) > min_tuple_confidence }
                i += 1

    @staticmethod
    def cluster_tuples(self, matched_tuples):
        """
        single-pass clustering
        """
        # Initialize: if no patterns exist, first tuple goes to first cluster
        if len(self.patterns) == 0:
            c1 = Pattern(matched_tuples[0])
            self.patterns.append(c1)

        # Compute the similarity between an instance with each pattern
        # go through all tuples
        for t in matched_tuples:
            max_similarity = 0
            max_similarity_cluster_index = 0

            # go through all patterns(clusters of tuples) and find the one with the
            # highest similarity score
            for i in range(0, len(self.patterns), 1):
                extraction_pattern = self.patterns[i]
                similarity = 0

                # each pattern has one or more vectors representing ReVerb patterns
                # compute the similarity between the instance vector and each vector from a pattern
                # in two different ways:
                #   1 - compare each vector from the extraction pattern with the tuple vector
                #   2 - when then extraction pattern is represented as single vector, sum of all vectors
                #

                # 1 - compara similarity with all vectors, if majority is above threshold
                #     assume
                if self.config.similarity == "all":
                    # TODO: só estou a usar o primeiro pattern, a frase pode ter mais
                    accept, score = similarity_all(t, extraction_pattern, self.config)
                    if accept is True and similarity > max_similarity:
                        print "entrei"
                        max_similarity = similarity
                        max_similarity_cluster_index = i

                    """
                    print "tuple words:", t.patterns_words[0]
                    print "pattern    :", extraction_pattern.patterns_words
                    print score
                    print "\n"
                    """

                # 2 - similarity calculate with just one vector
                elif self.config.similarity == "single-vector":
                    # TODO: só estou a usar o primeiro pattern, a frase pode ter mais
                    score = similarity(t.pattern_vectors[0], extraction_pattern)
                    if score > max_similarity:
                        max_similarity = similarity
                        max_similarity_cluster_index = i

            # If max_similarity < min_degree_match create a new cluster having this tuple as the centroid
            if max_similarity < self.config.threshold_similarity:
                c = Pattern(t)
                self.patterns.append(c)

            # If max_similarity >= min_degree_match add to the cluster with the highest similarity
            else:
                self.patterns[max_similarity_cluster_index].add_tuple(t)

    @staticmethod
    def match_seeds_tuples(self):
        """
        checks if an extracted tuple matches seeds tuples
        """
        matched_tuples = list()
        count_matches = dict()
        for t in self.processed_tuples:
            for s in self.config.seed_tuples:
                if t.e1 == s.e1 and t.e2 == s.e2:
                    matched_tuples.append(t)
                    try:
                        count_matches[t] += 1
                    except KeyError:
                        count_matches[t] = 1

        return count_matches, matched_tuples

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


def similarity_sum(sentence_vector, extraction_pattern, config):
    """
    Cosine similarity between a Cluster/Extraction Pattern represented as a single vector
    and the vector of a ReVerb pattern extracted from a sentence
    """
    extraction_pattern.calculate_single_vector()
    score = dot(matutils.unitvec(sentence_vector), matutils.unitvec(extraction_pattern.single_vector))
    return score


def similarity_all(t, extraction_pattern, config):
    """
    Cosine similarity between all patterns part of a Cluster/Extraction Pattern
    and the vector of a ReVerb pattern extracted from a sentence
    """
    good = 0
    bad = 0
    max_similarity = 0
    for p in extraction_pattern.patterns_words:
        vector = Word2VecWrapper.pattern2vector(p, config)
        score = dot(matutils.unitvec(t.patterns_vectors[0]), matutils.unitvec(vector))
        #print t.patterns_words[0], p, score
        if score > max_similarity:
            max_similarity = score
        if score >= config.threshold_similarity:
            good += 1
        else:
            bad += 1

        if good >= bad:
            return True, max_similarity
        else:
            return False, 0.0


def main():
    configuration = sys.argv[1]
    senteces_file = sys.argv[2]
    seeds_file = sys.argv[3]
    breads = BREADS(configuration, seeds_file)
    breads.generate_tuples(senteces_file)
    breads.start()


if __name__ == "__main__":
    main()