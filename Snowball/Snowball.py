#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import pickle
import sys
import os
import codecs
import TFIDF

from collections import defaultdict
from Sentence import Sentence
from Pattern import Pattern
from Config import Config
from Tuple import Tuple


class Snowball(object):

    def __init__(self, config_file, seeds_file):
        self.tf_idf = None
        self.patterns = list()
        self.processed_tuples = list()
        self.candidate_tuples = defaultdict(list)
        self.config = Config(config_file, seeds_file)

    def generate_vsm_model(self, sentences_file):
        try:
            os.path.isfile("tf_idf.pkl")
            f = open("tf_idf.pkl", "r")
            print "\nLoading tf-idf model from disk..."
            self.processed_tuples = pickle.load(f)
            f.close()
            print len(self.processed_tuples), "tuples loaded"

        except IOError:
            print "\nGenerating tf-idf model from sentences..."
            self.tf_idf = TFIDF.tf_idf(sentences_file)
            print self.tf_idf
            sys.exit(0)

    def generate_tuples(self, sentences_file):
        """
        Generate tuples instances from a text file with sentences
        where named entities are already tagged
        """
        try:
            os.path.isfile("processed_tuples.pkl")
            f = open("processed_tuples.pkl", "r")
            print "\nLoading processed tuples from disk..."
            self.processed_tuples = pickle.load(f)
            f.close()
            print len(self.processed_tuples), "tuples loaded"
        except IOError:
            print "\nGenerating relationship instances from sentences..."
            f_sentences = codecs.open(sentences_file, encoding='utf-8')
            for line in f_sentences:
                sentence = Sentence(line.strip())
                for rel in sentence.relationships:
                    if rel.arg1type == self.config.e1_type and rel.arg2type == self.config.e2_type:
                        t = Tuple(rel.ent1, rel.ent2, rel.sentence, rel.before, rel.between, rel.after, self.config)
                        if len(t.patterns_vectors) >= 1:
                            self.processed_tuples.append(t)
            f_sentences.close()

            print len(self.processed_tuples), "tuples generated"

            f = open("processed_tuples.pkl", "wb")
            pickle.dump(self.processed_tuples, f)
            f.close()

    def start(self):
        """
        starts a bootstrap iteration
        """
        i = 0
        while i <= self.config.number_iterations:
            print "\nStarting iteration", i
            print "\nLooking for seed matches of:"
            for s in self.config.seed_tuples:
                print s.e1, '\t', s.e2

            # Looks for sentences macthing the seed instances
            count_matches, matched_tuples = self.match_seeds_tuples(self)

            if len(matched_tuples) == 0:
                print "\nNo seed matches found"
                sys.exit(0)

            else:
                """
                print "\nNumber of seed matches found"
                sorted_counts = sorted(count_matches.items(), key=operator.itemgetter(1), reverse=True)

                for t in sorted_counts:
                    print t[0].e1, '\t', t[0].e2, t[1]
                """

                # Cluster the matched instances: generate patterns/update patterns
                print "\nClustering matched instances to generate patterns"
                self.cluster_tuples(self, matched_tuples)

                # Eliminate patterns supported by less than 'min_pattern_support' tuples
                new_patterns = [p for p in self.patterns if len(p.tuples) >= 2]
                self.patterns = new_patterns
                print "\n", len(self.patterns), "patterns generated"
                if i == 0 and len(self.patterns) == 0:
                    print "No patterns generated"
                    sys.exit(0)

                # Look for sentences with occurrence of seeds semantic types (e.g., ORG - LOC)
                # This was already collect and its stored in: self.processed_tuples
                #
                # Measure the similarity of each occurrence with each extraction pattern
                # and store each pattern that has a similarity higher than a given threshold
                #
                # Each candidate tuple will then have a number of patterns that helped generate it,
                # each with an associated de gree of match. Snowball uses this infor
                print "\nCollecting instances based on extraction patterns"
                for t in self.processed_tuples:
                    sim_best = 0
                    for extraction_pattern in self.patterns:
                        if self.config.similarity == "all":
                            score = similarity(t, extraction_pattern)
                            if score > sim_best:
                                sim_best = score
                                pattern_best = extraction_pattern

                    if sim_best >= self.config.threshold_similarity:
                        # If this instance was not seen before, associate this pattern with the instance
                        # and the similarity score
                        self.candidate_tuples[t].append((pattern_best, sim_best))

                        # if this tuple was already extracted, check it this pattern is already associated with it
                        # associate this Pattern and similarity score with the Tuple
                        patterns = self.candidate_tuples[t]
                        if not extraction_pattern in patterns:
                            self.candidate_tuples[t].append((pattern_best, sim_best))

                    # update extraction pattern confidence
                    if iter > 0:
                        extraction_pattern.confidence_old = extraction_pattern.confidence
                        extraction_pattern.update_confidence()

                print "\nExtraction patterns confidence:"
                tmp = sorted(self.patterns)
                for p in tmp:
                    print p, '\t', len(p.patterns_words), '\t', p.confidence

                # update tuple confidence based on patterns confidence
                print "\nCalculating tuples confidence"
                for t in self.candidate_tuples.keys():
                    confidence = 1
                    t.confidence_old = t.confidence
                    for p in self.candidate_tuples.get(t):
                        confidence *= 1 - (p[0].confidence * p[1])
                    t.confidence = 1 - confidence

                    # use past confidence values to calculate new confidence
                    # if parameter Wupdt < 0.5 the system trusts new examples less on each iteration
                    # which will lead to more conservative patterns and have a damping effect.
                    if iter > 0:
                        t.confidence = t.confidence * self.config.wUpdt + t.confidence_old * (1 - self.config.wUpdt)

                # update seed set of tuples to use in next iteration
                # seeds = { T | Conf(T) > min_tuple_confidence }
                if i+1 < self.config.number_iterations:
                    print "Adding tuples to seed with confidence =>" + str(self.config.instance_confidance)
                    for t in self.candidate_tuples.keys():
                        if t.confidence >= self.config.instance_confidance:
                            self.config.seed_tuples.add(t)

                # increment the number of iterations
                i += 1

        print "\nWriting extracted relationships to disk"
        f_output = open("relationships.txt", "w")
        tmp = sorted(self.candidate_tuples.keys(), reverse=True)
        for t in tmp:
            f_output.write("instance: "+t.e1+'\t'+t.e2+'\tscore:'+str(t.confidence)+'\n')
            f_output.write("sentence: "+t.sentence.encode("utf8")+'\n')
            f_output.write("pattern : "+t.patterns_words[0]+'\n')
            f_output.write("\n")
        f_output.close()

        print "Writing generated patterns to disk"
        f_output = open("patterns.txt", "w")
        tmp = sorted(self.patterns, reverse=True)
        for p in tmp:
            f_output.write(str(p.patterns_words)+'\t'+str(p.confidence)+'\n')
        f_output.close()

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

                # TODO: aplicar a formula do Snowball
                score = similarity(t, extraction_pattern)
                if score > max_similarity:
                    max_similarity = score
                    max_similarity_cluster_index = i

            # if max_similarity < min_degree_match create a new cluster having this tuple as the centroid
            if max_similarity < self.config.threshold_similarity:
                c = Pattern(t)
                self.patterns.append(c)

            # if max_similarity >= min_degree_match add to the cluster with the highest similarity
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


def similarity(t, extraction_pattern):
    return 0
    pass


def main():
    configuration = sys.argv[1]
    sentences_file = sys.argv[2]
    seeds_file = sys.argv[3]
    snowball = Snowball(configuration, seeds_file)
    snowball.generate_vsm_model(sentences_file)
    snowball.generate_tuples(sentences_file)
    snowball.start()

if __name__ == "__main__":
    main()