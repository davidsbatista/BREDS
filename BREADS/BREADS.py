#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import pickle
import sys
import os
import codecs
import operator

from numpy import dot
from gensim import matutils
from nltk import PunktWordTokenizer
from collections import defaultdict

from Pattern import Pattern
from Config import Config
from Tuple import Tuple
from Sentence import Sentence
from Word2VecWrapper import Word2VecWrapper


class BREADS(object):

    def __init__(self, config_file, seeds_file):
        self.patterns = list()
        self.processed_tuples = list()
        self.candidate_tuples = defaultdict(list)
        self.config = Config(config_file, seeds_file)

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
                print "\nNumber of seed matches found"
                sorted_counts = sorted(count_matches.items(), key=operator.itemgetter(1), reverse=True)

                for t in sorted_counts:
                    print t[0][0], '\t', t[0][1], t[1]

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
                            accept, score = similarity_all(t, extraction_pattern, self.config)
                            if accept is True:
                                extraction_pattern.update_selectivity(t, self.config)
                                if score > sim_best:
                                    sim_best = score
                                    pattern_best = extraction_pattern

                    if sim_best >= self.config.threshold_similarity:
                        # if this tuple was already extracted, check if this extraction pattern is already associated
                        # with it. if not associate this pattern with it and similarity score
                        patterns = self.candidate_tuples[t]
                        if patterns is not None:
                            # patterns          : list<(Pattern,float)>
                            # extraction_pattern: Pattern
                            #print patterns, extraction_pattern, type(patterns), type(extraction_pattern)
                            if extraction_pattern not in [x[0] for x in patterns]:
                                self.candidate_tuples[t].append((pattern_best, sim_best))

                        # If this tuple was not extracted before, associate this pattern with the instance
                        # and the similarity score
                        else:
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

                # each pattern has one or more vectors representing ReVerb patterns
                # compute the similarity between the instance vector and each vector from a pattern
                # in two different ways:
                # 1 - compare similarity with all vectors, if majority is above threshold
                #     assume
                if self.config.similarity == "all":
                    try:
                        accept, score = similarity_all(t, extraction_pattern, self.config)
                        if accept is True and score > max_similarity:
                            max_similarity = score
                            max_similarity_cluster_index = i
                    except Exception, e:
                        # TODO: t e extraction_pattern nao podem ser vazios
                        # ver pq isto estah a acontecer
                        print e
                        print "tuple"
                        print t.sentence
                        print t.e1, '\t', t.e2
                        print extraction_pattern
                        sys.exit(0)

                # 2 - similarity calculate with just one vector, representd by the sum of all
                #     tuple's vectors in a pattern/cluster
                elif self.config.similarity == "single-vector":
                    score = similarity_sum(t.pattern_vectors[0], extraction_pattern)
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
                        count_matches[(t.e1, t.e2)] += 1
                    except KeyError:
                        count_matches[(t.e1, t.e2)] = 1

        return count_matches, matched_tuples


def similarity_sum(sentence_vector, extraction_pattern):
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
    for p in list(extraction_pattern.patterns_words):
        tokens = PunktWordTokenizer().tokenize(p)
        vector = Word2VecWrapper.pattern2vector(tokens, config)
        score = dot(matutils.unitvec(t.patterns_vectors[0]), matutils.unitvec(vector))
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