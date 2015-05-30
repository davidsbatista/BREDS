#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import multiprocessing
import cPickle
import sys
import os
import codecs
import operator
import numpy as np
import re
import time
import Queue

from nltk.corpus import stopwords
from nltk import word_tokenize
from numpy import dot
from gensim import matutils
from sklearn.cluster import DBSCAN
from collections import defaultdict

from BREDS.PatternParsing import PatternMatrices
from BREDS.PatternPoS import Pattern
from BREDS.Config import Config
from BREDS.TuplePoS import Tuple
from BREDS.TupleParsing import TupleOfParser
from Common.SentenceBreds import Sentence
from Common.SentenceBreds import SentenceParser
from Common.Seed import Seed

# usefull stuff for debugging
PRINT_TUPLES = False
PRINT_PATTERNS = False


class BREDSDistSim(object):

    def __init__(self, config_file, seeds_file, negative_seeds, similarity, confidance, sentences_file):
        self.curr_iteration = 0
        self.patterns = list()
        self.processed_tuples = list()
        self.candidate_tuples = defaultdict(list)
        self.config = Config(config_file, seeds_file, negative_seeds, similarity, confidance, sentences_file)

        # to control the semantic drift using the seeds from different iterations
        self.seeds_by_iteration = dict()

    def generate_tuples(self, sentences_file):
        """
        Generate tuples instances from a text file with sentences where named entities are already tagged
        """
        try:
            os.path.isfile("processed_tuples.pkl")
            f = open("processed_tuples.pkl", "r")
            print "\nLoading processed tuples from disk..."
            self.processed_tuples = cPickle.load(f)
            f.close()
            print len(self.processed_tuples), "tuples loaded"

        except IOError:
            self.read_word2vec()
            print "\nGenerating relationship instances from sentences"
            f_sentences = codecs.open(sentences_file, encoding='utf-8')
            count = 0
            for line in f_sentences:
                count += 1
                if count % 10000 == 0:
                    sys.stdout.write(".")

                sentence = Sentence(line.strip(), self.config.e1_type, self.config.e2_type, self.config.max_tokens_away,
                                    self.config.min_tokens_away, self.config.context_window_size)
                for rel in sentence.relationships:
                    if rel.arg1type == self.config.e1_type and rel.arg2type == self.config.e2_type:
                        t = Tuple(rel.ent1, rel.ent2, rel.sentence, rel.before, rel.between, rel.after, self.config)
                        self.processed_tuples.append(t)

            f_sentences.close()

            print "\n", len(self.processed_tuples), "tuples generated"
            print "Writing generated tuples to disk"
            f = open("processed_tuples.pkl", "wb")
            cPickle.dump(self.processed_tuples, f)
            f.close()

    def similarity_3_contexts(self, p, t):
        (bef, bet, aft) = (0, 0, 0)

        if t.bef_vector is not None and p.bef_vector is not None:
            bef = dot(matutils.unitvec(t.bef_vector), matutils.unitvec(p.bef_vector))

        if t.bet_vector is not None and p.bet_vector is not None:
            bet = dot(matutils.unitvec(t.bet_vector), matutils.unitvec(p.bet_vector))

        if t.aft_vector is not None and p.aft_vector is not None:
            aft = dot(matutils.unitvec(t.aft_vector), matutils.unitvec(p.aft_vector))

        return self.config.alpha*bef + self.config.beta*bet + self.config.gamma*aft

    @staticmethod
    def drift_two(r, i):
        # The third approach constrains the instances to be added to the seed set on two conditions. An extracted
        # instance x is only added to the seed set if its similarity with at least N instances extracted in the previous
        # i iteration is above a threshold τ sim ,where N is the number of instances extracted inthe previous iteration.
        pass

    def init_bootstrapp(self, tuples):
        m = multiprocessing.Manager()
        """
        starts a bootstrap iteration
        """
        if tuples is not None:
            f = open(tuples, "r")
            print "\nLoading processed tuples from disk..."
            self.processed_tuples = cPickle.load(f)
            f.close()
            print len(self.processed_tuples), "tuples loaded"

        self.curr_iteration = 0
        while self.curr_iteration <= self.config.number_iterations:
            print "=========================================="
            print "\nStarting iteration", self.curr_iteration
            print "\nLooking for seed matches of:"
            for s in self.config.seed_tuples:
                print s.e1, '\t', s.e2

            # Looks for sentences macthing the seed instances
            count_matches, matched_tuples = self.match_seeds_tuples()

            if len(matched_tuples) == 0:
                print "\nNo seed matches found"
                sys.exit(0)

            else:
                print "\nNumber of seed matches found"
                sorted_counts = sorted(count_matches.items(), key=operator.itemgetter(1), reverse=True)
                for t in sorted_counts:
                    print t[0][0], '\t', t[0][1], t[1]

                print "\n", len(matched_tuples), "tuples matched"

                # Cluster the matched instances: generate patterns/update patterns
                print "\nClustering matched instances"
                clusters = self.cluster_dbscan_vectors(matched_tuples)
                self.seeds_by_iteration[0] = list()
                for k in clusters:
                    #print "label", len(clusters[k])
                    for rel in clusters[k]:
                        self.seeds_by_iteration[0].append(rel)

                if len(self.seeds_by_iteration[0]) == 0:
                    print "No clusters generated"
                    sys.exit(0)

                """
                print "Seed sentences"
                print len(self.seeds_by_iteration[0])
                for rel in self.seeds_by_iteration[0]:
                    print rel.e1, '\t', rel.e2
                    print rel.sentence
                    print
                """

                # Look for sentences with occurrence of seeds semantic types (e.g., ORG - LOC)
                # This was already collect and its stored in: self.processed_tuples
                count = 0
                print "Number of sentences to be analyzed:", len(self.processed_tuples)
                print "\nCollecting instances based on seed sentences and average distributional similarity"
                """
                queue = multiprocessing.Queue()
                num_cpus = multiprocessing.cpu_count()
                results = [m.list() for _ in range(num_cpus)]
                processes = [multiprocessing.Process(target=self.calculate_similarity, args=(queue, results[i])) for i in range(num_cpus)]

                print "Queue"
                for t in self.processed_tuples:
                    queue.put(t)

                print "size", queue.qsize()
                print len(self.processed_tuples)


                for proc in processes:
                    proc.start()
                    print proc
                for proc in processes:
                    proc.join()

                all_results = list()
                for l in results:
                    all_results.extend(l)
                """
                for t in self.processed_tuples:
                    count += 1
                    if count % 1000 == 0:
                        sys.stdout.write(".")
                        sys.stdout.flush()

                    previous, current = self.average_similarity(t)
                    if current > 0.5:
                        print current
                        print previous

                # increment the number of iterations
                self.curr_iteration += 1

        print "\nWriting extracted relationships to disk"
        f_output = open("relationships.txt", "w")
        tmp = sorted(self.candidate_tuples.keys(), reverse=True)
        for t in tmp:
            f_output.write("instance: "+t.e1.encode("utf8")+'\t'+t.e2.encode("utf8")+'\tscore:'+str(t.confidence)+'\n')
            f_output.write("sentence: "+t.sentence.encode("utf8")+'\n')
            f_output.write("pattern_bef: " + t.bef_words+'\n')
            f_output.write("pattern_bet: " + t.bet_words+'\n')
            f_output.write("pattern_aft: " + t.aft_words+'\n')
            if t.passive_voice is False:
                f_output.write("passive voice: False\n")
            elif t.passive_voice is True:
                f_output.write("passive voice: True\n")
            f_output.write("\n")
        f_output.close()

    def calculate_similarity(self, queue, results):
        while True:
            try:
                t = queue.get_nowait()
                drift = self.average_similarity(t)
                if drift != 0:
                    print drift
            except Queue.Empty:
                break

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

    def cluster_dbscan_vectors(self, matched_tuples):
        # build a matrix with all the pairwise distances between all the matrix representing sentences
        matrix = np.zeros((len(matched_tuples), len(matched_tuples)))
        print "Calculating pairwise distances..."
        for ex1 in range(len(matched_tuples)):
            for ex2 in range(len(matched_tuples)):
                similarity = self.similarity_3_contexts(matched_tuples[ex1], matched_tuples[ex2])
                # dbscan deals with distances, transform similarity into a distance
                matrix[ex1, ex2] = 1-similarity

        # perform DBSCAN
        db = DBSCAN(eps=0.5, min_samples=2, metric='precomputed')
        db.fit(matrix)
        clusters = defaultdict(list)

        print db.labels_
        assert len(db.labels_) == len(matched_tuples)

        # aggregate results by label
        # -1 represents noise, discarded
        for v in range(len(matched_tuples)):
            label = db.labels_[v]
            if label > -1:
                clusters[label].append(matched_tuples[v])

        return clusters


class BREDS(object):

    def __init__(self, config_file, seeds_file, negative_seeds, similarity, confidance, sentences_file):
        self.curr_iteration = 0
        self.patterns = list()
        self.processed_tuples = list()
        self.candidate_tuples = defaultdict(list)
        self.config = Config(config_file, seeds_file, negative_seeds, similarity, confidance, sentences_file)

        # to control the semantic drift using the seeds from different iterations
        self.seeds_by_iteration = dict()

    def find_relationships(self, queue, results):
        count = 0
        while True:
            try:
                line = queue.get_nowait()
                count += 1
                if count % 5000 == 0:
                    print "Queue:", queue.qsize()
                    print multiprocessing.current_process(), len(results)

                sentence = Sentence(line, self.config.e1_type, self.config.e2_type, self.config.max_tokens_away,
                                    self.config.min_tokens_away, self.config.context_window_size, self.config)
                for rel in sentence.relationships:
                    t = Tuple(rel.ent1, rel.ent2, rel.sentence, rel.before, rel.between, rel.after, self.config)
                    results.append(t)
            except Queue.Empty:
                print multiprocessing.current_process(), "queue empty"
                break

    def generate_tuples(self, sentences_file):
        """
        Generate tuples instances from a text file with sentences where named entities are already tagged
        """
        try:
            os.path.isfile("processed_tuples.pkl")
            f = open("processed_tuples.pkl", "r")
            print "\nLoading processed tuples from disk..."
            self.processed_tuples = cPickle.load(f)
            f.close()
            print len(self.processed_tuples), "tuples loaded"

        except IOError:

            self.config.read_word2vec()
            sentences_tmp = list()

            m = multiprocessing.Manager()
            queue = m.Queue()
            num_cpus = multiprocessing.cpu_count()
            #num_cpus = 1

            print "Loading sentences into Queue"
            f_sentences = codecs.open(sentences_file, encoding='utf-8')
            count = 0
            for line in f_sentences:
                queue.put(line.strip())
                count += 1
                if count % 10000 == 0:
                    sys.stdout.write(".")
            f_sentences.close()
            print "\nDone all"

            print "\nGenerating relationship instances from sentences"
            results = [m.list() for _ in range(num_cpus)]
            processes = [multiprocessing.Process(target=self.find_relationships, args=(queue, results[i])) for i in range(num_cpus)]

            print "Launching", num_cpus, "processes"
            for proc in processes:
                proc.start()
            for proc in processes:
                proc.join()

            all_results = list()
            for l in results:
                all_results.extend(l)

            for t in all_results:
                self.processed_tuples.append(t)

            print len(self.processed_tuples)


            """
            print "\nGenerating relationship instances from sentences"
            f_sentences = codecs.open(sentences_file, encoding='utf-8')
            count = 0
            for line in f_sentences:
                count += 1
                if count % 500 == 0:
                    sys.stdout.write(".")

                #sentence_no_tags = re.sub(self.config.tags_regex, "", line.strip())
                #if len(word_tokenize(sentence_no_tags)) > 35:
                #    continue
                #else:
                if self.config.embeddings != 'fcm':
                    sentence = Sentence(line.strip(), self.config.e1_type, self.config.e2_type, self.config.max_tokens_away,
                                    self.config.min_tokens_away, self.config.context_window_size)

                    for rel in sentence.relationships:
                        t = Tuple(rel.ent1, rel.ent2, rel.sentence, rel.before, rel.between, rel.after, self.config)
                        self.processed_tuples.append(t)

                # Embeddings based on matrices
                elif self.config.embeddings == 'fcm':
                    sentence = SentenceParser(line.strip(), self.config.e1_type, self.config.e2_type)
                    if sentence.valid is True:
                        sentences_tmp.append(sentence)

            f_sentences.close()
            """

            if self.config.embeddings == 'fcm':
                sentences = list()
                print "\n\n"+str(len(sentences_tmp)), "sentences to parse"
                text_to_parse = list()
                start = time.time()
                for s in sentences_tmp:
                    sentence_no_tags = re.sub(self.config.tags_regex, "", s.sentence)
                    if len(word_tokenize(sentence_no_tags)) > 35:
                        continue
                    else:
                        sentences.append(s)
                        text_to_parse.append(sentence_no_tags)

                #trees, deps = self.config.parser.raw_parse_sents_deps(text_to_parse)
                trees = self.config.parser.raw_parse_sents_deps(text_to_parse)
                end = time.time()
                seconds = end - start
                m, s = divmod(seconds, 60)
                h, m = divmod(m, 60)
                print "Time taken: %d:%02d:%02d" % (h, m, s)

                print "\nConverting", str(len(sentences)), "trees"
                start = time.time()
                deps = self.config.parser.convert_trees(trees)
                end = time.time()
                seconds = end - start
                m, s = divmod(seconds, 60)
                h, m = divmod(m, 60)
                print "Time taken: %d:%02d:%02d" % (h, m, s)

                try:
                    assert len(trees) == len(deps) == len(sentences)
                except AssertionError:
                    print "Error in Parsing"
                    print "Trees", len(trees)
                    print "Deps", len(deps)
                    print "Sentences", len(sentences)
                    sys.exit(0)

                """
                print "Computing FCM embeddings"
                queue = multiprocessing.Queue()
                #num_cpus = multiprocessing.cpu_count()
                num_cpus = 1

                print "Putting everything in a Queue"
                for i in range(len(sentences)):
                    sentences[i].tree = trees[i]
                    sentences[i].deps = deps[i]
                    queue.put(sentences[i])
                print "Done all"

                print "Generating matrices"
                # 2. função que retira da queue, cria TupleParser completo, e poem em processed_tuples
                m = multiprocessing.Manager()
                results = [m.list() for _ in range(num_cpus)]
                processes = [multiprocessing.Process(target=self.create_matrices, args=(queue, results[i])) for i in range(num_cpus)]

                for proc in processes:
                    proc.start()
                for proc in processes:
                    proc.join()

                all_results = list()
                for l in results:
                    all_results.extend(l)

                for t in all_results:
                    self.processed_tuples.append(t)
                """

                print "\nProcessing", str(len(sentences)), "sentences"
                for i in range(len(sentences)):
                    s = sentences[i]
                    s.deps = deps[i]
                    for e1 in s.entities:
                        for e2 in s.entities:
                            if e1 == e2:
                                continue
                            arg1match = re.match("<([A-Z]+)>", e1)
                            arg2match = re.match("<([A-Z]+)>", e2)
                            if arg1match.group(1) == self.config.e1_type and arg2match.group(1) == self.config.e2_type:
                                entity1 = re.sub(self.config.tags_regex, "", e1)
                                entity2 = re.sub(self.config.tags_regex, "", e2)
                                t = TupleOfParser(entity1, entity2, s.deps, s.sentence, self.config)
                                self.processed_tuples.append(t)
                print "Done all"

                # globally normalize matrix values to [0,1]
                print "Normalizing matrixes values"
                max_value = 0
                for t in self.processed_tuples:
                    if t.matrix.max() > max_value:
                        max_value = t.matrix.max()

                for t in self.processed_tuples:
                    t.matrix = np.divide(t.matrix, t.matrix.max())

            print "\n", len(self.processed_tuples), "tuples generated"
            print "Writing generated tuples to disk"
            f = open("processed_tuples.pkl", "wb")
            cPickle.dump(self.processed_tuples, f)
            f.close()

    def create_matrices(self, queue, results):
        while True:
            try:
                s = queue.get_nowait()
                print "Analyzing sentence"
                for e1 in s.entities:
                    for e2 in s.entities:
                        if e1 == e2:
                            continue
                        arg1match = re.match("<([A-Z]+)>", e1)
                        arg2match = re.match("<([A-Z]+)>", e2)
                        if arg1match.group(1) == self.config.e1_type and arg2match.group(1) == self.config.e2_type:
                            entity1 = re.sub(self.config.tags_regex, "", e1)
                            entity2 = re.sub(self.config.tags_regex, "", e2)
                            t = TupleOfParser(entity1, entity2, s.deps, s.sentence, self.config)
                            results.append(t)
            except Queue.Empty:
                break

    def similarity_3_contexts(self, p, t):
        (bef, bet, aft) = (0, 0, 0)

        """
        print "Tuple"
        print t.e1, '\t', t.e2
        print t.sentence
        print t.bef_words
        print t.bet_words
        print t.aft_words

        print "Pattern"
        print p.e1, '\t', p.e2
        print p.sentence
        print p.bef_words
        print p.bet_words
        print p.aft_words
        """

        if t.bef_vector is not None and p.bef_vector is not None:
            bef = dot(matutils.unitvec(t.bef_vector), matutils.unitvec(p.bef_vector))

        if t.bet_vector is not None and p.bet_vector is not None:
            bet = dot(matutils.unitvec(t.bet_vector), matutils.unitvec(p.bet_vector))

        if t.aft_vector is not None and p.aft_vector is not None:
            aft = dot(matutils.unitvec(t.aft_vector), matutils.unitvec(p.aft_vector))

        """
        print "scores:"
        print "bef", bef
        print "bet", bet
        print "aft", aft
        print "score", self.config.alpha*bef + self.config.beta*bet + self.config.gamma*aft
        print "\n"
        """
        return self.config.alpha*bef + self.config.beta*bet + self.config.gamma*aft

    def average_similarity(self, r, current, previous):
        # calculate similarity with current
        avg_sim_current = 0.0
        for t in current:
            if t == r:
                continue
            avg_sim_current += self.similarity_3_contexts(t, r)
        if avg_sim_current > 0:
            avg_sim_current /= len(current)
        else:
            avg_sim_current = 0

        # calculate similarity with previous
        avg_sim_previous = 0.0
        for t in previous:
            if t == r:
                continue
            avg_sim_previous += self.similarity_3_contexts(t, r)
        if avg_sim_previous > 0:
            avg_sim_previous /= len(previous)
        else:
            avg_sim_previous = 0

        return avg_sim_previous, avg_sim_current

    def init_bootstrapp(self, tuples):
        """
        starts a bootstrap iteration
        """
        if tuples is not None:
            f = open(tuples, "r")
            print "\nLoading processed tuples from disk..."
            self.processed_tuples = cPickle.load(f)
            f.close()
            print len(self.processed_tuples), "tuples loaded"

        self.curr_iteration = 0
        while self.curr_iteration <= self.config.number_iterations:
            print "=========================================="
            print "\nStarting iteration", self.curr_iteration
            print "\nLooking for seed matches of:"
            for s in self.config.seed_tuples:
                print s.e1, '\t', s.e2

            # Looks for sentences macthing the seed instances
            count_matches, matched_tuples = self.match_seeds_tuples()

            if len(matched_tuples) == 0:
                print "\nNo seed matches found"
                sys.exit(0)

            else:
                print "\nNumber of seed matches found"
                sorted_counts = sorted(count_matches.items(), key=operator.itemgetter(1), reverse=True)
                for t in sorted_counts:
                    print t[0][0], '\t', t[0][1], t[1]

                print "\n", len(matched_tuples), "tuples matched"

                # Cluster the matched instances: generate patterns/update patterns
                print "\nClustering matched instances to generate patterns"
                if self.config.embeddings == 'fcm':
                    self.cluster_dbscan(matched_tuples)
                else:
                    self.cluster_tuples(matched_tuples)
                    # Eliminate patterns supported by less than 'min_pattern_support' tuples
                    new_patterns = [p for p in self.patterns if len(p.tuples) >= 2]
                    self.patterns = new_patterns

                print "\n", len(self.patterns), "patterns generated"

                if PRINT_PATTERNS is True:
                    count = 1
                    print "\nPatterns:"
                    for p in self.patterns:
                        print count
                        for t in p.tuples:
                            print "BEF", t.bef_words
                            print "BET", t.bet_words
                            print "AFT", t.aft_words
                            print "========"
                            print "\n"
                        count += 1

                if self.curr_iteration == 0 and len(self.patterns) == 0:
                    print "No patterns generated"
                    sys.exit(0)

                # Look for sentences with occurrence of seeds semantic types (e.g., ORG - LOC)
                # This was already collect and its stored in: self.processed_tuples
                #
                # Measure the similarity of each occurrence with each extraction pattern
                # and store each pattern that has a similarity higher than a given threshold
                #
                # Each candidate tuple will then have a number of patterns that extracted it
                # each with an associated degree of match.
                print "Number of tuples to be analyzed:", len(self.processed_tuples)

                print "\nCollecting instances based on extraction patterns"
                count = 0

                for t in self.processed_tuples:
                    count += 1
                    if count % 1000 == 0:
                        sys.stdout.write(".")
                        sys.stdout.flush()
                    sim_best = 0
                    accept = 0
                    for extraction_pattern in self.patterns:

                        if self.config.embeddings == 'fcm':
                            accept, score = self.similarity_matrix_all_2(t, extraction_pattern)

                        elif self.config.embeddings == 'sum':
                            accept, score = self.similarity_all_1(t, extraction_pattern)

                        if accept is True:
                            extraction_pattern.update_selectivity(t, self.config)
                            if score > sim_best:
                                sim_best = score
                                pattern_best = extraction_pattern

                    if sim_best >= self.config.threshold_similarity:
                        # if this tuple was already extracted, check if this extraction pattern is already associated
                        # with it, if not, associate this pattern with it and similarity score
                        patterns = self.candidate_tuples[t]
                        if patterns is not None:
                            if pattern_best not in [x[0] for x in patterns]:
                                self.candidate_tuples[t].append((pattern_best, sim_best))
                        # If this tuple was not extracted before, associate this pattern with the instance
                        # and the similarity score
                        else:
                            self.candidate_tuples[t].append((pattern_best, sim_best))
                    # update extraction pattern confidence
                    if iter > 0:
                        extraction_pattern.confidence_old = extraction_pattern.confidence
                        extraction_pattern.update_confidence()

                # normalize patterns confidence
                # find the maximum value of confidence and divide all by the maximum
                max_confidence = 0
                for p in self.patterns:
                    if p.confidence > max_confidence:
                        max_confidence = p.confidence

                if max_confidence > 0:
                    for p in self.patterns:
                        p.confidence = float(p.confidence) / float(max_confidence)

                if PRINT_PATTERNS is True:
                    print "\nPatterns:"
                    for p in self.patterns:
                        for t in p.tuples:
                            print "BEF", t.bef_words
                            print "BET", t.bet_words
                            print "AFT", t.aft_words
                            print "========"
                        print "Positive", p.positive
                        print "Negative", p.negative
                        print "Unknown", p.unknown
                        print "Tuples", len(p.tuples)
                        print "Pattern Confidence", p.confidence
                        print "\n"

                # update tuple confidence based on patterns confidence
                print "\n\nCalculating tuples confidence"
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

                # sort tuples by confidence and print
                if PRINT_TUPLES is True:
                    extracted_tuples = self.candidate_tuples.keys()
                    tuples_sorted = sorted(extracted_tuples, key=lambda tpl: tpl.confidence, reverse=True)
                    for t in tuples_sorted:
                        print t.sentence
                        print t.e1, t.e2
                        print t.confidence
                        print "\n"

                if self.config.semantic_drift == 0:
                    # update seed set of tuples to use in next iteration
                    # seeds = { T | conf(T) > instance_confidance }
                    if self.curr_iteration < self.config.number_iterations+1:
                        print "Adding tuples to seed with confidence >=" + str(self.config.instance_confidance)
                        self.seeds_by_iteration[self.curr_iteration] = list()
                        for t in self.candidate_tuples.keys():
                            if t.confidence >= self.config.instance_confidance:
                                seed = Seed(t.e1, t.e2)
                                self.config.seed_tuples.add(seed)
                                # for filtering semantic drift by comparing with previous sentence extractions
                                # keeps tracks of the seeds instances extracted at each iteration
                                self.seeds_by_iteration[self.curr_iteration].append(t)

                elif self.config.semantic_drift == 1 and self.curr_iteration > 0:
                    # update seed set of tuples to use in next iteration
                    # seeds = { T | conf(T) > instance_confidance }
                    if self.curr_iteration < self.config.number_iterations+1:
                        added = 0
                        # gather all previous
                        previous = list()
                        for i in range(self.curr_iteration):
                            previous.extend(self.seeds_by_iteration[i])

                        print "Adding tuples to seed with confidence >=" + str(self.config.instance_confidance)
                        self.seeds_by_iteration[self.curr_iteration] = list()
                        for t in self.candidate_tuples.keys():
                            if t.confidence >= self.config.instance_confidance:
                                # for filtering semantic drift by comparing with previous sentence extractions
                                # keeps tracks of the seeds instances extracted at each iteration
                                self.seeds_by_iteration[self.curr_iteration].append(t)

                        if len(previous) > 0 and len(self.seeds_by_iteration[self.curr_iteration]) > 0:
                            print "Using distributional similarity to filter seeds"
                            print "previous:", len(previous)
                            print "current :", len(self.seeds_by_iteration[self.curr_iteration])
                            count = 0
                            for r in self.seeds_by_iteration[self.curr_iteration]:
                                if count % 1000 == 0:
                                    sys.stdout.write(".")
                                    sys.stdout.flush()
                                avg_sim_previous, avg_sim_current = self.average_similarity(r, self.seeds_by_iteration[self.curr_iteration], previous)
                                if avg_sim_current > avg_sim_previous:
                                    if avg_sim_current-avg_sim_previous > 0.1:
                                        print "ELIMINATED FROM SEEDS:"
                                        print r.e1, '\t', r.e2
                                        print r.sentence
                                        print "avg_sim_previous :", avg_sim_previous
                                        print "avg_sim_current  :", avg_sim_current
                                        print "difference       :", avg_sim_current-avg_sim_previous
                                    else:
                                        seed = Seed(t.e1, t.e2)
                                        self.config.seed_tuples.add(seed)
                                        added += 1
                                else:
                                    seed = Seed(t.e1, t.e2)
                                    self.config.seed_tuples.add(seed)
                                    added += 1

                                count += 1

                        print added, "tuples added"

                elif self.config.semantic_drift == 1 and self.curr_iteration == 0:
                    print "Adding tuples to seed with confidence >=" + str(self.config.instance_confidance)
                    self.seeds_by_iteration[self.curr_iteration] = list()
                    for t in self.candidate_tuples.keys():
                        if t.confidence >= self.config.instance_confidance:
                            seed = Seed(t.e1, t.e2)
                            self.config.seed_tuples.add(seed)
                            # for filtering semantic drift by comparing with previous sentence extractions
                            # keeps tracks of the seeds instances extracted at each iteration
                            self.seeds_by_iteration[self.curr_iteration].append(t)

                # increment the number of iterations
                self.curr_iteration += 1

                #TODO: se há novos tuplos válidos extraidos ou os patterns a alterarem a confianca, entao continua, senao para

        print "\nWriting extracted relationships to disk"
        f_output = open("relationships.txt", "w")
        tmp = sorted(self.candidate_tuples.keys(), reverse=True)
        for t in tmp:
            f_output.write("instance: "+t.e1.encode("utf8")+'\t'+t.e2.encode("utf8")+'\tscore:'+str(t.confidence)+'\n')
            f_output.write("sentence: "+t.sentence.encode("utf8")+'\n')
            f_output.write("pattern_bef: " + str(t.bef_words)+'\n')
            f_output.write("pattern_bet: " + str(t.bet_words)+'\n')
            f_output.write("pattern_aft: " + str(t.aft_words)+'\n')
            if t.passive_voice is False:
                f_output.write("passive voice: False\n")
            elif t.passive_voice is True:
                f_output.write("passive voice: True\n")
            f_output.write("\n")
        f_output.close()

        """
        print "Writing generated patterns to disk"
        f_output = open("patterns.txt", "w")
        tmp = sorted(self.patterns, reverse=True)
        for p in tmp:
            f_output.write("confidence : " + str(p.confidence)+'\n')
            f_output.write("pattern_bef: " + t.bef_words+'\n')
            f_output.write("pattern_bet: " + t.bet_words+'\n')
            f_output.write("pattern_aft: " + t.aft_words+'\n')
            f_output.write("=================================\n")
        f_output.close()
        """

    def similarity_all_1(self, t, extraction_pattern):
        """
        Cosine similarity between all patterns part of a Cluster/Extraction Pattern
        and the vector of a ReVerb pattern extracted from a sentence
        returns the max
        """
        good = 0
        bad = 0
        max_similarity = 0

        for p in list(extraction_pattern.tuples):
            score = self.similarity_3_contexts(t, p)
            if score > max_similarity:
                max_similarity = score
            if score >= self.config.threshold_similarity:
                good += 1
            else:
                bad += 1

        if good >= bad:
            return True, max_similarity
        else:
            return False, 0.0

    def similarity_all_2(self, t, extraction_pattern):
        """
        Cosine similarity between all patterns part of a Cluster/Extraction Pattern
        and the vector of a ReVerb pattern extracted from a sentence
        returns the average
        """
        good = 0
        bad = 0
        max_similarity = 0
        similarities = list()

        for p in list(extraction_pattern.tuples):
            score = self.similarity_3_contexts(t, p)
            if score > max_similarity:
                max_similarity = score
            if score >= self.config.threshold_similarity:
                good += 1
                similarities.append(score)
            else:
                bad += 1

        if good >= bad:
            assert good == len(similarities)
            return True, float(sum(similarities)) / float(good)
        else:
            return False, 0.0

    def similarity_matrix_all_2(self, t, extraction_pattern):
        """
        Cosine similarity between all patterns part of a Cluster/Extraction Pattern
        and the vector of a ReVerb pattern extracted from a sentence
        returns the average
        """
        good = 0
        bad = 0
        max_similarity = 0
        similarities = list()

        for p in list(extraction_pattern.matrixes):
            score = self.sim_matrix_l2(t, p)
            if score > max_similarity:
                max_similarity = score
            if score >= self.config.threshold_similarity:
                good += 1
                similarities.append(score)
            else:
                bad += 1

        if good >= bad:
            assert good == len(similarities)
            return True, float(sum(similarities)) / float(good)
        else:
            return False, 0.0

    def cluster_tuples(self, matched_tuples):
        """
        Single-pass Clustering
        """
        # Initialize: if no patterns exist, first tuple goes to first cluster
        if len(self.patterns) == 0:
            c1 = Pattern(self.config, matched_tuples[0])
            self.patterns.append(c1)
            #print "Pattern Words", self.patterns[0].patterns_words

        # Compute the similarity between an instance with each pattern
        # go through all tuples
        count = 0
        for t in matched_tuples:
            count += 1
            if count % 1000 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
            max_similarity = 0
            max_similarity_cluster_index = 0

            # go through all patterns(clusters of tuples) and find the one with the
            # highest similarity score
            for i in range(0, len(self.patterns), 1):
                extraction_pattern = self.patterns[i]
                # compute the similarity between the instance vector and each vector from a pattern
                # if majority is above threshold
                try:
                    #accept, score = self.similarity_all_1(t, extraction_pattern)
                    accept, score = self.similarity_all_2(t, extraction_pattern)
                    if accept is True and score > max_similarity:
                        max_similarity = score
                        max_similarity_cluster_index = i
                except Exception, e:
                    print "Error! Tuple and Extraction pattern are empty!"
                    print e
                    print "tuple"
                    print t.sentence
                    print t.e1, '\t', t.e2
                    print extraction_pattern
                    sys.exit(0)

            # if max_similarity < min_degree_match create a new cluster having this tuple as the centroid
            if max_similarity < self.config.threshold_similarity:
                c = Pattern(self.config, t)
                self.patterns.append(c)
                #print "New Cluster", c.patterns_words
                #print "\n"

            # if max_similarity >= min_degree_match add to the cluster with the highest similarity
            else:
                #print "\n"
                #print "good match", t.patterns_words, self.patterns[max_similarity_cluster_index], max_similarity
                self.patterns[max_similarity_cluster_index].add_tuple(t)
                #print "Cluster", self.patterns[max_similarity_cluster_index].patterns_words

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

    @staticmethod
    def tokenize(text):
        return [word for word in word_tokenize(text.lower()) if word not in stopwords.words('english')]

    def cluster_dbscan_vectors(self, matched_tuples):
        # build a matrix with all the pairwise distances between all the matrix representing sentences
        matrix = np.zeros((len(matched_tuples), len(matched_tuples)))
        print "Calculating pairwise distances..."
        for ex1 in range(len(matched_tuples)):
            for ex2 in range(len(matched_tuples)):
                similarity = self.similarity_3_contexts(matched_tuples[ex1], matched_tuples[ex2])
                print matched_tuples[ex1].sentence
                print matched_tuples[ex2].sentence
                # dbscan deals with distances, transform similarity into a distance
                matrix[ex1, ex2] = 1-similarity

        # normalized all the distances
        matrix = np.divide(matrix, matrix.max())

        # perform DBSCAN
        db = DBSCAN(eps=0.2, min_samples=2, metric='precomputed')
        db.fit(matrix)
        clusters = defaultdict(list)

        print db.labels_
        print "labels", len(db.labels_)
        print "examples", len(matched_tuples)
        assert len(db.labels_) == len(matched_tuples)

        # aggregate results by label
        # -1 represents noise, discarded
        for v in range(len(matched_tuples)):
            label = db.labels_[v]
            if label > -1:
                clusters[label].append(matched_tuples[v])

        """
        for k in clusters:
            print "label", k
            for rel in clusters[k]:
                print rel.sentence
            print "\n"
        """

        return clusters

    def cluster_dbscan(self, matched_tuples):
        # build a matrix with all the pairwise distances between all the matrix representing sentences
        matrix = np.zeros((len(matched_tuples), len(matched_tuples)))
        print "Calculating pairwise distances..."
        for ex1 in range(len(matched_tuples)):
            for ex2 in range(len(matched_tuples)):
                dist = self.sim_matrix_l2(matched_tuples[ex1].matrix, matched_tuples[ex2].matrix)
                matrix[ex1, ex2] = dist

        # normalized all the distances
        matrix = np.divide(matrix, matrix.max())

        # perform DBSCAN
        db = DBSCAN(eps=0.2, min_samples=2, metric='precomputed')
        db.fit(matrix)
        clusters = defaultdict(list)

        print db.labels_
        print "labels", len(db.labels_)
        print "examples", len(matched_tuples)
        assert len(db.labels_) == len(matched_tuples)

        # aggregate results by label
        # -1 represents noise, discarded
        for v in range(len(matched_tuples)):
            label = db.labels_[v]
            if label > -1:
                clusters[label].append(matched_tuples[v])

        for k in clusters:
            """
            print "label", k
            for rel in clusters[k]:
                print rel.sentence
            print "\n"
            """
            p = PatternMatrices(k)
            self.patterns.append(p)

    @staticmethod
    def sim_matrix_l2(matrix1, matrix2):
        diff_matrix = matrix1-matrix2
        diff_matrix = np.power(diff_matrix, 2)
        sum_diff = np.sum(diff_matrix)
        return np.sqrt(sum_diff)


def main():
    configuration = sys.argv[1]
    sentences_file = sys.argv[2]
    seeds_file = sys.argv[3]
    negative_seeds = sys.argv[4]
    # threshold similarity for clustering/extracting instances
    similarity = sys.argv[5]
    # confidence threshold of an instance to used as seed
    confidance = sys.argv[6]
    breads = BREDS(configuration, seeds_file, negative_seeds, float(similarity), float(confidance), sentences_file)
    if sentences_file.endswith('.pkl'):
        print "Loading pre-processed sentences", sentences_file
        breads.init_bootstrapp(tuples=sentences_file)
    else:
        breads.generate_tuples(sentences_file)
        breads.init_bootstrapp(tuples=None)


if __name__ == "__main__":
    main()