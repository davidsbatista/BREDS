#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import cPickle
import sys
import codecs
import operator
import multiprocessing
import Queue

from nltk.corpus import stopwords
from nltk import word_tokenize
from numpy import dot
from gensim import matutils
from collections import defaultdict

from BREDS.PatternPoS import Pattern
from BREDS.Config import Config
from BREDS.TuplePoS import Tuple
from Common.Sentence import Sentence
from Common.Seed import Seed

# usefull stuff for debugging
PRINT_TUPLES = False
PRINT_PATTERNS = False


class BREDS(object):

    def __init__(self, config_file, seeds_file, negative_seeds, similarity, confidance, sentences_file):

        self.m = multiprocessing.Manager()
        self.queue = self.m.Queue()
        self.num_cpus = multiprocessing.cpu_count()

        self.processed_tuples = list()
        self.candidate_tuples = defaultdict(list)

        self.curr_iteration = 0
        self.patterns = list()

        self.config = Config(config_file, seeds_file, negative_seeds, similarity, confidance, sentences_file)

    def generate_tuples(self, sentences_file):
        """
        Generate tuples instances from a text file with sentences where named entities are already tagged
        """
        self.config.read_word2vec()
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

    def init_bootstrapp(self, tuples):
        """
        starts a bootstrap iteration
        """
        if tuples is not None:
            f = open(tuples, "r")
            print "Loading pre-processed sentences", tuples
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

                # create copies of patterns to be passed to each process, each copy will have its number of positive and
                # negative matches accordingly to the number of correct/incorrect instances extracted
                patterns_copies = [self.m.list() for _ in range(self.num_cpus)]
                for i in range(len(patterns_copies)):
                    patterns_copies[i] = list(self.patterns)

                # copy all tuples into a Queue shared by all processes
                for t in self.processed_tuples:
                    self.queue.put(t)

                # each distinct process receives as arguments:
                #   - a shared Queue of the tuples
                #   - a copy of all the extraction patterns
                #   - a list to store the tuples which matched a pattern
                tuples_results = [defaultdict(list) for _ in range(self.num_cpus)]
                processes = [multiprocessing.Process(target=self.find_instances,
                                                     args=(patterns_copies[i], tuples_results[i], self.queue))
                             for i in range(self.num_cpus)]

                print "Lauching ", len(processes), " processes"

                for proc in processes:
                    proc.start()

                for proc in processes:
                    proc.join()

                # TODO:
                # patterns_copies -> patterns com positive/negative/unknown que precisam de ser agregados
                # tuples_results é o candidate tuples, cada tuple é unico é necessário apenas fazer o merge
                """
                for i in range(len(patterns_copies)):
                    for p_copy in patterns_copies[i]:
                        for p_original in self.patterns:
                            if p_copy.id == p_original.id:
                                print "Original"
                                print p_original.positive
                                print p_original.negative
                                print "Copy"
                                print p_copy.positive
                                print p_copy.negative

                                # update extraction pattern confidence
                                if iter > 0:
                                    extraction_pattern.confidence_old = extraction_pattern.confidence
                                    extraction_pattern.update_confidence()
                """

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

                # update seed set of tuples to use in next iteration
                # seeds = { T | conf(T) > instance_confidance }
                if self.curr_iteration < self.config.number_iterations+1:
                    print "Adding tuples to seed with confidence >=" + str(self.config.instance_confidance)
                    for t in self.candidate_tuples.keys():
                        if t.confidence >= self.config.instance_confidance:
                            seed = Seed(t.e1, t.e2)
                            self.config.seed_tuples.add(seed)

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

    def similarity_all(self, t, extraction_pattern):
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
                    accept, score = self.similarity_all(t, extraction_pattern)
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

    def find_instances(self, patterns, candidate_tuples, instances):
        while True:
            try:
                t = instances.get_nowait()
                for extraction_pattern in patterns:
                    # measure similarity towards an extraction pattern
                    sim_best = 0
                    accept, score = self.similarity_all(t, extraction_pattern)
                    if accept is True:
                        extraction_pattern.update_selectivity(t, self.config)
                        if score > sim_best:
                            sim_best = score
                            pattern_best = extraction_pattern

                    # if its above a threshold associated the pattern with it
                    if sim_best >= self.config.threshold_similarity:
                        print multiprocessing.current_process(), instances.qsize()
                        # if this tuple was already extracted, check if this extraction pattern is already associated
                        # with it, if not, associate this pattern with it and similarity score
                        patterns = candidate_tuples[tuple]
                        if patterns is not None:
                            if pattern_best not in [x[0] for x in patterns]:
                                candidate_tuples[t].append((pattern_best, sim_best))
                        # If this tuple was not extracted before, associate this pattern with the instance
                        # and the similarity score
                        else:
                            candidate_tuples[t].append((pattern_best, sim_best))
                        if self.curr_iteration > 0:
                            extraction_pattern.confidence_old = extraction_pattern.confidence
                            extraction_pattern.update_confidence()

            except Queue.Empty:
                print "Queue is Empty"
                break


def main():
    if len(sys.argv) != 7:
        print "\nBREDS.py paramters.cfg sentences_file seeds_file_positive seeds_file_negative similarity_threshold" \
              " confidance_threshold\n"
        sys.exit(0)
    else:
        configuration = sys.argv[1]
        sentences_file = sys.argv[2]
        seeds_file = sys.argv[3]
        negative_seeds = sys.argv[4]
        similarity = sys.argv[5]    # threshold similarity for clustering/extracting instances
        confidance = sys.argv[6]    # confidence threshold of an instance to used as seed
        breads = BREDS(configuration, seeds_file, negative_seeds, float(similarity), float(confidance), sentences_file)
        if sentences_file.endswith('.pkl'):
            breads.init_bootstrapp(tuples=sentences_file)
        else:
            breads.generate_tuples(sentences_file)
            breads.init_bootstrapp(tuples=None)


if __name__ == "__main__":
    main()