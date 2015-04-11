#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import cPickle
import sys
import os
import codecs
import operator

from Sentence import Sentence
from Pattern import Pattern
from Config import Config
from Tuple import Tuple
from Seed import Seed

from nltk import PunktWordTokenizer
from nltk.corpus import stopwords
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise

from Word2VecWrapper import Word2VecWrapper
from numpy import dot
from gensim import matutils
from collections import defaultdict

# usefull stuff for debugging
PRINT_TUPLES = False
PRINT_PATTERNS = True


class BREADS(object):

    def __init__(self, config_file, seeds_file, negative_seeds, similarity, confidance):
        self.patterns = list()
        self.processed_tuples = list()
        self.candidate_tuples = defaultdict(list)
        self.config = Config(config_file, seeds_file, negative_seeds, similarity, confidance)

    def generate_tuples(self, sentences_file):
        """
        Generate tuples instances from a text file with sentences
        where named entities are already tagged
        """
        try:
            os.path.isfile("processed_tuples.pkl")
            f = open("processed_tuples.pkl", "r")
            print "\nLoading processed tuples from disk..."
            self.processed_tuples = cPickle.load(f)
            f.close()
            print len(self.processed_tuples), "tuples loaded"

        except IOError:
            print "\nGenerating relationship instances from sentences"
            f_sentences = codecs.open(sentences_file, encoding='utf-8')
            count = 0
            for line in f_sentences:
                count += 1
                if count % 10000 == 0:
                    sys.stdout.write(".")
                sentence = Sentence(line.strip(), self.config.e1_type, self.config.e2_type, self.config.max_tokens_away, self.config.min_tokens_away, self.config.context_window_size)
                for rel in sentence.relationships:
                    if rel.arg1type == self.config.e1_type and rel.arg2type == self.config.e2_type:
                        t = Tuple(rel.ent1, rel.ent2, rel.sentence, rel.before, rel.between, rel.after, self.config)
                        #if len(t.patterns_vectors) >= 1:
                        self.processed_tuples.append(t)
            f_sentences.close()

            print "\n", len(self.processed_tuples), "tuples generated"
            print "Writing generated tuples to disk"
            f = open("processed_tuples.pkl", "wb")
            cPickle.dump(self.processed_tuples, f)
            f.close()

    @staticmethod
    def drift(r, i):
        pass
        #TODO:
        # select all instances extracted in iterations < i
        #   P = instances extracted at iteration in previous iterations
        # select all instances extracted in iteration i
        #   Z = instances extracted at iteration i

        # calculate similarity
        # for t in P:
        #   avg_sim += sim(t,r)
        # avg_sim_P /= len(P)

        # for t in Z:
        #   avg_sim += sim(t,r)
        # avg_sim_Z /= len(Z)

        # return (avg_sim_P / avg_sim_Z)

    def start(self, tuples):
        """
        starts a bootstrap iteration
        """
        if tuples is not None:
            f = open(tuples, "r")
            print "\nLoading processed tuples from disk..."
            self.processed_tuples = cPickle.load(f)
            f.close()
            print len(self.processed_tuples), "tuples loaded"

        i = 0
        while i <= self.config.number_iterations:
            print "\nStarting iteration", i
            print "\nLooking for seed matches of:"
            for s in self.config.seed_tuples:
                print s.e1, '\t', s.e2

            # Looks for sentences macthing the seed instances
            count_matches, matched_tuples = self.match_seeds_tuples()

            """
            for t in matched_tuples:
                print t.e1, '\t', t.e2
                print t.sentence
                print t.bef_vector
                print t.bet_vector
                print t.aft_vector
                print "\n"
            """

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
                #self.cluster_dbscan(self, matched_tuples)
                self.cluster_tuples(self, matched_tuples)

                # Eliminate patterns supported by less than 'min_pattern_support' tuples
                new_patterns = [p for p in self.patterns if len(p.tuples) >= 2]
                self.patterns = new_patterns
                print "\n", len(self.patterns), "patterns generated"

                if PRINT_PATTERNS is True:
                    print "\nPatterns:"
                    for p in self.patterns:
                        print "Pattern", p.patterns_words
                        print "Positive", p.positive
                        print "Negative", p.negative
                        print "Unknown", p.unknown
                        print "Tuples", len(p.tuples)
                        print "Pattern Confidence", p.confidence
                        print "\n"

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
                count = 0
                for t in self.processed_tuples:
                    count += 1
                    if count % 1000 == 0:
                        sys.stdout.write(".")
                    sim_best = 0
                    for extraction_pattern in self.patterns:

                        if self.config.similarity == "all":
                            accept, score = similarity_all(t, extraction_pattern, self.config)
                            if accept is True:
                                extraction_pattern.update_selectivity(t, self.config)
                                if score > sim_best:
                                    sim_best = score
                                    pattern_best = extraction_pattern

                        elif self.config.similarity == "single-vector":
                            score = similarity_sum(t.patterns_vectors[0], extraction_pattern, self.config)
                            if score > self.config.threshold_similarity:
                                extraction_pattern.update_selectivity(t, self.config)
                                if score > sim_best:
                                    sim_best = score
                                    pattern_best = extraction_pattern

                    if sim_best >= self.config.threshold_similarity:
                        # if this tuple was already extracted, check if this extraction pattern is already associated
                        # with it. if not associate this pattern with it and similarity score
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
                        print "Pattern    :", p.patterns_words
                        print "Positive   :", p.positive
                        print "Negative   :", p.negative
                        print "Unknown    :", p.unknown
                        print "Tuples     :", len(p.tuples)
                        print "Confidence :", p.confidence
                        print "\n"

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

                # sort tuples by confidence and print
                if PRINT_TUPLES is True:
                    extracted_tuples = self.candidate_tuples.keys()
                    tuples_sorted = sorted(extracted_tuples, key=lambda t: t.confidence, reverse=True)
                    for t in tuples_sorted:
                        best_pattern = self.candidate_tuples[t]
                        print t.e1, t.patterns_words, t.e2, t.confidence
                        for b in best_pattern:
                            print b[0].patterns_words, b[1]
                        print "\n"

                # update seed set of tuples to use in next iteration
                # seeds = { T | Conf(T) > min_tuple_confidence }
                if i+1 < self.config.number_iterations:
                    print "Adding tuples to seed with confidence =>" + str(self.config.instance_confidance)
                    for t in self.candidate_tuples.keys():
                        if t.confidence >= self.config.instance_confidance:
                            seed = Seed(t.e1, t.e2)
                            self.config.seed_tuples.add(seed)

                # increment the number of iterations
                i += 1

        print "\nWriting extracted relationships to disk"
        f_output = open("relationships.txt", "w")
        tmp = sorted(self.candidate_tuples.keys(), reverse=True)
        for t in tmp:
            f_output.write("instance: "+t.e1.encode("utf8")+'\t'+t.e2.encode("utf8")+'\tscore:'+str(t.confidence)+'\n')
            f_output.write("sentence: "+t.sentence.encode("utf8")+'\n')
            f_output.write("pattern: "+' '.join(t.patterns_words)+'\n')
            if t.passive_voice is False:
                f_output.write("passive voice: False\n")
            elif t.passive_voice is True:
                f_output.write("passive voice: True\n")
            f_output.write("\n")
        f_output.close()

        print "Writing generated patterns to disk"
        f_output = open("patterns.txt", "w")
        tmp = sorted(self.patterns, reverse=True)
        for p in tmp:
            f_output.write(str(p.patterns_words)+'\t'+str(p.confidence)+'\n')
        f_output.close()

    @staticmethod
    def cluster_dbscan(self, matched_tuples):
        #TODO: usar o self.config para ler os params eps=0.1, min_samples=2
        # add all bet_vectors
        vectors = []
        for t in matched_tuples:
            vectors.append(t.patterns_vectors[0])

        # build a matrix with all the pairwise distances between all the vectors
        matrix = pairwise.pairwise_distances(numpy.array(vectors), metric='cosine', n_jobs=-1)

        # perform DBSCAN
        db = DBSCAN(eps=0.1, min_samples=2, metric='precomputed')
        db.fit(matrix)
        clusters = defaultdict(list)

        # aggregate results by label, discard -1 which is noise
        for v in range(0, len(vectors)-1):
            label = db.labels_[v]
            if label > -1:
                clusters[label].append(matched_tuples[v])

        # create Patterns instances from the clustered Tuples
        patterns_generated = list()
        for k in clusters.keys():
            c = Pattern(self.config, clusters[k][0])
            for p in range(1, len(clusters[k])):
                c.add_tuple(clusters[k][p])
            patterns_generated.append(c)

        """
        print "\nPatterns generated in this iteration"
        for p in patterns_generated:
            print p, len(p.tuples)
        """

        print "\nAlready existing patterns"
        for p in self.patterns:
            print p

        if len(self.patterns) == 0:
            # no patterns exist, these are the initial patterns
            for p in patterns_generated:
                self.patterns.append(p)
        else:
            # Only compare new patterns
            print "\nLooking for repetead patterns"
            if len(self.patterns) > 0:
                new_patterns = list()
                for new_p in patterns_generated:
                    equals = False
                    for p in self.patterns:
                        if p.patterns_words == new_p.patterns_words:
                            equals = True
                            continue
                    if equals is False:
                        new_patterns.append(new_p)

            # compare generated patterns similarity with already existing patterns
            if len(new_patterns) > 0:

                print "\nNew patterns: ", len(new_patterns)

                for pattern in new_patterns:
                    added = False
                    for i in range(0, len(self.patterns)):
                        if added is True:
                            break
                        else:
                            extraction_pattern = self.patterns[i]
                            max_similarity = 0
                            # each pattern has one or more vectors representing ReVerb patterns
                            # compute the similarity between the instance vector and each vector from a pattern

                            # 1 - compare similarity with all vectors, majority decides if is above threshold
                            if self.config.similarity == "all":
                                print "Similarity between: ", pattern.patterns_words, " and", extraction_pattern.patterns_words
                                accept, score = similarity_all_dbscan(pattern, extraction_pattern, self.config)
                                if accept is True and score > max_similarity:
                                    max_similarity = score
                                    max_similarity_cluster_index = i

                            # 2 - similarity calculate with just one vector, representd by the sum of all
                            #     tuple's vectors in a pattern/cluster
                            elif self.config.similarity == "single-vector":
                                score = similarity_sum(pattern.patterns_vectors[0], extraction_pattern, self.config)
                                if score > max_similarity:
                                    max_similarity = score
                                    max_similarity_cluster_index = i

                            # create new Pattern
                            if max_similarity < self.config.threshold_similarity:
                                print "New pattern created:", pattern
                                self.patterns.append(pattern)
                                added = True

                            # if max_similarity >= min_degree_match add to the cluster with the highest similarity
                            else:
                                self.patterns[max_similarity_cluster_index].add_pattern(pattern)
                                print "New pattern added to:", self.patterns[max_similarity_cluster_index], pattern
                                added = True

                            print "\n"

    @staticmethod
    def cluster_tuples(self, matched_tuples):
        """
        Single-pass Clustering
        """
        # Initialize: if no patterns exist, first tuple goes to first cluster
        if len(self.patterns) == 0:
            c1 = Pattern(self.config, matched_tuples[0])
            self.patterns.append(c1)

            print "Pattern Words", self.patterns[0].patterns_words

        # Compute the similarity between an instance with each pattern
        # go through all tuples
        count = 0
        for t in matched_tuples:
            count += 1
            if count % 500 == 0:
                sys.stdout.write(".")
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
                if self.config.similarity == "all":
                    try:
                        accept, score = similarity_all(t, extraction_pattern, self.config)
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

                # 2 - similarity calculate with just one vector, representd by the sum of all
                #     tuple's vectors in a pattern/cluster
                elif self.config.similarity == "single-vector":
                    score = similarity_sum(t.patterns_vectors[0], extraction_pattern, self.config)
                    if score > max_similarity:
                        max_similarity = score
                        max_similarity_cluster_index = i

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


def similarity_sum(t_pattern, extraction_pattern, config):
    """
    Cosine similarity between a Cluster/Extraction Pattern represented as a single vector
    and the vector of a ReVerb pattern extracted from a sentence
    """
    extraction_pattern.calculate_single_vector(config)
    score = dot(matutils.unitvec(t_pattern), matutils.unitvec(extraction_pattern.single_vector))
    return score


def similarity_all_dbscan(pattern, extraction_pattern, config):
    """
    Cosine similarity between all patterns part of a Cluster/Extraction Pattern
    and the vector of a ReVerb pattern extracted from a sentence
    """
    good = 0
    bad = 0
    max_similarity = 0
    for p in list(extraction_pattern.patterns_words):
        tokens = tokenize(p)
        vector_p = Word2VecWrapper.pattern2vector_sum(tokens, config)
        for w in pattern.patterns_words:
            vector_w = Word2VecWrapper.pattern2vector_sum(tokenize(w), config)
            score = dot(matutils.unitvec(vector_w), matutils.unitvec(vector_p))
            print "p:", p, "w:", w, score

            if score > max_similarity:
                max_similarity = score
            if score >= config.threshold_similarity:
                good += 1
            else:
                bad += 1

    print "good", good
    print "bad", bad
    if good >= bad:
        return True, max_similarity
    else:
        return False, 0.0


def similarty_3_contexts(p, t, config):
    (bef, bet, aft) = (0, 0, 0)
    """
    print "Tuple"
    print t.e1, '\t', t.e2
    print t.sentence
    print t.bef_vector
    print t.bet_vector
    print t.aft_vector

    print "Pattern"
    print p.e1, '\t', p.e2
    print p.bef_vector
    print p.sentence
    print p.bet_vector
    print p.aft_vector
    """

    if t.bef_vector is not None and p.bef_vector is not None:
        #bef = cossim(t.bef_vector, p.bef_vector)
        bef = dot(matutils.unitvec(t.bef_vector), matutils.unitvec(p.bef_vector))

    if t.bet_vector is not None and p.bet_vector is not None:
        #bet = cossim(t.bet_vector, p.bet_vector)
        bet = dot(matutils.unitvec(t.bet_vector), matutils.unitvec(p.bet_vector))

    if t.aft_vector is not None and p.aft_vector is not None:
        #aft = cossim(t.aft_vector, p.aft_vector)
        aft = dot(matutils.unitvec(t.aft_vector), matutils.unitvec(p.aft_vector))

    print "bef", bef
    print "bet", bet
    print "aft", aft
    print "score", config.alpha*bef + config.beta*bet + config.gamma*aft
    print "\n"
    return config.alpha*bef + config.beta*bet + config.gamma*aft


def similarity_all(t, extraction_pattern, config):
    """
    Cosine similarity between all patterns part of a Cluster/Extraction Pattern
    and the vector of a ReVerb pattern extracted from a sentence
    """
    good = 0
    bad = 0
    max_similarity = 0

    if config.vector == 'version_3':
        for p in list(extraction_pattern.tuples):
            score = similarty_3_contexts(t, p, config)
            if score >= config.threshold_similarity:
                good += 1
            else:
                bad += 1

        if good >= bad:
            return True, max_similarity
        else:
            return False, 0.0

    elif config.vector == 'version_2' or config.vector == 'version_1':
        for vector in list(extraction_pattern.vectors):
            score = dot(matutils.unitvec(t.vector), matutils.unitvec(vector))
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

    """
    for p in list(extraction_pattern.patterns_words):
        tokens = PunktWordTokenizer().tokenize(p)
        vector = Word2VecWrapper.pattern2vector_sum(tokens, config)
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
    """



def tokenize(text):
    return [word for word in PunktWordTokenizer().tokenize(text.lower()) if word not in stopwords.words('english')]


def main():
    configuration = sys.argv[1]
    sentences_file = sys.argv[2]
    seeds_file = sys.argv[3]
    negative_seeds = sys.argv[4]
    similarity = sys.argv[5]
    confidance = sys.argv[6]
    breads = BREADS(configuration, seeds_file, negative_seeds, float(similarity), float(confidance))
    if sentences_file.endswith('.pkl'):
        print "Loading pre-processed sentences", sentences_file
        breads.start(tuples=sentences_file)
    else:
        breads.generate_tuples(sentences_file)
        breads.start(tuples=None)


if __name__ == "__main__":
    main()