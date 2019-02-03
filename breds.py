import sys
import os
import pickle
import operator

from numpy import dot
from gensim import matutils
from collections import defaultdict
from nltk.data import load

from breds.seed import Seed
from breds.pattern import Pattern
from breds.config import Config
from breds.tuple import Tuple
from breds.sentence import Sentence

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

# useful for debugging
PRINT_TUPLES = False
PRINT_PATTERNS = False


class BREDS(object):

    def __init__(self, config_file, seeds_file, negative_seeds, similarity, confidence):
        self.curr_iteration = 0
        self.patterns = list()
        self.processed_tuples = list()
        self.candidate_tuples = defaultdict(list)
        self.config = Config(config_file, seeds_file, negative_seeds, similarity, confidence)

    def generate_tuples(self, sentences_file):
        """
        Generate tuples instances from a text file with sentences where named entities are
        already tagged

        :param sentences_file:
        """
        if os.path.exists("processed_tuples.pkl"):

            with open("processed_tuples.pkl", "rb") as f_in:
                print("\nLoading processed tuples from disk...")
                self.processed_tuples = pickle.load(f_in)
            print(len(self.processed_tuples), "tuples loaded")

        else:

            # load needed stuff, word2vec model and a pos-tagger
            self.config.read_word2vec()
            tagger = load('taggers/maxent_treebank_pos_tagger/english.pickle')

            print("\nGenerating relationship instances from sentences")
            with open(sentences_file, encoding='utf-8') as f_sentences:
                count = 0
                for line in f_sentences:
                    if line.startswith("#"):
                        continue
                    count += 1
                    if count % 10000 == 0:
                        sys.stdout.write(".")

                    sentence = Sentence(line.strip(),
                                        self.config.e1_type,
                                        self.config.e2_type,
                                        self.config.max_tokens_away,
                                        self.config.min_tokens_away,
                                        self.config.context_window_size, tagger,
                                        self.config)

                    for rel in sentence.relationships:
                        t = Tuple(rel.e1, rel.e2,
                                  rel.sentence, rel.before, rel.between, rel.after,
                                  self.config)
                        self.processed_tuples.append(t)
                print("\n", len(self.processed_tuples), "tuples generated")

            print("Writing generated tuples to disk")
            with open("processed_tuples.pkl", "wb") as f_out:
                pickle.dump(self.processed_tuples, f_out)

    def similarity_3_contexts(self, p, t):
        (bef, bet, aft) = (0, 0, 0)

        if t.bef_vector is not None and p.bef_vector is not None:
            bef = dot(matutils.unitvec(t.bef_vector), matutils.unitvec(p.bef_vector))

        if t.bet_vector is not None and p.bet_vector is not None:
            bet = dot(matutils.unitvec(t.bet_vector), matutils.unitvec(p.bet_vector))

        if t.aft_vector is not None and p.aft_vector is not None:
            aft = dot(matutils.unitvec(t.aft_vector), matutils.unitvec(p.aft_vector))

        return self.config.alpha*bef + self.config.beta*bet + self.config.gamma*aft

    def similarity_all(self, t, extraction_pattern):

        # calculates the cosine similarity between all patterns part of a
        # cluster (i.e., extraction pattern) and the vector of a ReVerb pattern
        # extracted from a sentence;

        # returns the max similarity scores

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

    def match_seeds_tuples(self):
        # checks if an extracted tuple matches seeds tuples
        matched_tuples = list()
        count_matches = dict()
        for t in self.processed_tuples:
            for s in self.config.positive_seed_tuples:
                if t.e1 == s.e1 and t.e2 == s.e2:
                    matched_tuples.append(t)
                    try:
                        count_matches[(t.e1, t.e2)] += 1
                    except KeyError:
                        count_matches[(t.e1, t.e2)] = 1

        return count_matches, matched_tuples

    def write_relationships_to_disk(self):
        print("\nWriting extracted relationships to disk")
        f_output = open("relationships.txt", "w")
        tmp = sorted(list(self.candidate_tuples.keys()), reverse=True)
        for t in tmp:
            f_output.write("instance: " + t.e1+'\t'+t.e2+'\tscore:'+str(t.confidence)+'\n')
            f_output.write("sentence: "+t.sentence+'\n')
            f_output.write("pattern_bef: "+t.bef_words+'\n')
            f_output.write("pattern_bet: "+t.bet_words+'\n')
            f_output.write("pattern_aft: "+t.aft_words+'\n')
            if t.passive_voice is False:
                f_output.write("passive voice: False\n")
            elif t.passive_voice is True:
                f_output.write("passive voice: True\n")
            f_output.write("\n")
        f_output.close()

    def init_bootstrap(self, tuples):

        # starts a bootstrap iteration

        if tuples is not None:
            f = open(tuples, "r")
            print("\nLoading processed tuples from disk...")
            self.processed_tuples = pickle.load(f)
            f.close()
            print(len(self.processed_tuples), "tuples loaded")

        self.curr_iteration = 0
        while self.curr_iteration <= self.config.number_iterations:
            print("==========================================")
            print("\nStarting iteration", self.curr_iteration)
            print("\nLooking for seed matches of:")
            for s in self.config.positive_seed_tuples:
                print(s.e1, '\t', s.e2)

            # Looks for sentences matching the seed instances
            count_matches, matched_tuples = self.match_seeds_tuples()

            if len(matched_tuples) == 0:
                print("\nNo seed matches found")
                sys.exit(0)

            else:
                print("\nNumber of seed matches found")
                sorted_counts = sorted(
                    list(count_matches.items()),
                    key=operator.itemgetter(1),
                    reverse=True
                )
                for t in sorted_counts:
                    print(t[0][0], '\t', t[0][1], t[1])

                print("\n", len(matched_tuples), "tuples matched")

                # Cluster the matched instances, to generate
                # patterns/update patterns
                print("\nClustering matched instances to generate patterns")
                self.cluster_tuples(matched_tuples)

                # Eliminate patterns supported by less than
                # 'min_pattern_support' tuples
                new_patterns = [p for p in self.patterns if len(p.tuples) >
                                self.config.min_pattern_support]
                self.patterns = new_patterns

                print("\n", len(self.patterns), "patterns generated")

                if PRINT_PATTERNS is True:
                    count = 1
                    print("\nPatterns:")
                    for p in self.patterns:
                        print(count)
                        for t in p.tuples:
                            print("BEF", t.bef_words)
                            print("BET", t.bet_words)
                            print("AFT", t.aft_words)
                            print("========")
                            print("\n")
                        count += 1

                if self.curr_iteration == 0 and len(self.patterns) == 0:
                    print("No patterns generated")
                    sys.exit(0)

                # Look for sentences with occurrence of seeds
                # semantic types (e.g., ORG - LOC)
                # This was already collect and its stored in:
                # self.processed_tuples
                #
                # Measure the similarity of each occurrence with each
                # extraction pattern and store each pattern that has a
                # similarity higher than a given threshold
                #
                # Each candidate tuple will then have a number of patterns
                # that extracted it each with an associated degree of match.
                print("Number of tuples to be analyzed:", len(self.processed_tuples))

                print("\nCollecting instances based on extraction patterns")
                count = 0

                for t in self.processed_tuples:

                    count += 1
                    if count % 1000 == 0:
                        sys.stdout.write(".")
                        sys.stdout.flush()

                    sim_best = 0
                    for extraction_pattern in self.patterns:
                        accept, score = self.similarity_all(
                            t, extraction_pattern
                        )
                        if accept is True:
                            extraction_pattern.update_selectivity(
                                t, self.config
                            )
                            if score > sim_best:
                                sim_best = score
                                pattern_best = extraction_pattern

                    if sim_best >= self.config.threshold_similarity:
                        # if this tuple was already extracted, check if this
                        # extraction pattern is already associated with it,
                        # if not, associate this pattern with it and store the
                        # similarity score
                        patterns = self.candidate_tuples[t]
                        if patterns is not None:
                            if pattern_best not in [x[0] for x in patterns]:
                                self.candidate_tuples[t].append(
                                    (pattern_best, sim_best)
                                )

                        # If this tuple was not extracted before
                        # associate this pattern with the instance
                        # and the similarity score
                        else:
                            self.candidate_tuples[t].append(
                                (pattern_best, sim_best)
                            )

                # update all patterns confidence
                for p in self.patterns:
                    p.update_confidence(self.config)

                if PRINT_PATTERNS is True:
                    print("\nPatterns:")
                    for p in self.patterns:
                        for t in p.tuples:
                            print("BEF", t.bef_words)
                            print("BET", t.bet_words)
                            print("AFT", t.aft_words)
                            print("========")
                        print("Positive", p.positive)
                        print("Negative", p.negative)
                        print("Unknown", p.unknown)
                        print("Tuples", len(p.tuples))
                        print("Pattern Confidence", p.confidence)
                        print("\n")

                # update tuple confidence based on patterns confidence
                print("\n\nCalculating tuples confidence")
                for t in list(self.candidate_tuples.keys()):
                    confidence = 1
                    t.confidence_old = t.confidence
                    for p in self.candidate_tuples.get(t):
                        confidence *= 1 - (p[0].confidence * p[1])
                    t.confidence = 1 - confidence

                # sort tuples by confidence and print
                if PRINT_TUPLES is True:
                    extracted_tuples = list(self.candidate_tuples.keys())
                    tuples_sorted = sorted(extracted_tuples, key=lambda tpl: tpl.confidence,
                                           reverse=True)
                    for t in tuples_sorted:
                        print(t.sentence)
                        print(t.e1, t.e2)
                        print(t.confidence)
                        print("\n")

                print("Adding tuples to seed with confidence >= {}".format(
                    str(self.config.instance_confidence)))
                for t in list(self.candidate_tuples.keys()):
                    if t.confidence >= self.config.instance_confidence:
                        seed = Seed(t.e1, t.e2)
                        self.config.positive_seed_tuples.add(seed)

                # increment the number of iterations
                self.curr_iteration += 1

        self.write_relationships_to_disk()

    def cluster_tuples(self, matched_tuples):
        # this is a single-pass clustering
        # Initialize: if no patterns exist, first tuple goes to first cluster
        if len(self.patterns) == 0:
            c1 = Pattern(matched_tuples[0])
            self.patterns.append(c1)

        count = 0
        for t in matched_tuples:
            count += 1
            if count % 1000 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
            max_similarity = 0
            max_similarity_cluster_index = 0

            # go through all patterns(clusters of tuples) and find the one
            # with the highest similarity score
            for i in range(0, len(self.patterns), 1):
                extraction_pattern = self.patterns[i]
                accept, score = self.similarity_all(t, extraction_pattern)
                if accept is True and score > max_similarity:
                    max_similarity = score
                    max_similarity_cluster_index = i

            # if max_similarity < min_degree_match create a new cluster having
            #  this tuple as the centroid
            if max_similarity < self.config.threshold_similarity:
                c = Pattern(t)
                self.patterns.append(c)

            # if max_similarity >= min_degree_match add to the cluster with
            # the highest similarity
            else:
                self.patterns[max_similarity_cluster_index].add_tuple(t)


def main():
    if len(sys.argv) != 7:
        print("\nBREDS.py parameters sentences positive_seeds negative_seeds "
              "similarity confidence\n")
        sys.exit(0)
    else:
        configuration = sys.argv[1]
        sentences_file = sys.argv[2]
        seeds_file = sys.argv[3]
        negative_seeds = sys.argv[4]
        similarity = float(sys.argv[5])
        confidence = float(sys.argv[6])

        breads = BREDS(configuration, seeds_file, negative_seeds, similarity, confidence)

        if sentences_file.endswith('.pkl'):
            print("Loading pre-processed sentences", sentences_file)
            breads.init_bootstrap(tuples=sentences_file)
        else:
            breads.generate_tuples(sentences_file)
            breads.init_bootstrap(tuples=None)

if __name__ == "__main__":
    main()
