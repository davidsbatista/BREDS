import operator
import os
import pickle
import sys
from collections import defaultdict


from gensim import matutils
from nltk.data import load
from numpy import dot
from tqdm import tqdm

from config import Config
from pattern import Pattern
from seed import Seed
from sentence import Sentence
from tuple import Tuple

__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"

# useful for debugging
PRINT_TUPLES = False
PRINT_PATTERNS = False


def blocks(files, size=65536):
    """
    Read the file block-wise and then count the '\n' characters in each block.
    """
    while True:
        buffer = files.read(size)
        if not buffer:
            break
        yield buffer


class BREDS:
    """
    BREDS is a system that extracts relationships between named entities from text.
    """

    def __init__(self, config_file, seeds_file, negative_seeds, similarity, confidence):
        # pylint: disable=too-many-arguments
        self.curr_iteration = 0
        self.patterns = []
        self.processed_tuples = []
        self.candidate_tuples = defaultdict(list)
        self.config = Config(config_file, seeds_file, negative_seeds, similarity, confidence)

    def generate_tuples(self, sentences_file: str):
        """
        Generate tuples instances from a text file with sentences where named entities are already tagged.

        :param sentences_file: input sentences, one per line, with named-entities tagged
        """
        if os.path.exists("processed_tuples.pkl"):
            with open("processed_tuples.pkl", "rb") as f_in:
                print("\nLoading processed tuples from disk...")
                self.processed_tuples = pickle.load(f_in)
            print(len(self.processed_tuples), "tuples loaded")

        else:
            # load needed stuff, word2vec model and a pos-tagger
            self.config.read_word2vec()
            tagger = load("taggers/maxent_treebank_pos_tagger/english.pickle")

            with open(sentences_file, "r", encoding="utf8") as f_in:
                total = sum(bl.count("\n") for bl in blocks(f_in))

            print("\nGenerating relationship instances from sentences")
            with open(sentences_file, encoding="utf-8") as f_sentences:
                for line in tqdm(f_sentences, total=total):
                    sentence = Sentence(
                        line.strip(),
                        self.config.e1_type,
                        self.config.e2_type,
                        self.config.max_tokens_away,
                        self.config.min_tokens_away,
                        self.config.context_window_size,
                        tagger,
                    )

                    for rel in sentence.relationships:
                        tpl = Tuple(rel.ent1, rel.ent2, rel.sentence, rel.before, rel.between, rel.after, self.config)
                        self.processed_tuples.append(tpl)

                print("\n", len(self.processed_tuples), "tuples generated")

            print("Writing generated tuples to disk")
            with open("processed_tuples.pkl", "wb") as f_out:
                pickle.dump(self.processed_tuples, f_out)

    def similarity_3_contexts(self, pattern, tuple):
        """
        Calculates the cosine similarity between the context vectors of a pattern and a tuple.
        """
        (bef, bet, aft) = (0, 0, 0)

        if tuple.bef_vector is not None and pattern.bef_vector is not None:
            bef = dot(matutils.unitvec(tuple.bef_vector), matutils.unitvec(pattern.bef_vector))

        if tuple.bet_vector is not None and pattern.bet_vector is not None:
            bet = dot(matutils.unitvec(tuple.bet_vector), matutils.unitvec(pattern.bet_vector))

        if tuple.aft_vector is not None and pattern.aft_vector is not None:
            aft = dot(matutils.unitvec(tuple.aft_vector), matutils.unitvec(pattern.aft_vector))

        return self.config.alpha * bef + self.config.beta * bet + self.config.gamma * aft

    def similarity_all(self, tpl, extraction_pattern):
        """
        Calculates the cosine similarity between all patterns part of a cluster (i.e., extraction pattern) and the
        vector of a ReVerb pattern extracted from a sentence;

        Returns the max similarity score
        """
        good = 0
        bad = 0
        max_similarity = 0

        for pattern in list(extraction_pattern.tuples):
            score = self.similarity_3_contexts(tpl, pattern)
            if score > max_similarity:
                max_similarity = score
            if score >= self.config.threshold_similarity:
                good += 1
            else:
                bad += 1

        if good >= bad:
            return True, max_similarity
        return False, 0.0

    def match_seeds_tuples(self):
        """
        Checks if the extracted tuples match the seeds tuples.
        """
        matched_tuples = []
        count_matches = {}
        for tuple in self.processed_tuples:
            for sent in self.config.positive_seed_tuples:
                if tuple.ent1 == sent.ent1 and tuple.ent2 == sent.ent2:
                    matched_tuples.append(tuple)
                    try:
                        count_matches[(tuple.ent1, tuple.ent2)] += 1
                    except KeyError:
                        count_matches[(tuple.ent1, tuple.ent2)] = 1

        return count_matches, matched_tuples

    def write_relationships_to_disk(self):
        """
        Writes the extracted relationships to disk.
        """
        print("\nWriting extracted relationships to disk")
        f_output = open("relationships.txt", "w")
        tmp = sorted(list(self.candidate_tuples.keys()), reverse=True)
        for t in tmp:
            f_output.write("instance: " + t.ent1 + "\t" + t.ent2 + "\tscore:" + str(t.confidence) + "\n")
            f_output.write("sentence: " + t.sentence + "\n")
            f_output.write("pattern_bef: " + t.bef_words + "\n")
            f_output.write("pattern_bet: " + t.bet_words + "\n")
            f_output.write("pattern_aft: " + t.aft_words + "\n")
            if t.passive_voice is False:
                f_output.write("passive voice: False\n")
            elif t.passive_voice is True:
                f_output.write("passive voice: True\n")
            f_output.write("\n")
        f_output.close()

    def init_bootstrap(self, tuples):  # noqa: C901
        """Initializes the bootstrap process"""
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
            for sent in self.config.positive_seed_tuples:
                print(sent.ent1, "\t", sent.ent2)

            # Looks for sentences matching the seed instances
            count_matches, matched_tuples = self.match_seeds_tuples()

            if len(matched_tuples) == 0:
                print("\nNo seed matches found")
                sys.exit(0)

            else:
                print("\nNumber of seed matches found")
                sorted_counts = sorted(list(count_matches.items()), key=operator.itemgetter(1), reverse=True)
                for t in sorted_counts:
                    print(t[0][0], "\t", t[0][1], t[1])

                print("\n", len(matched_tuples), "tuples matched")

                # Cluster the matched instances, to generate patterns/update patterns
                print("\nClustering matched instances to generate patterns")
                self.cluster_tuples(matched_tuples)

                # Eliminate patterns supported by less than 'min_pattern_support' tuples
                new_patterns = [p for p in self.patterns if len(p.tuples) > self.config.min_pattern_support]
                self.patterns = new_patterns

                print("\n", len(self.patterns), "patterns generated")

                if PRINT_PATTERNS is True:
                    count = 1
                    print("\nPatterns:")
                    for pattern in self.patterns:
                        print(count)
                        for tuple in pattern.tuples:
                            print("BEF", tuple.bef_words)
                            print("BET", tuple.bet_words)
                            print("AFT", tuple.aft_words)
                            print("========")
                            print("\n")
                        count += 1

                if self.curr_iteration == 0 and len(self.patterns) == 0:
                    print("No patterns generated")
                    sys.exit(0)

                # Look for sentences with occurrence of seeds
                # semantic types (e.g., ORG - LOC)
                # This was already collect and it's stored in:
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
                for t in tqdm(self.processed_tuples):
                    sim_best = 0
                    for extraction_pattern in self.patterns:
                        accept, score = self.similarity_all(t, extraction_pattern)
                        if accept is True:
                            extraction_pattern.update_selectivity(t, self.config)
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
                                self.candidate_tuples[t].append((pattern_best, sim_best))

                        # If this tuple was not extracted before
                        # associate this pattern with the instance
                        # and the similarity score
                        else:
                            self.candidate_tuples[t].append((pattern_best, sim_best))

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
                    tuples_sorted = sorted(extracted_tuples, key=lambda tpl: tpl.confidence, reverse=True)
                    for t in tuples_sorted:
                        print(t.sentence)
                        print(t.ent1, t.ent2)
                        print(t.confidence)
                        print("\n")

                print("Adding tuples to seed with confidence >= {}".format(str(self.config.instance_confidence)))
                for t in list(self.candidate_tuples.keys()):
                    if t.confidence >= self.config.instance_confidence:
                        seed = Seed(t.ent1, t.ent2)
                        self.config.positive_seed_tuples.add(seed)

                # increment the number of iterations
                self.curr_iteration += 1

        self.write_relationships_to_disk()

    def cluster_tuples(self, matched_tuples):
        """
        Single Pass Clustering Algorithm
        Cluster the matched tuples to generate patterns
        """
        # Initialize: if no patterns exist, first tuple goes to first cluster
        if len(self.patterns) == 0:
            c1 = Pattern(matched_tuples[0])
            self.patterns.append(c1)

        count = 0
        for tuple in matched_tuples:
            count += 1
            if count % 1000 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
            max_similarity = 0
            max_similarity_cluster_index = 0

            # go through all patterns(clusters of tuples) and find the one with the highest similarity score
            for i in range(0, len(self.patterns), 1):
                extraction_pattern = self.patterns[i]
                accept, score = self.similarity_all(tuple, extraction_pattern)
                if accept is True and score > max_similarity:
                    max_similarity = score
                    max_similarity_cluster_index = i

            # if max_similarity < min_degree_match create a new cluster having this tuple as the centroid
            if max_similarity < self.config.threshold_similarity:
                c = Pattern(tuple)
                self.patterns.append(c)

            # if max_similarity >= min_degree_match add to the cluster with the highest similarity
            else:
                self.patterns[max_similarity_cluster_index].add_tuple(tuple)


def main():
    if len(sys.argv) != 7:
        print("\nBREDS.py parameters sentences positive_seeds negative_seeds similarity confidence\n")
        sys.exit(0)
    else:
        configuration = sys.argv[1]
        sentences_file = sys.argv[2]
        seeds_file = sys.argv[3]
        negative_seeds = sys.argv[4]
        similarity = float(sys.argv[5])
        confidence = float(sys.argv[6])

        breads = BREDS(configuration, seeds_file, negative_seeds, similarity, confidence)

        if sentences_file.endswith(".pkl"):
            print("Loading pre-processed sentences", sentences_file)
            breads.init_bootstrap(tuples=sentences_file)
        else:
            breads.generate_tuples(sentences_file)
            breads.init_bootstrap(tuples=None)


if __name__ == "__main__":
    main()
