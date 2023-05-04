__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"

import json
import operator
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from gensim import matutils
from nltk.data import load
from numpy import dot
from tqdm import tqdm

from breds.breds_tuple import BREDSTuple
from breds.commons import blocks
from breds.config import Config
from breds.pattern import Pattern
from breds.seed import Seed
from breds.sentence import Sentence

PRINT_TUPLES = False
PRINT_PATTERNS = False


class BREDS:
    """
    BREDS is a system that extracts relationships between named entities from text.
    """

    def __init__(self, config_file: str, seeds_file: str, negative_seeds: str, similarity: float, confidence: float):
        # pylint: disable=too-many-arguments
        self.curr_iteration = 0
        self.patterns: List[Pattern] = []
        self.processed_tuples: List[BREDSTuple] = []
        self.candidate_tuples: Dict[BREDSTuple, List[Tuple[Pattern, float]]] = defaultdict(list)
        self.config = Config(config_file, seeds_file, negative_seeds, similarity, confidence)
        self.config.print_config()

    def generate_tuples(self, sentences_file: str) -> None:
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
            self.config.word2vec = self.config.read_word2vec(self.config.word2vec_model_path)
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
                        tpl = BREDSTuple(
                            rel.ent1, rel.ent2, rel.sentence, rel.before, rel.between, rel.after, self.config
                        )
                        self.processed_tuples.append(tpl)

                print("\n", len(self.processed_tuples), "tuples generated")

            print("Writing generated tuples to disk")
            with open("processed_tuples.pkl", "wb") as f_out:
                pickle.dump(self.processed_tuples, f_out)

    def similarity_3_contexts(self, tpl: BREDSTuple, pattern: BREDSTuple) -> float:
        """
        Calculates the cosine similarity between the context vectors of a pattern and a tuple.
        """
        (bef, bet, aft) = (0, 0, 0)

        if tpl.bef_vector is not None and pattern.bef_vector is not None:
            bef = dot(matutils.unitvec(tpl.bef_vector), matutils.unitvec(pattern.bef_vector))

        if tpl.bet_vector is not None and pattern.bet_vector is not None:
            bet = dot(matutils.unitvec(tpl.bet_vector), matutils.unitvec(pattern.bet_vector))

        if tpl.aft_vector is not None and pattern.aft_vector is not None:
            aft = dot(matutils.unitvec(tpl.aft_vector), matutils.unitvec(pattern.aft_vector))

        return self.config.alpha * bef + self.config.beta * bet + self.config.gamma * aft

    def similarity_all(self, tpl: BREDSTuple, extraction_pattern: Pattern) -> Tuple[bool, float]:
        """
        Calculates the cosine similarity between all patterns part of a cluster (i.e., extraction pattern) and the
        vector of a ReVerb pattern extracted from a sentence.

        Returns the max similarity score
        """
        good: int = 0
        bad: int = 0
        max_similarity: float = 0.0

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

    def match_seeds_tuples(self) -> Tuple[Dict[Tuple[str, str], int], List[BREDSTuple]]:
        """
        Checks if the extracted tuples match the seeds tuples.
        """
        matched_tuples: List[BREDSTuple] = []
        count_matches: Dict[Tuple[str, str], int] = defaultdict(int)
        for tpl in self.processed_tuples:
            for sent in self.config.positive_seed_tuples:
                if tpl.ent1 == sent.ent1 and tpl.ent2 == sent.ent2:
                    matched_tuples.append(tpl)
                    try:
                        count_matches[(tpl.ent1, tpl.ent2)] += 1
                    except KeyError:
                        count_matches[(tpl.ent1, tpl.ent2)] = 1

        return count_matches, matched_tuples

    def write_relationships_to_disk(self) -> None:
        """
        Writes the extracted relationships to disk.
        The output file is a JSONL file with one relationship per line.
        """
        print("\nWriting extracted relationships to disk")
        with open("relationships.jsonl", "wt", encoding="utf8") as f_out:
            for tpl in sorted(list(self.candidate_tuples.keys()), reverse=True):
                f_out.write(json.dumps(tpl.to_json()) + "\n")

    def cluster_tuples(self, matched_tuples: List[BREDSTuple]) -> None:
        """
        Single Pass Clustering Algorithm
        Cluster the matched tuples to generate patterns
        """
        # Initialize: if no patterns exist, first tuple goes to first cluster
        if len(self.patterns) == 0:
            self.patterns.append(Pattern(matched_tuples[0]))

        count = 0
        for tpl in matched_tuples:
            count += 1
            if count % 1000 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
            max_similarity: float = 0.0
            max_similarity_cluster_index = 0

            # go through all patterns(clusters of tuples) and find the one with the highest similarity score
            for i in range(0, len(self.patterns), 1):
                extraction_pattern = self.patterns[i]
                accept, score = self.similarity_all(tpl, extraction_pattern)
                if accept is True and score > max_similarity:
                    max_similarity = score
                    max_similarity_cluster_index = i

            # if max_similarity < min_degree_match create a new cluster having this tuple as the centroid
            if max_similarity < self.config.threshold_similarity:
                cluster = Pattern(tpl)
                self.patterns.append(cluster)

            # if max_similarity >= min_degree_match add to the cluster with the highest similarity
            else:
                self.patterns[max_similarity_cluster_index].add_tuple(tpl)

    def debug_patterns_1(self) -> None:
        """
        Prints the patterns to the console
        """
        count = 1
        print("\nPatterns:")
        for pattern in self.patterns:
            print(count)
            for pattern_tuple in pattern.tuples:
                print("BEF", pattern_tuple.bef_words)
                print("BET", pattern_tuple.bet_words)
                print("AFT", pattern_tuple.aft_words)
                print("========")
                print("\n")
            count += 1

    def debug_patterns_2(self) -> None:
        """
        Prints the patterns to the console
        """
        print("\nPatterns:")
        for pattern in self.patterns:
            for tpl in pattern.tuples:
                print("BEF", tpl.bef_words)
                print("BET", tpl.bet_words)
                print("AFT", tpl.aft_words)
                print("========")
            print("Positive", pattern.positive)
            print("Negative", pattern.negative)
            print("Unknown", pattern.unknown)
            print("Tuples", len(pattern.tuples))
            print("Pattern Confidence", pattern.confidence)
            print("\n")

    def debug_tuples(self) -> None:
        """
        Prints the tuples to the console
        """
        if PRINT_TUPLES is True:
            extracted_tuples = list(self.candidate_tuples.keys())
            tuples_sorted = sorted(extracted_tuples, key=lambda tpl: tpl.confidence, reverse=True)
            for tpl in tuples_sorted:
                print(tpl.sentence)
                print(tpl.ent1, tpl.ent2)
                print(tpl.confidence)
                print("\n")

    def updated_tuple_confidence(self) -> None:
        """
        Updates the confidence of the tuples
        """
        print("\n\nCalculating tuples confidence")
        for tpl, patterns in self.candidate_tuples.items():
            confidence: float = 1.0
            tpl.confidence_old = tpl.confidence
            for pattern in patterns:
                confidence *= 1 - (pattern[0].confidence * pattern[1])
            tpl.confidence = 1 - confidence

    def generate_candidate_tuples(self) -> None:
        """
        Generates the candidate tuples
        """
        for tpl in tqdm(self.processed_tuples):
            sim_best: float = 0.0
            for extraction_pattern in self.patterns:
                accept, score = self.similarity_all(tpl, extraction_pattern)
                if accept is True:
                    extraction_pattern.update_selectivity(tpl, self.config)
                    if score > sim_best:
                        sim_best = score
                        pattern_best = extraction_pattern

            if sim_best >= self.config.threshold_similarity:
                # if this tuple was already extracted, check if this
                # extraction pattern is already associated with it,
                # if not, associate this pattern with it and store the
                # similarity score
                patterns = self.candidate_tuples[tpl]
                if patterns is not None:
                    if pattern_best not in [x[0] for x in patterns]:
                        self.candidate_tuples[tpl].append((pattern_best, sim_best))

                # If this tuple was not extracted before
                # associate this pattern with the instance
                # and the similarity score
                else:
                    self.candidate_tuples[tpl].append((pattern_best, sim_best))

    def init_bootstrap(self, processed_tuples: Optional[str] = None) -> None:  # noqa: C901
        """Initializes the bootstrap process"""
        if processed_tuples is not None:
            print("\nLoading processed tuples from disk...")
            with open(processed_tuples, "rb") as f_in:
                self.processed_tuples = pickle.load(f_in)
            print(len(self.processed_tuples), "tuples loaded")

        self.curr_iteration = 0

        while self.curr_iteration <= self.config.number_iterations:
            print("==========================================")
            print("\nStarting iteration", self.curr_iteration)
            print("\nLooking for seed matches of:")
            for sent in self.config.positive_seed_tuples:
                # ToDo: replace with f-strings
                print(sent.ent1, "\t", sent.ent2)

            # Looks for sentences matching the seed instances
            count_matches, matched_tuples = self.match_seeds_tuples()

            if len(matched_tuples) == 0:
                print("\nNo seed matches found")
                sys.exit(0)

            else:
                print("\nNumber of seed matches found")
                for seed_match in sorted(list(count_matches.items()), key=operator.itemgetter(1), reverse=True):
                    print(f"{seed_match[0][0]}\t{seed_match[0][1]}\t{seed_match[1]}")
                print(f"\n{len(matched_tuples)} tuples matched")

                # Cluster the matched instances, to generate patterns/update patterns
                print("\nClustering matched instances to generate patterns")
                self.cluster_tuples(matched_tuples)
                # Eliminate patterns supported by less than 'min_pattern_support' tuples
                new_patterns = [p for p in self.patterns if len(p.tuples) > self.config.min_pattern_support]
                self.patterns = new_patterns

                print("\n", len(self.patterns), "patterns generated")
                if PRINT_PATTERNS is True:
                    self.debug_patterns_1()

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
                self.generate_candidate_tuples()

                # update all patterns confidence
                for pattern in self.patterns:
                    pattern.update_confidence(self.config)

                if PRINT_PATTERNS is True:
                    self.debug_patterns_2()

                # update tuple confidence based on patterns confidence
                self.updated_tuple_confidence()

                # sort tuples by confidence and print
                self.debug_tuples()

                print(f"Adding tuples to seed with confidence >= {str(self.config.instance_confidence)}")
                for tpl, _ in self.candidate_tuples.items():
                    if tpl.confidence >= self.config.instance_confidence:
                        seed = Seed(tpl.ent1, tpl.ent2)
                        self.config.positive_seed_tuples.add(seed)

                # increment the number of iterations
                self.curr_iteration += 1

        self.write_relationships_to_disk()
