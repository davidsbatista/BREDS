__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import math
import multiprocessing
import operator
import pickle
import queue
import sys
from collections import defaultdict
from typing import Optional, List, Tuple, Dict

from gensim import matutils
from nltk.data import load
from numpy import dot, asarray
from tqdm import tqdm

from breds.breds_tuple import BREDSTuple
from breds.commons import blocks
from breds.config import Config
from breds.pattern import Pattern
from breds.seed import Seed
from breds.sentence import Sentence

PRINT_TUPLES = False
PRINT_PATTERNS = False


class BREDSParallel:
    def __init__(
        self,
        config_file: str,
        seeds_file: str,
        negative_seeds: str,
        similarity: float,
        confidence: float,
        num_cores: int = 0,
    ) -> None:
        # pylint: disable=too-many-arguments
        if num_cores == 0:
            self.num_cpus = multiprocessing.cpu_count()
        else:
            self.num_cpus = num_cores
        self.processed_tuples = []
        self.candidate_tuples = defaultdict(list)
        self.curr_iteration: int = 0
        self.patterns = []
        self.patterns_index = {}
        self.config = Config(config_file, seeds_file, negative_seeds, similarity, confidence)

    def generate_tuples(self, sentences_file: str) -> None:
        # pylint: disable=too-many-locals
        """
        Generate tuples from a text file with sentences where named entities are already tagged
        """

        # load word2vec model
        self.config.word2vec = self.config.read_word2vec(self.config.word2vec_model_path)

        # copy all sentences from input file into a Queue shared by all processes
        manager = multiprocessing.Manager()
        jobs_queue = manager.Queue()

        print("\nLoading sentences from file")
        with open(sentences_file, "r", encoding="utf8") as f_in:
            total = sum(bl.count("\n") for bl in blocks(f_in))
        with open(sentences_file, "r", encoding="utf8") as f_sentences:
            for line in tqdm(f_sentences, total=total):
                if line.startswith("#"):
                    continue
                jobs_queue.put(line.strip())

        pipes = [multiprocessing.Pipe(False) for _ in range(self.num_cpus)]
        processes = [
            multiprocessing.Process(target=self.generate_instances, args=(jobs_queue, pipes[i][1]))
            for i in range(self.num_cpus)
        ]

        print("\nGenerating relationship instances from sentences")
        print("Running", len(processes), " processes")
        for proc in processes:
            proc.start()

        for pipe in pipes:
            data = pipe[0].recv()
            child_instances = data[1]
            for child in child_instances:
                self.processed_tuples.append(child)

        for proc in processes:
            proc.join()

        print(f"\n{len(self.processed_tuples)} instances generated")
        print("Writing generated tuples to disk")
        with open("processed_tuples.pkl", "wb") as f_out:
            pickle.dump(self.processed_tuples, f_out)

    def generate_instances(self, sentences, child_conn):
        """
        Generates instances from sentences
        Each process has its own NLTK PoS-tagger
        """
        tagger = load("taggers/maxent_treebank_pos_tagger/english.pickle")
        instances = []
        while True:
            try:
                text = sentences.get_nowait()
                if sentences.qsize() % 500 == 0:
                    print(multiprocessing.current_process(), "Instances to process", sentences.qsize())

                sentence = Sentence(
                    text,
                    self.config.e1_type,
                    self.config.e2_type,
                    self.config.max_tokens_away,
                    self.config.min_tokens_away,
                    self.config.context_window_size,
                    tagger,
                )
                for rel in sentence.relationships:
                    tpl = BREDSTuple(rel.ent1, rel.ent2, rel.sentence, rel.before, rel.between, rel.after, self.config)
                    instances.append(tpl)

            except queue.Empty:
                print(multiprocessing.current_process(), "Queue is Empty")
                pid = multiprocessing.current_process().pid
                child_conn.send((pid, instances))
                break

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

        Returns the max similarity scores
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
        Checks if an extracted tuple matches seeds tuples
        """

        matched_tuples: List[BREDSTuple] = []
        count_matches: Dict[Tuple[str, str], int] = defaultdict(int)
        for tpl in self.processed_tuples:
            for sentence in self.config.positive_seed_tuples:
                if tpl.ent1 == sentence.ent1 and tpl.ent2 == sentence.ent2:
                    matched_tuples.append(tpl)
                    count_matches[(tpl.ent1, tpl.ent2)] += 1

        return count_matches, matched_tuples

    def cluster_tuples(self, matched_tuples):
        """
        Clusters tuples based on their similarity score w/ a single-pass clustering
        """

        # Initialize: if no patterns exist, first tuple goes to first cluster
        if len(self.patterns) == 0:
            self.patterns.append(Pattern(matched_tuples[0]))

        count = 0
        for matched_tpl in matched_tuples:
            count += 1
            if count % 1000 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
            max_similarity = 0
            max_similarity_cluster_index = 0

            # go through all patterns(clusters of tuples) and find the one with the highest similarity score
            for i in range(0, len(self.patterns), 1):
                extraction_pattern = self.patterns[i]
                accept, score = self.similarity_all(matched_tpl, extraction_pattern)
                if accept is True and score > max_similarity:
                    max_similarity = score
                    max_similarity_cluster_index = i

            # if max_similarity < min_degree_match create a new cluster having this tuple as the centroid
            if max_similarity < self.config.threshold_similarity:
                self.patterns.append(Pattern(matched_tpl))

            # if max_similarity >= min_degree_match add to the cluster with the highest similarity
            else:
                self.patterns[max_similarity_cluster_index].add_tuple(matched_tpl)

    def write_relationships_to_disk(self) -> None:
        """
        Writes extracted relationships to disk
        """

        print("\nWriting extracted relationships to disk")
        with open("relationships.txt", "w", encoding="utf8") as f_out:
            for tpl in sorted(list(self.candidate_tuples.keys()), reverse=True):
                f_out.write("instance: " + tpl.ent1 + "\t" + tpl.ent2 + "\tscore:" + str(tpl.confidence) + "\n")
                f_out.write("sentence: " + tpl.sentence + "\n")
                f_out.write("pattern_bef: " + tpl.bef_words + "\n")
                f_out.write("pattern_bet: " + tpl.bet_words + "\n")
                f_out.write("pattern_aft: " + tpl.aft_words + "\n")
                if tpl.passive_voice is False:
                    f_out.write("passive voice: False\n")
                elif tpl.passive_voice is True:
                    f_out.write("passive voice: True\n")
                f_out.write("\n")

    def debug_patterns_1(self) -> None:
        """
        Print patterns after aggregation
        """
        if PRINT_PATTERNS is True:
            print("\nPatterns:")
            for pattern in self.patterns:
                print("\n" + str(pattern.id))
                if self.config.alpha == 0 and self.config.gamma == 0:
                    for bet_words in pattern.bet_uniques_words:
                        print("BET", bet_words)
                else:
                    for tpl in pattern.tuples:
                        print("BEF", tpl.bef_words)
                        print("BET", tpl.bet_words)
                        print("AFT", tpl.aft_words)
                        print("========")

    def debug_patterns_2(self) -> None:
        """
        Print patterns after aggregation
        """
        if PRINT_PATTERNS is True:
            print("\nPatterns:")
            for pattern in self.patterns:
                print(pattern.id)
                print("Positive", pattern.positive)
                print("Negative", pattern.negative)
                print("Pattern Confidence", pattern.confidence)
                print("\n")

    def debug_tuples(self) -> None:
        """
        Print extracted tuples
        """
        if PRINT_TUPLES is True:
            extracted_tuples = list(self.candidate_tuples.keys())
            tuples_sorted = sorted(extracted_tuples, key=lambda tpl: tpl.confidence, reverse=True)
            for tpl in tuples_sorted:
                print(tpl.sentence)
                print(tpl.ent1, tpl.ent2)
                print(tpl.confidence)
                print("\n")

    def similarity_cluster(self, pattern_1, pattern_2):
        """
        Calculates the similarity between two extraction patterns
        """
        count = 0
        score = 0
        if self.config.alpha == 0 and self.config.gamma == 0:
            pattern_1.merge_all_tuples_bet()
            pattern_2.merge_all_tuples_bet()
            for v_bet1 in pattern_1.bet_uniques_vectors:
                for v_bet2 in pattern_2.bet_uniques_vectors:
                    if v_bet1 is not None and v_bet2 is not None:
                        score += dot(matutils.unitvec(asarray(v_bet1)), matutils.unitvec(asarray(v_bet2)))
                        count += 1
        else:
            for tuple_1 in pattern_1.tuples:
                for tuple_2 in pattern_2.tuples:
                    score += self.similarity_3_contexts(tuple_1, tuple_2)
                    count += 1

        return float(score) / float(count)

    def find_instances(self, patterns, instances, child_conn):  # noqa: C901
        # pylint: disable=too-many-nested-blocks,too-many-branches
        """
        Finds instances for a set of extraction patterns
        """
        updated_patterns = []
        candidate_tuples = []
        while True:
            try:
                tpl = instances.get_nowait()
                if instances.qsize() % 500 == 0:
                    sys.stdout.write(
                        str(multiprocessing.current_process())
                        + " Instances to process: "
                        + str(instances.qsize())
                        + "\n"
                    )
                    sys.stdout.flush()

                # measure similarity towards every extraction pattern
                max_similarity = 0
                pattern_best = None
                score = 0
                for pattern in patterns:
                    good = 0
                    bad = 0
                    if self.config.alpha == 0 and self.config.gamma == 0:
                        for p_bet_v in list(pattern.bet_uniques_vectors):
                            if tpl.bet_vector is not None and p_bet_v is not None:
                                score = dot(matutils.unitvec(tpl.bet_vector), matutils.unitvec(asarray(p_bet_v)))
                                if score >= self.config.threshold_similarity:
                                    good += 1
                                else:
                                    bad += 1

                    if good > bad:
                        pattern.update_selectivity(tpl, self.config)
                        if score > max_similarity:
                            max_similarity = score
                            pattern_best = pattern

                # if it's above a threshold associated the pattern with it
                if max_similarity >= self.config.threshold_similarity:
                    candidate_tuples.append((tpl, pattern_best, max_similarity))

            except queue.Empty:
                print(multiprocessing.current_process(), "Queue is Empty")
                for pattern in patterns:
                    updated_patterns.append(pattern)
                pid = multiprocessing.current_process().pid
                child_conn.send((pid, updated_patterns, candidate_tuples))
                break

    def cluster_tuples_parallel(self, patterns, matched_tuples, child_conn):
        """
        Clusters tuples in parallel
        """
        updated_patterns = list(patterns)
        count = 0
        for matched_tpl in matched_tuples:
            count += 1
            if count % 500 == 0:
                print(multiprocessing.current_process(), count, "tuples processed")

            # go through all patterns(clusters of tuples) and find the one with the highest similarity score
            max_similarity = 0
            max_similarity_cluster_index = 0
            for idx, extraction_pattern in enumerate(updated_patterns):
                accept, score = self.similarity_all(matched_tpl, extraction_pattern)
                if accept is True and score > max_similarity:
                    max_similarity = score
                    max_similarity_cluster_index = idx

            # if max_similarity < min_degree_match create a new cluster
            if max_similarity < self.config.threshold_similarity:
                updated_patterns.append(Pattern(matched_tpl))

            # if max_similarity >= min_degree_match add to the cluster with the highest similarity
            else:
                updated_patterns[max_similarity_cluster_index].add_tuple(matched_tpl)

        # eliminate clusters with two or fewer patterns
        new_patterns = [p for p in updated_patterns if len(p.tuples) > 5]
        pid = multiprocessing.current_process().pid
        print(multiprocessing.current_process(), "Patterns: ", len(new_patterns))
        child_conn.send((pid, new_patterns))

    def init_bootstrap(self, processed_tuples: Optional[str] = None) -> None:  # noqa: C901
        # pylint: disable=too-many-branches,too-many-statements, too-many-locals, too-many-nested-blocks
        """
        Initializes the bootstrap algorithm
        """
        if processed_tuples is not None:
            print("Loading pre-processed sentences", processed_tuples)
            with open(processed_tuples, "rb") as f_in:
                self.processed_tuples = pickle.load(f_in)
            print(len(self.processed_tuples), "tuples loaded")

        self.curr_iteration = 0
        while self.curr_iteration <= self.config.number_iterations:
            print("==========================================")
            print("\nStarting iteration", self.curr_iteration)
            print("\nLooking for seed matches of:")
            for seed in self.config.positive_seed_tuples:
                print(seed.ent1, "\t", seed.ent2)

            # Looks for sentences matching the seed instances
            count_matches, matched_tuples = self.match_seeds_tuples()
            if len(matched_tuples) == 0:
                print("\nNo seed matches found")
                sys.exit(0)
            else:
                print("\nNumber of seed matches found")
                for match in sorted(list(count_matches.items()), key=operator.itemgetter(1), reverse=True):
                    print(match[0][0], "\t", match[0][1], match[1])
                print("\n", len(matched_tuples), "tuples matched")

                # Cluster the matched instances: generate patterns
                print("\nClustering matched instances to generate patterns")
                if len(self.patterns) == 0:
                    self.cluster_tuples(matched_tuples)
                    # Eliminate patterns supported by less than 'min_pattern_support' tuples
                    new_patterns = [p for p in self.patterns if len(p.tuples) > self.config.min_pattern_support]
                    self.patterns = new_patterns
                else:
                    # Parallelize single-pass clustering
                    # Each tuple must be compared with each extraction pattern

                    # Map:
                    # - Divide the tuples into smaller lists, accordingly to the number of CPUs
                    # - Pass to each CPU a sub-list of tuples and all the patterns, comparison is done by each CPU

                    # Merge:
                    # - Each CPU sends to the father process the updated patterns and new patterns
                    # - Merge patterns based on a pattern_id
                    # - Cluster new created patterns with single-pass clustering

                    # make a copy of the extraction patterns to be passed to each CPU
                    patterns = [list(self.patterns) for _ in range(self.num_cpus)]

                    # distribute tuples per different CPUs
                    chunks: List[List] = [[] for _ in range(self.num_cpus)]
                    n_tuples_per_child = int(math.ceil(float(len(matched_tuples)) / self.num_cpus))

                    print("\n#CPUS", self.num_cpus, "\t", "Tuples per CPU", n_tuples_per_child)

                    chunk_n = 0
                    chunk_begin = 0
                    chunk_end = n_tuples_per_child

                    while chunk_n < self.num_cpus:
                        chunks[chunk_n] = matched_tuples[chunk_begin:chunk_end]
                        chunk_begin = chunk_end
                        chunk_end += n_tuples_per_child
                        chunk_n += 1

                    count = 0
                    for chunk in chunks:
                        print("CPU_" + str(count), "  ", len(chunk), "patterns")
                        count += 1

                    pipes = [multiprocessing.Pipe(False) for _ in range(self.num_cpus)]
                    processes = [
                        multiprocessing.Process(
                            target=self.cluster_tuples_parallel, args=(patterns[i], chunks[i], pipes[i][1])
                        )
                        for i in range(self.num_cpus)
                    ]

                    print("\nRunning", len(processes), " processes")
                    for proc in processes:
                        proc.start()

                    # Receive and merge all patterns by 'pattern_id' new created patterns (new pattern_id) go into
                    # 'child_patterns' and then are merged by single-pass clustering between patterns
                    child_patterns = []
                    for pipe in pipes:
                        data = pipe[0].recv()
                        patterns = data[1]
                        for p_updated in patterns:
                            pattern_exists = False
                            for p_original in self.patterns:
                                if p_original.uuid == p_updated.uuid:
                                    p_original.tuples.update(p_updated.tuples)
                                    pattern_exists = True
                                    break

                            if pattern_exists is False:
                                child_patterns.append(p_updated)

                    for proc in processes:
                        proc.join()

                    print("\nSELF Patterns:")
                    for pattern in self.patterns:
                        pattern.merge_all_tuples_bet()
                        print("\n" + str(pattern.uuid))
                        if self.config.alpha == 0 and self.config.gamma == 0:
                            for bet_words in pattern.bet_uniques_words:
                                print("BET", bet_words.encode("utf8"))

                    print("\nChild Patterns:")
                    for ch_pattern in child_patterns:
                        ch_pattern.merge_all_tuples_bet()
                        print("\n" + str(ch_pattern.uuid))
                        if self.config.alpha == 0 and self.config.gamma == 0:
                            for bet_words in ch_pattern.bet_uniques_words:
                                print("BET", bet_words.encode("utf8"))

                    print(len(child_patterns), "new created patterns")

                    # merge/aggregate similar patterns generated by the child processes

                    # start comparing smaller ones with greater ones
                    child_patterns.sort(key=lambda y: len(y.tuples), reverse=False)
                    count = 0
                    new_list = list(self.patterns)
                    for pattern_1 in child_patterns:
                        print("\nNew Patterns", len(child_patterns), "Processed", count)
                        print("New List", len(new_list))
                        print("Pattern:", pattern_1.uuid, "Tuples:", len(pattern_1.tuples))
                        max_similarity = 0
                        max_similarity_cluster = None
                        for pattern_2 in new_list:
                            if pattern_1 == pattern_2:
                                continue
                            score = self.similarity_cluster(pattern_1, pattern_2)
                            if score > max_similarity:
                                max_similarity = score
                                max_similarity_cluster = pattern_2
                        if max_similarity >= self.config.threshold_similarity:
                            for pattern_tpl in pattern_1.tuples:
                                max_similarity_cluster.tuples.add(pattern_tpl)
                        else:
                            new_list.append(pattern_1)
                        count += 1

                    # add merged patterns to main patterns structure
                    for new_pattern in new_list:
                        if new_pattern not in self.patterns:
                            self.patterns.append(new_pattern)

                if self.curr_iteration == 0 and len(self.patterns) == 0:
                    print("No patterns generated")
                    sys.exit(0)

                print("\n", len(self.patterns), "patterns generated")

                # merge equal tuples inside patterns to make less comparisons in collecting instances
                for pattern_tpl_merged in self.patterns:
                    # if only the BET context is being used, merge only based on BET contexts
                    if self.config.alpha == 0 and self.config.gamma == 0:
                        pattern_tpl_merged.merge_all_tuples_bet()

                self.debug_patterns_1()

                # Look for sentences with occurrence of seeds semantic types (e.g., ORG - LOC)

                # This was already collect, and it's stored in self.processed_tuples
                #
                # Measure the similarity of each occurrence with each extraction pattern and store each pattern that
                # has a similarity higher than a given threshold
                #
                # Each candidate tuple will then have a number of patterns
                # that extracted it each with an associated degree of match.
                print("\nNumber of tuples to be analyzed:", len(self.processed_tuples))
                print("\nCollecting instances based on", len(self.patterns), "extraction patterns")

                # create copies of generated extraction patterns to be passed to each process
                patterns = [list(self.patterns) for _ in range(self.num_cpus)]

                # copy all tuples into a Queue shared by all processes
                manager = multiprocessing.Manager()
                job_queue = manager.Queue()
                for processed_tpl in self.processed_tuples:
                    job_queue.put(processed_tpl)

                # each distinct process receives as arguments:
                #   - a list, copy of all the original extraction patterns
                #   - a Queue of the tuples
                #   - a pipe to return the collected tuples and updated
                #     patterns to the parent process

                pipes = [multiprocessing.Pipe(False) for _ in range(self.num_cpus)]
                processes = [
                    multiprocessing.Process(target=self.find_instances, args=(patterns[i], job_queue, pipes[i][1]))
                    for i in range(self.num_cpus)
                ]

                print("Running", len(processes), " processes")
                for proc in processes:
                    proc.start()

                # structures to store each process altered patterns
                # and collected tuples
                patterns_updated = []
                collected_tuples = []

                for pipe in pipes:
                    data = pipe[0].recv()
                    child_pid = data[0]
                    patterns = data[1]
                    tuples = data[2]
                    print(child_pid, "patterns", len(patterns), "tuples", len(tuples))
                    patterns_updated.extend(patterns)
                    collected_tuples.extend(tuples)

                for proc in processes:
                    proc.join()

                # Extraction patterns aggregation happens here:
                for p_updated in patterns_updated:
                    for p_original in self.patterns:
                        if p_original.uuid == p_updated.uuid:
                            p_original.positive += p_updated.positive
                            p_original.negative += p_updated.negative
                            p_original.unknown += p_updated.unknown

                # Index the patterns in a hashtable for later use
                for n_pattern in self.patterns:
                    self.patterns_index[n_pattern.uuid] = n_pattern

                # update all patterns confidence
                for u_pattern in self.patterns:
                    u_pattern.update_confidence(self.config)

                self.debug_patterns_2()

                # Candidate tuples aggregation happens here:
                print("Collecting generated candidate tuples")
                for element in collected_tuples:
                    tpl = element[0]
                    pattern_best = element[1]
                    sim_best = element[2]

                    # if this tuple was already extracted, check if this
                    # extraction pattern is already associated with it, if not,
                    # associate this pattern with it and similarity score
                    if tpl in self.candidate_tuples:
                        t_patterns = self.candidate_tuples[tpl]
                        if t_patterns is not None:
                            if pattern_best not in [x[0] for x in t_patterns]:
                                self.candidate_tuples[tpl].append((self.patterns_index[pattern_best.uuid], sim_best))

                    # if this tuple was not extracted before, associate this pattern with the instance and
                    # the similarity score
                    else:
                        self.candidate_tuples[tpl].append((self.patterns_index[pattern_best.uuid], sim_best))

                # update tuple confidence based on patterns confidence
                print("\n\nCalculating tuples confidence")
                for cand_tpl in list(self.candidate_tuples.keys()):
                    confidence = 1
                    cand_tpl.confidence_old = cand_tpl.confidence
                    for cand_tpl_pattern in self.candidate_tuples.get(cand_tpl, None):
                        confidence *= 1 - (cand_tpl_pattern[0].confidence * cand_tpl_pattern[1])
                    cand_tpl.confidence = 1 - confidence

                    if self.curr_iteration > 0:
                        cand_tpl.confidence = cand_tpl.confidence * self.config.w_updt + cand_tpl.confidence_old * (
                            1 - self.config.w_updt
                        )

                # sort tuples by confidence and print
                self.debug_tuples()

                # update seed set of tuples to use in next iteration seeds = { T | conf(T) > instance_confidence }
                print("Adding tuples to seed with confidence >=" + str(self.config.instance_confidence))
                for candidate_tpl in list(self.candidate_tuples.keys()):
                    if candidate_tpl.confidence >= self.config.instance_confidence:
                        seed = Seed(candidate_tpl.ent1, candidate_tpl.ent2)
                        self.config.positive_seed_tuples.add(seed)

                # increment the number of iterations
                self.curr_iteration += 1

        self.write_relationships_to_disk()
