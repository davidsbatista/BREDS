import codecs
import math
import multiprocessing
import operator
import pickle
import queue
import sys
from collections import defaultdict

from gensim import matutils
from nltk.data import load
from numpy import dot, asarray

from breds.config import Config
from breds.pattern import Pattern
from breds.seed import Seed
from breds.sentence import Sentence
from breds.tuple import Tuple

__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"

# useful stuff for debugging
PRINT_TUPLES = False
PRINT_PATTERNS = False


class BREDS(object):
    def __init__(self, config_file, seeds_file, negative_seeds, similarity, confidence, num_cores):
        if num_cores == 0:
            self.num_cpus = multiprocessing.cpu_count()
        else:
            self.num_cpus = num_cores
        self.processed_tuples = list()
        self.candidate_tuples = defaultdict(list)
        self.curr_iteration = 0
        self.patterns = list()
        self.patterns_index = dict()
        self.config = Config(config_file, seeds_file, negative_seeds, similarity, confidence)

    def generate_tuples(self, sentences_file):
        # generate tuples instances from a text file with sentences
        # where named entities are already tagged

        # load word2vec model
        self.config.read_word2vec()

        # copy all sentences from input file into a Queue
        # shared by all processes
        manager = multiprocessing.Manager()
        queue = manager.Queue()

        print("\nLoading sentences from file")
        f_sentences = codecs.open(sentences_file, encoding="utf-8")
        count = 0
        for line in f_sentences:
            if line.startswith("#"):
                continue
            count += 1
            if count % 10000 == 0:
                sys.stdout.write(".")
            queue.put(line.strip())
        f_sentences.close()

        pipes = [multiprocessing.Pipe(False) for _ in range(self.num_cpus)]
        processes = [
            multiprocessing.Process(target=self.generate_instances, args=(queue, pipes[i][1]))
            for i in range(self.num_cpus)
        ]

        print("\nGenerating relationship instances from sentences")
        print("Running", len(processes), " processes")
        for proc in processes:
            proc.start()

        for i in range(len(pipes)):
            data = pipes[i][0].recv()
            child_instances = data[1]
            for x in child_instances:
                self.processed_tuples.append(x)

        for proc in processes:
            proc.join()

        print("\n", len(self.processed_tuples), "instances generated")
        print("Writing generated tuples to disk")
        f = open("processed_tuples.pkl", "wb")
        pickle.dump(self.processed_tuples, f)
        f.close()

    def generate_instances(self, sentences, child_conn):
        # Each process has its own NLTK PoS-tagger
        tagger = load("taggers/maxent_treebank_pos_tagger/english.pickle")
        instances = list()
        while True:
            try:
                s = sentences.get_nowait()
                if sentences.qsize() % 500 == 0:
                    print(multiprocessing.current_process(), "Instances to process", sentences.qsize())

                sentence = Sentence(
                    s,
                    self.config.e1_type,
                    self.config.e2_type,
                    self.config.max_tokens_away,
                    self.config.min_tokens_away,
                    self.config.context_window_size,
                    tagger,
                    self.config,
                )

                for rel in sentence.relationships:
                    t = Tuple(rel.e1, rel.e2, rel.sentence, rel.before, rel.between, rel.after, self.config)
                    instances.append(t)

            except queue.Empty:
                print(multiprocessing.current_process(), "Queue is Empty")
                pid = multiprocessing.current_process().pid
                child_conn.send((pid, instances))
                break

    def similarity_3_contexts(self, t, p):
        (bef, bet, aft) = (0, 0, 0)

        if t.bef_vector is not None and p.bef_vector is not None:
            bef = dot(matutils.unitvec(t.bef_vector), matutils.unitvec(p.bef_vector))

        if t.bet_vector is not None and p.bet_vector is not None:
            bet = dot(matutils.unitvec(t.bet_vector), matutils.unitvec(p.bet_vector))

        if t.aft_vector is not None and p.aft_vector is not None:
            aft = dot(matutils.unitvec(t.aft_vector), matutils.unitvec(p.aft_vector))

        return self.config.alpha * bef + self.config.beta * bet + self.config.gamma * aft

    def similarity_all(self, t, extraction_pattern):
        # calculates the cosine similarity between all patterns part of a
        # cluster (i.e., extraction pattern) and the vector of a ReVerb pattern
        # extracted from a sentence;
        #
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

    def cluster_tuples(self, matched_tuples):
        # single-Pass clustering

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

            # if max_similarity < min_degree_match create a new cluster
            # having this tuple as the centroid
            if max_similarity < self.config.threshold_similarity:
                c = Pattern(t)
                self.patterns.append(c)

            # if max_similarity >= min_degree_match add to the cluster with
            # the highest similarity
            else:
                self.patterns[max_similarity_cluster_index].add_tuple(t)

    def write_relationships_to_disk(self):
        print("\nWriting extracted relationships to disk")
        f_output = open("relationships.txt", "w")
        tmp = sorted(list(self.candidate_tuples.keys()), reverse=True)
        for t in tmp:
            f_output.write("instance: " + t.e1 + "\t" + t.e2 + "\tscore:" + str(t.confidence) + "\n")
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
        # starts a bootstrap iteration

        if tuples is not None:
            f = open(tuples, "r")
            print("Loading pre-processed sentences", tuples)
            self.processed_tuples = pickle.load(f)
            f.close()
            print(len(self.processed_tuples), "tuples loaded")

        self.curr_iteration = 0
        while self.curr_iteration <= self.config.number_iterations:
            print("==========================================")
            print("\nStarting iteration", self.curr_iteration)
            print("\nLooking for seed matches of:")
            for s in self.config.positive_seed_tuples:
                print(s.e1, "\t", s.e2)

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

                # Cluster the matched instances: generate patterns
                print("\nClustering matched instances to generate patterns")
                if len(self.patterns) == 0:
                    self.cluster_tuples(matched_tuples)

                    # Eliminate patterns supported by less than
                    # 'min_pattern_support' tuples
                    new_patterns = [p for p in self.patterns if len(p.tuples) > self.config.min_pattern_support]
                    self.patterns = new_patterns

                else:
                    # Parallelize single-pass clustering
                    # Each tuple must be compared with each extraction pattern

                    # Map:
                    # - Divide the tuples into smaller lists,
                    # accordingly to the number of CPUs
                    # - Pass to each CPU a sub-list of tuples and all the
                    # patterns, comparison is done by each CPU

                    # Merge:
                    # - Each CPU sends to the father process the updated
                    # patterns and new patterns
                    # - Merge patterns based on a pattern_id
                    # - Cluster new created patterns with single-pass clustering

                    # make a copy of the extraction patterns to be
                    # passed to each CPU
                    patterns = [list(self.patterns) for _ in range(self.num_cpus)]

                    # distribute tuples per different CPUs
                    chunks = [list() for _ in range(self.num_cpus)]
                    n_tuples_per_child = int(math.ceil(float(len(matched_tuples)) / self.num_cpus))

                    print("\n#CPUS", self.num_cpus, "\t", "Tuples per CPU", n_tuples_per_child)

                    chunk_n = 0
                    chunck_begin = 0
                    chunck_end = n_tuples_per_child

                    while chunk_n < self.num_cpus:
                        chunks[chunk_n] = matched_tuples[chunck_begin:chunck_end]
                        chunck_begin = chunck_end
                        chunck_end += n_tuples_per_child
                        chunk_n += 1

                    count = 0
                    for c in chunks:
                        print("CPU_" + str(count), "  ", len(c), "patterns")
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

                    # Receive and merge all patterns by 'pattern_id'
                    # new created patterns (new pattern_id) go into
                    # 'child_patterns' and then are merged
                    # by single-pass clustering between patterns

                    child_patterns = list()

                    for i in range(len(pipes)):
                        data = pipes[i][0].recv()
                        patterns = data[1]
                        for p_updated in patterns:
                            pattern_exists = False
                            for p_original in self.patterns:
                                if p_original.id == p_updated.id:
                                    p_original.tuples.update(p_updated.tuples)
                                    pattern_exists = True
                                    break

                            if pattern_exists is False:
                                child_patterns.append(p_updated)

                    for proc in processes:
                        proc.join()

                    print("\nSELF Patterns:")
                    for p in self.patterns:
                        p.merge_all_tuples_bet()
                        print("\n" + str(p.id))
                        if self.config.alpha == 0 and self.config.gamma == 0:
                            for bet_words in p.bet_uniques_words:
                                print("BET", bet_words.encode("utf8"))

                    print("\nChild Patterns:")
                    for p in child_patterns:
                        p.merge_all_tuples_bet()
                        print("\n" + str(p.id))
                        if self.config.alpha == 0 and self.config.gamma == 0:
                            for bet_words in p.bet_uniques_words:
                                print("BET", bet_words.encode("utf8"))

                    print(len(child_patterns), "new created patterns")

                    # merge/aggregate similar patterns generated by
                    # the child processes

                    # start comparing smaller ones with greater ones
                    child_patterns.sort(key=lambda y: len(y.tuples), reverse=False)
                    count = 0
                    new_list = list(self.patterns)
                    for p1 in child_patterns:
                        print("\nNew Patterns", len(child_patterns), "Processed", count)
                        print("New List", len(new_list))
                        print("Pattern:", p1.id, "Tuples:", len(p1.tuples))
                        max_similarity = 0
                        max_similarity_cluster = None
                        for p2 in new_list:
                            if p1 == p2:
                                continue
                            score = self.similarity_cluster(p1, p2)
                            if score > max_similarity:
                                max_similarity = score
                                max_similarity_cluster = p2
                        if max_similarity >= self.config.threshold_similarity:
                            for t in p1.tuples:
                                max_similarity_cluster.tuples.add(t)
                        else:
                            new_list.append(p1)
                        count += 1

                    # add merged patterns to main patterns structure
                    for p in new_list:
                        if p not in self.patterns:
                            self.patterns.append(p)

                if self.curr_iteration == 0 and len(self.patterns) == 0:
                    print("No patterns generated")
                    sys.exit(0)

                print("\n", len(self.patterns), "patterns generated")

                # merge equal tuples inside patterns to make
                # less comparisons in collecting instances
                for p in self.patterns:
                    # if only the BET context is being used,
                    # merge only based on BET contexts
                    if self.config.alpha == 0 and self.config.gamma == 0:
                        p.merge_all_tuples_bet()

                if PRINT_PATTERNS is True:
                    print("\nPatterns:")
                    for p in self.patterns:
                        print("\n" + str(p.id))
                        if self.config.alpha == 0 and self.config.gamma == 0:
                            for bet_words in p.bet_uniques_words:
                                print("BET", bet_words)
                        else:
                            for t in p.tuples:
                                print("BEF", t.bef_words)
                                print("BET", t.bet_words)
                                print("AFT", t.aft_words)
                                print("========")

                # Look for sentences with occurrence of
                # seeds semantic types (e.g., ORG - LOC)

                # This was already collect and its stored in
                # self.processed_tuples
                #
                # Measure the similarity of each occurrence with
                # each extraction pattern and store each pattern that has a
                # similarity higher than a given threshold
                #
                # Each candidate tuple will then have a number of patterns
                # that extracted it each with an associated degree of match.
                print("\nNumber of tuples to be analyzed:", len(self.processed_tuples))

                print("\nCollecting instances based on", len(self.patterns), "extraction patterns")

                # create copies of generated extraction patterns
                # to be passed to each process
                patterns = [list(self.patterns) for _ in range(self.num_cpus)]

                # copy all tuples into a Queue shared by all processes
                manager = multiprocessing.Manager()
                queue = manager.Queue()
                for t in self.processed_tuples:
                    queue.put(t)

                # each distinct process receives as arguments:
                #   - a list, copy of all the original extraction patterns
                #   - a Queue of the tuples
                #   - a pipe to return the collected tuples and updated
                #     patterns to the parent process

                pipes = [multiprocessing.Pipe(False) for _ in range(self.num_cpus)]
                processes = [
                    multiprocessing.Process(target=self.find_instances, args=(patterns[i], queue, pipes[i][1]))
                    for i in range(self.num_cpus)
                ]

                print("Running", len(processes), " processes")
                for proc in processes:
                    proc.start()

                # structures to store each process altered patterns
                # and collected tuples
                patterns_updated = list()
                collected_tuples = list()

                for i in range(len(pipes)):
                    data = pipes[i][0].recv()
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
                        if p_original.id == p_updated.id:
                            p_original.positive += p_updated.positive
                            p_original.negative += p_updated.negative
                            p_original.unknown += p_updated.unknown

                # Index the patterns in an hashtable for later use
                for p in self.patterns:
                    self.patterns_index[p.id] = p

                # update all patterns confidence
                for p in self.patterns:
                    p.update_confidence(self.config)

                if PRINT_PATTERNS is True:
                    print("\nPatterns:")
                    for p in self.patterns:
                        print(p.id)
                        print("Positive", p.positive)
                        print("Negative", p.negative)
                        print("Pattern Confidence", p.confidence)
                        print("\n")

                # Candidate tuples aggregation happens here:
                print("Collecting generated candidate tuples")
                for e in collected_tuples:
                    t = e[0]
                    pattern_best = e[1]
                    sim_best = e[2]

                    # if this tuple was already extracted, check if this
                    # extraction pattern is already associated with it, if not,
                    # associate this pattern with it and similarity score
                    if t in self.candidate_tuples:
                        t_patterns = self.candidate_tuples[t]
                        if t_patterns is not None:
                            if pattern_best not in [x[0] for x in t_patterns]:
                                self.candidate_tuples[t].append((self.patterns_index[pattern_best.id], sim_best))

                    # if this tuple was not extracted before, associate this
                    # pattern with the instance and the similarity score
                    else:
                        self.candidate_tuples[t].append((self.patterns_index[pattern_best.id], sim_best))

                # update tuple confidence based on patterns confidence
                print("\n\nCalculating tuples confidence")
                for t in list(self.candidate_tuples.keys()):
                    confidence = 1
                    t.confidence_old = t.confidence
                    for p in self.candidate_tuples.get(t):
                        confidence *= 1 - (p[0].confidence * p[1])
                    t.confidence = 1 - confidence

                    if self.curr_iteration > 0:
                        t.confidence = t.confidence * self.config.wUpdt + t.confidence_old * (1 - self.config.wUpdt)

                # sort tuples by confidence and print
                if PRINT_TUPLES is True:
                    extracted_tuples = list(self.candidate_tuples.keys())
                    tuples_sorted = sorted(extracted_tuples, key=lambda tl: tl.confidence, reverse=True)
                    for t in tuples_sorted:
                        print(t.sentence)
                        print(t.e1, t.e2)
                        print(t.confidence)
                        print("\n")

                # update seed set of tuples to use in next iteration
                # seeds = { T | conf(T) > instance_confidence }
                print("Adding tuples to seed with confidence >=" + str(self.config.instance_confidence))
                for t in list(self.candidate_tuples.keys()):
                    if t.confidence >= self.config.instance_confidence:
                        seed = Seed(t.e1, t.e2)
                        self.config.positive_seed_tuples.add(seed)

                # increment the number of iterations
                self.curr_iteration += 1

        self.write_relationships_to_disk()

    def similarity_cluster(self, p1, p2):
        count = 0
        score = 0
        if self.config.alpha == 0 and self.config.gamma == 0:
            p1.merge_all_tuples_bet()
            p2.merge_all_tuples_bet()
            for v_bet1 in p1.bet_uniques_vectors:
                for v_bet2 in p2.bet_uniques_vectors:
                    if v_bet1 is not None and v_bet2 is not None:
                        score += dot(matutils.unitvec(asarray(v_bet1)), matutils.unitvec(asarray(v_bet2)))
                        count += 1
        else:
            for t1 in p1.tuples:
                for t2 in p2.tuples:
                    score += self.similarity_3_contexts(t1, t2)
                    count += 1

        return float(score) / float(count)

    def find_instances(self, patterns, instances, child_conn):  # noqa: C901
        updated_patterns = list()
        candidate_tuples = list()
        while True:
            try:
                t = instances.get_nowait()
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
                for p in patterns:
                    good = 0
                    bad = 0
                    if self.config.alpha == 0 and self.config.gamma == 0:
                        for p_bet_v in list(p.bet_uniques_vectors):
                            if t.bet_vector is not None and p_bet_v is not None:
                                score = dot(matutils.unitvec(t.bet_vector), matutils.unitvec(asarray(p_bet_v)))
                                if score >= self.config.threshold_similarity:
                                    good += 1
                                else:
                                    bad += 1

                    if good > bad:
                        p.update_selectivity(t, self.config)
                        if score > max_similarity:
                            max_similarity = score
                            pattern_best = p

                # if its above a threshold associated the pattern with it
                if max_similarity >= self.config.threshold_similarity:
                    candidate_tuples.append((t, pattern_best, max_similarity))

            except queue.Empty:
                print(multiprocessing.current_process(), "Queue is Empty")
                for p in patterns:
                    updated_patterns.append(p)
                pid = multiprocessing.current_process().pid
                child_conn.send((pid, updated_patterns, candidate_tuples))
                break

    def cluster_tuples_parallel(self, patterns, matched_tuples, child_conn):
        updated_patterns = list(patterns)
        count = 0
        for t in matched_tuples:
            count += 1
            if count % 500 == 0:
                print(multiprocessing.current_process(), count, "tuples processed")

            # go through all patterns(clusters of tuples) and find the one with
            # the highest similarity score
            max_similarity = 0
            max_similarity_cluster_index = 0
            for i in range(0, len(updated_patterns)):
                extraction_pattern = updated_patterns[i]
                accept, score = self.similarity_all(t, extraction_pattern)
                if accept is True and score > max_similarity:
                    max_similarity = score
                    max_similarity_cluster_index = i

            # if max_similarity < min_degree_match create a new cluster
            if max_similarity < self.config.threshold_similarity:
                c = Pattern(t)
                updated_patterns.append(c)

            # if max_similarity >= min_degree_match add to the cluster with
            # the highest similarity
            else:
                updated_patterns[max_similarity_cluster_index].add_tuple(t)

        # Eliminate clusters with two or less patterns
        new_patterns = [p for p in updated_patterns if len(p.tuples) > 5]
        pid = multiprocessing.current_process().pid
        print(multiprocessing.current_process(), "Patterns: ", len(new_patterns))
        child_conn.send((pid, new_patterns))


def main():
    if len(sys.argv) != 8:
        print(
            "\nBREDS.py paramters.cfg sentences_file positive_seeds "
            "negative_seeds similarity_threshold confidence_threshold "
            "#cpus_to_use\n"
        )
        sys.exit(0)
    else:
        configuration = sys.argv[1]
        sentences_file = sys.argv[2]
        seeds_file = sys.argv[3]
        negative_seeds = sys.argv[4]
        similarity = sys.argv[5]
        confidence = sys.argv[6]
        num_cores = int(sys.argv[7])

        breads = BREDS(configuration, seeds_file, negative_seeds, float(similarity), float(confidence), num_cores)

        if sentences_file.endswith(".pkl"):
            breads.init_bootstrap(tuples=sentences_file)
        else:
            breads.generate_tuples(sentences_file)
            breads.init_bootstrap(tuples=None)


if __name__ == "__main__":
    main()
