#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jellyfish
import fileinput
import functools
import multiprocessing
import re
import time
import sys
import os
import pickle

from breds.sentence import Sentence
from whoosh.index import open_dir, os
from whoosh.query import spans
from whoosh import query
from nltk import word_tokenize, bigrams
from nltk.corpus import stopwords
from collections import defaultdict

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

# relational words used in calculating the set C and D with the proximity PMI

founded_unigrams = ['founder', 'co-founder', 'cofounder', 'co-founded', 'cofounded', 'founded',
                    'founders']
founded_bigrams = ['started by']

acquired_unigrams = ['owns', 'acquired', 'bought', 'acquisition']
acquired_bigrams = []

headquarters_unigrams = ['headquarters', 'headquartered', 'offices', 'office',
                         'building', 'buildings', 'factory', 'plant', 'compound']
headquarters_bigrams = ['based in', 'located in', 'main office', ' main offices',
                        'offices in', 'building in','office in', 'branch in',
                        'store in', 'firm in', 'factory in', 'plant in',
                        'head office', 'head offices', 'in central',
                        'in downtown', 'outskirts of', 'suburs of']

employment_unigrams = ['chief', 'scientist', 'professor', 'biologist', 'ceo',
                       'CEO', 'employer']
employment_bigrams = []

bad_tokens = [",", "(", ")", ";", "''",  "``", "'s", "-", "vs.", "v", "'", ":", ".", "--"]
stopwords_list = stopwords.words('english')
not_valid = bad_tokens + stopwords_list

# PMI value for proximity
PMI = 0.7

# Parameters for relationship extraction from Sentence
MAX_TOKENS_AWAY = 6
MIN_TOKENS_AWAY = 1
CONTEXT_WINDOW = 2

# DEBUG stuff
PRINT_NOT_FOUND = False

# stores all variations matched with database
manager = multiprocessing.Manager()
all_in_database = manager.dict()


class ExtractedFact(object):
    def __init__(self, _e1, _e2, _score, _bef, _bet, _aft, _sentence,
                 _passive_voice):
        self.ent1 = _e1
        self.ent2 = _e2
        self.score = _score
        self.bef_words = _bef
        self.bet_words = _bet
        self.aft_words = _aft
        self.sentence = _sentence
        self.passive_voice = _passive_voice

    def __cmp__(self, other):
            if other.score > self.score:
                return -1
            elif other.score < self.score:
                return 1
            else:
                return 0

    def __hash__(self):
        sig = hash(self.ent1) ^ hash(self.ent2) ^ hash(self.bef_words) ^ \
              hash(self.bet_words) ^ hash(self.aft_words) ^ \
              hash(self.score) ^ hash(self.sentence)
        return sig

    def __eq__(self, other):
        if self.ent1 == other.ent1 and \
           self.ent2 == other.ent2 and \
           self.score == other.score and \
           self.bef_words == other.bef_words and \
           self.bet_words == other.bet_words and \
           self.aft_words == other.aft_words and \
           self.sentence == other.sentence:
            return True
        else:
            return False


# ###########################################
# Misc., Utils, parsing corpus into memory #
# ###########################################

def timecall(f):
    @functools.wraps(f)
    def wrapper(*args, **kw):
        start = time.time()
        result = f(*args, **kw)
        end = time.time()
        # print "%s %.2f seconds" % (f.__name__, end - start)
        print("Time taken: %.2f seconds" % (end - start))
        return result

    return wrapper


def is_acronym(entity):
    if len(entity.split()) == 1 and entity.isupper():
        return True
    else:
        return False


def process_corpus(queue, g_dash, e1_type, e2_type):
    count = 0
    added = 0
    while True:
        try:
            if count % 25000 == 0:
                print(multiprocessing.current_process(),
                      "In Queue", queue.qsize(), "Total added: ", added)
            line = queue.get_nowait()
            s = Sentence(line.strip(), e1_type, e2_type,
                         MAX_TOKENS_AWAY, MIN_TOKENS_AWAY, CONTEXT_WINDOW)
            for r in s.relationships:
                tokens = word_tokenize(r.between)
                if all(x in not_valid for x in word_tokenize(r.between)):
                    continue
                elif "," in tokens and tokens[0] != ',':
                    continue
                else:
                    g_dash.append(r)
                    added += 1
            count += 1
        except queue.Empty:
            break


def process_output(data, threshold, rel_type):
    """
    parses the file with the relationships extracted by the system
    each relationship is transformed into a ExtracteFact class
    """
    system_output = list()
    for line in fileinput.input(data):
        if line.startswith('instance'):
            instance_parts, score = line.split("score:")
            e1, e2 = instance_parts.split("instance:")[1].strip().split('\t')

        if line.startswith('sentence'):
            sentence = line.split("sentence:")[1].strip()

        if line.startswith('pattern_bef:'):
            bef = line.split("pattern_bef:")[1].strip()

        if line.startswith('pattern_bet:'):
            bet = line.split("pattern_bet:")[1].strip()

        if line.startswith('pattern_aft:'):
            aft = line.split("pattern_aft:")[1].strip()

        if line.startswith('passive voice:'):
            tmp = line.split("passive voice:")[1].strip()
            if tmp == 'False':
                passive_voice = False
            elif tmp == 'True':
                passive_voice = True

        if line.startswith('\n') and float(score) >= threshold:
            if 'bef' not in locals():
                bef = ''
            if 'aft' not in locals():
                aft = ''
            if passive_voice is True and rel_type in ['acquired',
                                                      'headquarters']:
                r = ExtractedFact(e2, e1, float(score), bef, bet, aft,
                                  sentence, passive_voice)
            else:
                r = ExtractedFact(e1, e2, float(score), bef, bet, aft,
                                  sentence, passive_voice)

            if ("'s parent" in bet or 'subsidiary of' in bet or
                bet == 'subsidiary') and rel_type == 'acquired':
                r = ExtractedFact(e2, e1, float(score), bef, bet, aft,
                                  sentence, passive_voice)
            system_output.append(r)

    fileinput.close()
    return system_output


def process_freebase(data, rel_type):
    # Load relationships from Freebase and keep them in the same direction has
    # the output of the extraction system
    """
    # rel_type                   Gold standard directions
    founder_arg2_arg1            PER-ORG
    headquarters_arg1_arg2       ORG-LOC
    acquired_arg1_arg2           ORG-ORG
    contained_by_arg1_arg2       LOC-LOC
    """

    # store a tuple (entity1, entity2) in a dictionary
    database_1 = defaultdict(list)

    # store in a dictionary per relationship: dict['ent1'] = 'ent2'
    database_2 = defaultdict(list)

    # store in a dictionary per relationship: dict['ent2'] = 'ent1'
    database_3 = defaultdict(list)

    # regex used to clean entities
    numbered = re.compile(r'#[0-9]+$')

    # for the 'founder' relationships don't load those from freebase, as it
    # lists countries (i.e., LOC entities) as founders and not persons
    founder_to_ignore = ['UNESCO', 'World Trade Organization', 'European Union',
                         'United Nations']

    for line in fileinput.input(data):
        if line.startswith('#'):
            continue
        try:
            e1, r, e2 = line.split('\t')
        except Exception:
            print(line)
            print(line.split('\t'))
            sys.exit()

        # ignore some entities, which are Freebase identifiers or are ambigious
        if e1.startswith('/') or e2.startswith('/'):
            continue
        if e1.startswith('m/') or e2.startswith('m/'):
            continue
        if re.search(numbered, e1) or re.search(numbered, e2):
            continue
        if e2.strip() in founder_to_ignore:
            continue
        else:
            if "(" in e1:
                e1 = re.sub(r"\(.*\)", "", e1).strip()
            if "(" in e2:
                e2 = re.sub(r"\(.*\)", "", e2).strip()

            if rel_type == 'founder' or rel_type == 'employer':
                database_1[(e2.strip(), e1.strip())].append(r)
                database_2[e2.strip()].append(e1.strip())
                database_3[e1.strip()].append(e2.strip())
            else:
                database_1[(e1.strip(), e2.strip())].append(r)
                database_2[e1.strip()].append(e2.strip())
                database_3[e2.strip()].append(e1.strip())

    return database_1, database_2, database_3


def load_acronyms(data):
    acronyms = defaultdict(list)
    for line in fileinput.input(data):
        parts = line.split('\t')
        acronym = parts[0].strip()
        if "/" in acronym:
            continue
        expanded = parts[-1].strip()
        if "/" in expanded:
            continue
        acronyms[acronym].append(expanded)
    fileinput.close()
    return acronyms


def load_dbpedia(data, database_1, database_2):
    for line in fileinput.input(data):
        e1, rel, e2, p = line.split()
        e1 = e1.split('<http://dbpedia.org/resource/')[1].replace(">", "")
        e2 = e2.split('<http://dbpedia.org/resource/')[1].replace(">", "")
        e1 = re.sub("_", " ", e1)
        e2 = re.sub("_", " ", e2)

        if "(" in e1 or "(" in e2:
            e1 = re.sub("\(.*\)", "", e1)
            e2 = re.sub("\(.*\)", "", e2)

            # store a tuple (entity1, entity2) in a dictionary
            database_1[(e1.strip(), e2.strip())].append(p)

            # store in a dictionary per relationship: dict['ent1'] = 'ent2'
            database_2[e1.strip()].append(e2.strip())

        else:
            e1 = e1.decode("utf8").strip()
            e2 = e2.decode("utf8").strip()
            # store a tuple (entity1, entity2) in a dictionary
            database_1[(e1, e2)].append(p)

            # store in a dictionary per relationship: dict['ent1'] = 'ent2'
            database_2[e1.strip()].append(e2.strip())

    fileinput.close()

    return database_1, database_2


def extract_bigrams(text):
    tokens = word_tokenize(text)
    return [gram[0]+' '+gram[1] for gram in bigrams(tokens)]


# ########################################
# Estimations of sets and intersections #
# ########################################
@timecall
def calculate_a(not_in_database, e1_type, e2_type, index, rel_words_unigrams,
                rel_words_bigrams):
    m = multiprocessing.Manager()
    queue = m.Queue()
    num_cpus = multiprocessing.cpu_count()
    results = [m.list() for _ in range(num_cpus)]
    not_found = [m.list() for _ in range(num_cpus)]

    for r in not_in_database:
        queue.put(r)

    processes = [multiprocessing.Process(
        target=proximity_pmi_a,
        args=(e1_type, e2_type, queue, index, results[i], not_found[i],
              rel_words_unigrams, rel_words_bigrams)) for i in range(num_cpus)]

    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()

    a = list()
    for l in results:
        a.extend(l)

    wrong = list()
    for l in not_found:
        wrong.extend(l)

    return a, wrong


@timecall
def calculate_b(output, database_1, database_2, database_3, e1_type, e2_type):
    # intersection between the system output and the database
    # it is assumed that every fact in this region is correct
    m = multiprocessing.Manager()
    queue = m.Queue()
    num_cpus = multiprocessing.cpu_count()
    results = [m.list() for _ in range(num_cpus)]
    no_matches = [m.list() for _ in range(num_cpus)]

    for r in output:
        queue.put(r)

    processes = [multiprocessing.Process(
        target=string_matching_parallel,
        args=(results[i], no_matches[i], database_1, database_2, database_3,
              queue, e1_type, e2_type))
                 for i in range(num_cpus)]

    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()

    b = set()
    for l in results:
        b.update(l)

    not_found = set()
    for l in no_matches:
        not_found.update(l)

    return b, not_found


@timecall
def calculate_c(corpus, database_1, database_2, database_3, b, e1_type, e2_type,
                rel_type, rel_words_unigrams, rel_words_bigrams):

    # contains the database facts described in the corpus
    # but not extracted by the system
    #
    # G' = superset of G, cartesian product of all possible entities and
    # relations (i.e., G' = E x R x E)
    # for now, all relationships from a sentence
    print("Building G', a superset of G")
    m = multiprocessing.Manager()
    queue = m.Queue()
    g_dash = m.list()
    num_cpus = multiprocessing.cpu_count()

    # check if superset G' for e1_type, e2_type already exists and
    # if G' minus KB for rel_type exists

    # if it exists load into g_dash_set
    if os.path.isfile("superset_" + e1_type + "_" + e2_type + ".pkl"):
        f = open("superset_" + e1_type + "_" + e2_type + ".pkl")
        print("\nLoading superset G'", "superset_" + e1_type + "_" + \
                                       e2_type + ".pkl")
        g_dash_set = pickle.load(f)
        f.close()

    # else generate G' and G minus D
    else:
        with open(corpus) as f:
            data = f.readlines()
            count = 0
            print("Storing in shared Queue")
            for l in data:
                if count % 50000 == 0:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                queue.put(l)
                count += 1
        print("\nQueue size:", queue.qsize())

        processes = [multiprocessing.Process(
            target=process_corpus,
            args=(queue, g_dash, e1_type, e2_type))
                     for _ in range(num_cpus)]

        print("Extracting all possible " + e1_type + "," + e2_type + \
              " relationships from the corpus")
        print("Running", len(processes), "threads")

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        print(len(g_dash), "relationships built")
        g_dash_set = set(g_dash)
        print(len(g_dash_set), "unique relationships")
        print("Dumping into file", "superset_" + e1_type + "_" + e2_type + ".pkl")
        f = open("superset_" + e1_type + "_" + e2_type + ".pkl", "wb")
        pickle.dump(g_dash_set, f)
        f.close()

    # Estimate G \in D, look for facts in G' that a match a fact in the database
    # check if already exists for this particular relationship
    if os.path.isfile(rel_type + "_g_intersection_d.pkl") and \
            os.path.isfile(rel_type + "_g_minus_d.pkl"):
        f = open(rel_type + "_g_intersection_d.pkl", "r")
        print("\nLoading G intersected with D", rel_type + "_g_intersection_d.pkl")
        g_intersect_d = pickle.load(f)
        f.close()

        f = open(rel_type + "_g_minus_d.pkl")
        print("\nLoading superset G' minus D", rel_type + "_g_minus_d.pkl")
        g_minus_d = pickle.load(f)
        f.close()

    else:
        print("Estimating G intersection with D")
        g_intersect_d = set()
        print("G':", len(g_dash_set))
        print("Database:", len(list(database_1.keys())))

        # Facts not in the database, to use in estimating set d
        g_minus_d = set()

        queue = manager.Queue()
        results = [manager.list() for _ in range(num_cpus)]
        no_matches = [manager.list() for _ in range(num_cpus)]

        # Load everything into a shared queue
        for r in g_dash_set:
            queue.put(r)

        processes = [multiprocessing.Process(
            target=string_matching_parallel,
            args=(results[i], no_matches[i],
                  database_1, database_2, database_3, queue, e1_type, e2_type))
                     for i in range(num_cpus)]

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        for l in results:
            g_intersect_d.update(l)

        for l in no_matches:
            g_minus_d.update(l)

        print("Extra filtering: from the intersection of G' with D, " \
              "select only those based on keywords")
        print(len(g_intersect_d))
        filtered = set()
        for r in g_intersect_d:
            unigrams_bet = word_tokenize(r.between)
            unigrams_bef = word_tokenize(r.before)
            unigrams_aft = word_tokenize(r.after)
            bigrams_bet = extract_bigrams(r.between)
            if any(x in rel_words_unigrams for x in unigrams_bet):
                filtered.add(r)
                continue
            if any(x in rel_words_unigrams for x in unigrams_bef):
                filtered.add(r)
                continue
            if any(x in rel_words_unigrams for x in unigrams_aft):
                filtered.add(r)
                continue
            elif any(x in rel_words_bigrams for x in bigrams_bet):
                filtered.add(r)
                continue
        g_intersect_d = filtered
        print(len(g_intersect_d), "relationships in the corpus " \
                                  "which are in the KB")
        if len(g_intersect_d) > 0:
            # dump G intersected with D to file
            f = open(rel_type + "_g_intersection_d.pkl", "wb")
            pickle.dump(g_intersect_d, f)
            f.close()

        print("Extra filtering: from the G' not in D, select only " \
              "those based on keywords")
        filtered = set()
        for r in g_minus_d:
            unigrams_bet = word_tokenize(r.between)
            unigrams_bef = word_tokenize(r.before)
            unigrams_aft = word_tokenize(r.after)
            bigrams_bet = extract_bigrams(r.between)
            if any(x in rel_words_unigrams for x in unigrams_bet):
                filtered.add(r)
                continue
            if any(x in rel_words_unigrams for x in unigrams_bef):
                filtered.add(r)
                continue
            if any(x in rel_words_unigrams for x in unigrams_aft):
                filtered.add(r)
                continue
            elif any(x in rel_words_bigrams for x in bigrams_bet):
                filtered.add(r)
                continue
        g_minus_d = filtered
        print(len(g_minus_d), "relationships in the corpus not in the KB")
        if len(g_minus_d) > 0:
            # dump G - D to file, relationships in the corpus not in KB
            f = open(rel_type + "_g_minus_d.pkl", "wb")
            pickle.dump(g_minus_d, f)
            f.close()

    # having B and G_intersect_D => |c| = |G_intersect_D| - |b|
    c = g_intersect_d.difference(set(b))
    assert len(g_minus_d) > 0
    return c, g_minus_d


@timecall
def calculate_d(g_minus_d, a, e1_type, e2_type, index, rel_type,
                rel_words_unigrams, rel_words_bigrams):

    # contains facts described in the corpus that are not
    # in the system output nor in the database
    #
    # by applying the PMI of the facts not in the database (i.e., G' \in D)
    # we determine |G \ D|, then we can estimate |d| = |G \ D| - |a|
    #
    # |G' \ D|
    # determine facts not in the database, with high PMI, that is,
    # facts that are true and are not in the database

    # check if it was already calculated and stored in disk
    if os.path.isfile(rel_type + "_high_pmi_not_in_database.pkl"):
        f = open(rel_type + "_high_pmi_not_in_database.pkl")
        print("\nLoading high PMI facts not in the database", \
            rel_type + "_high_pmi_not_in_database.pkl")
        g_minus_d = pickle.load(f)
        f.close()

    else:
        m = multiprocessing.Manager()
        queue = m.Queue()
        num_cpus = multiprocessing.cpu_count()
        results = [m.list() for _ in range(num_cpus)]

        for r in g_minus_d:
            queue.put(r)

        # calculate PMI for r not in database
        processes = [multiprocessing.Process(
            target=proximity_pmi_rel_word,
            args=(e1_type, e2_type, queue, index,
                  results[i], rel_words_unigrams, rel_words_bigrams))
                     for i in range(num_cpus)]

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        g_minus_d = set()
        for l in results:
            g_minus_d.update(l)

        print("High PMI facts not in the database", len(g_minus_d))

        # dump high PMI facts not in the database
        if len(g_minus_d) > 0:
            f = open(rel_type + "_high_pmi_not_in_database.pkl", "wb")
            print("Dumping high PMI facts not in the database to", \
                rel_type + "_high_pmi_not_in_database.pkl")
            pickle.dump(g_minus_d, f)
            f.close()

    return g_minus_d.difference(a)


########################################################################
# Parallelized functions: each function will run as a different process #
########################################################################
def proximity_pmi_rel_word(e1_type, e2_type, queue, index, results,
                           rel_words_unigrams, rel_words_bigrams):
    idx = open_dir(index)
    count = 0
    distance = MAX_TOKENS_AWAY
    q_limit = 500
    with idx.searcher() as searcher:
        while True:
            try:
                r = queue.get_nowait()
                if count % 50 == 0:
                    print("\n", multiprocessing.current_process(), \
                        "In Queue", queue.qsize(), \
                        "Total Matched: ", len(results))
                if (r.ent1, r.ent2) not in all_in_database:
                    # if its not in the database calculate the PMI
                    entity1 = "<" + e1_type + ">" + r.ent1 + "</" + e1_type + ">"
                    entity2 = "<" + e2_type + ">" + r.ent2 + "</" + e2_type + ">"
                    t1 = query.Term('sentence', entity1)
                    t3 = query.Term('sentence', entity2)

                    # Entities proximity query without relational words
                    q1 = spans.SpanNear2(
                        [t1, t3], slop=distance,
                        ordered=True, mindist=1)
                    hits = searcher.search(q1, limit=q_limit)

                    # Entities proximity considering relational words
                    # From the results above count how many contain a
                    # valid relational word

                    hits_with_r = 0
                    hits_without_r = 0
                    for s in hits:
                        sentence = s.get("sentence")
                        s = Sentence(sentence, e1_type, e2_type,
                                     MAX_TOKENS_AWAY, MIN_TOKENS_AWAY,
                                     CONTEXT_WINDOW)

                        for s_r in s.relationships:
                            if r.ent1.decode("utf8") == s_r.ent1 and \
                                            r.ent2.decode("utf8") == s_r.ent2:

                                unigrams_rel_words = word_tokenize(s_r.between)
                                bigrams_rel_words = extract_bigrams(s_r.between)

                                if all(x in not_valid
                                       for x in unigrams_rel_words):
                                    hits_without_r += 1
                                    continue
                                elif any(x in rel_words_unigrams for x in
                                         unigrams_rel_words):

                                    hits_with_r += 1

                                elif any(x in rel_words_bigrams
                                         for x in bigrams_rel_words):

                                    hits_with_r += 1
                                else:
                                    hits_without_r += 1

                    if hits_with_r > 0 and hits_without_r > 0:
                        pmi = float(hits_with_r) / float(hits_without_r)
                        if pmi >= PMI:
                            if word_tokenize(s_r.between)[-1] == 'by':
                                tmp = s_r.ent2
                                s_r.ent2 = s_r.ent1
                                s_r.ent1 = tmp
                            results.append(r)

                count += 1
            except queue.Empty:
                break


def string_matching_parallel(matches, no_matches, database_1, database_2,
                             database_3, queue, e1_type, e2_type):
    count = 0
    while True:
        try:
            r = queue.get_nowait()
            found = False
            count += 1
            if count % 500 == 0:
                print(multiprocessing.current_process(), \
                    "In Queue", queue.qsize())

            # check if its in cache, i.e., if tuple was already matched
            if (r.ent1, r.ent2) in all_in_database:
                matches.append(r)
                found = True

            # check for a relationship with a direct string matching
            if found is False:
                if len(database_1[(r.ent1.decode("utf8"),
                                   r.ent2.decode("utf8"))]) > 0:
                    matches.append(r)
                    all_in_database[(r.ent1, r.ent2)] = "Found"
                    found = True

            if found is False:
                # database_2: arg_1 rel list(arg_2)
                # check for a direct string matching with all possible arg2
                # FOUNDER   : r.ent1:ORG   r.ent2:PER
                # DATABASE_1: (ORG,PER)
                # DATABASE_2: ORG   list<PER>
                # DATABASE_3: PER   list<ORG>

                ent2 = database_2[r.ent1.decode("utf8")]
                if len(ent2) > 0:
                    if r.ent2 in ent2:
                        matches.append(r)
                        all_in_database[(r.ent1, r.ent2)] = "Found"
                        found = True

            # if a direct string matching occur with arg_2, check for a
            # direct string matching with all possible arg1 entities
            if found is False:
                arg1_list = database_3[r.ent2]
                if arg1_list is not None:
                    for arg1 in arg1_list:
                        if e1_type == 'ORG':
                            new_arg1 = re.sub(r" Corporation| Inc\.", "", arg1)
                        else:
                            new_arg1 = arg1

                        # Jaccardi
                        set_1 = set(new_arg1.split())
                        set_2 = set(r.ent1.split())

                        jaccardi = \
                            float(len(set_1.intersection(set_2))) / \
                            float(len(set_1.union(set_2)))

                        if jaccardi >= 0.5:
                            matches.append(r)
                            all_in_database[(r.ent1, r.ent2)] = "Found"
                            found = True

                        # Jaro Winkler
                        elif jaccardi <= 0.5:
                            score = jellyfish.jaro_winkler(
                                new_arg1.upper(), r.ent1.upper()
                            )
                            if score >= 0.9:
                                matches.append(r)
                                all_in_database[(r.ent1, r.ent2)] = "Found"
                                found = True

            # if a direct string matching occur with arg_1,
            # check for a direct string matching
            # with all possible arg_2 entities
            if found is False:
                arg2_list = database_2[r.ent1]
                if arg2_list is not None:
                    for arg2 in arg2_list:
                        # Jaccardi
                        if e1_type == 'ORG':
                            new_arg2 = re.sub(r" Corporation| Inc\.", "", arg2)
                        else:
                            new_arg2 = arg2
                        set_1 = set(new_arg2.split())
                        set_2 = set(r.ent2.split())
                        jaccardi = \
                            float(len(set_1.intersection(set_2))) / \
                            float(len(set_1.union(set_2)))

                        if jaccardi >= 0.5:
                            matches.append(r)
                            all_in_database[(r.ent1, r.ent2)] = "Found"
                            found = True

                        # Jaro Winkler
                        elif jaccardi <= 0.5:
                            score = jellyfish.jaro_winkler(
                                new_arg2.upper(), r.ent2.upper()
                            )
                            if score >= 0.9:
                                matches.append(r)
                                all_in_database[(r.ent1, r.ent2)] = "Found"
                                found = True

            if found is False:
                no_matches.append(r)
                if PRINT_NOT_FOUND is True:
                    print(r.ent1, '\t', r.ent2)

        except queue.Empty:
            break


def proximity_pmi_a(e1_type, e2_type, queue, index, results, not_found,
                    rel_words_unigrams, rel_words_bigrams):
    idx = open_dir(index)
    count = 0
    q_limit = 500
    with idx.searcher() as searcher:
        while True:
            try:
                r = queue.get_nowait()
                count += 1
                if count % 50 == 0:
                    print(multiprocessing.current_process(), \
                        "To Process", queue.qsize(), \
                        "Correct found:", len(results))

                # if its not in the database calculate the PMI
                entity1 = "<" + e1_type + ">" + r.ent1 + "</" + e1_type + ">"
                entity2 = "<" + e2_type + ">" + r.ent2 + "</" + e2_type + ">"
                t1 = query.Term('sentence', entity1)
                t3 = query.Term('sentence', entity2)

                # First count the proximity (MAX_TOKENS_AWAY) occurrences
                # of entities r.e1 and r.e2
                q1 = spans.SpanNear2([t1, t3],
                                     slop=MAX_TOKENS_AWAY,
                                     ordered=True,
                                     mindist=1)
                hits = searcher.search(q1, limit=q_limit)

                # Entities proximity considering relational words
                # From the results above count how many contain a
                # valid relational word
                hits_with_r = 0
                hits_without_r = 0
                fact_bet_words_tokens = word_tokenize(r.bet_words)
                for s in hits:
                    sentence = s.get("sentence")
                    s = Sentence(sentence, e1_type, e2_type, MAX_TOKENS_AWAY,
                                 MIN_TOKENS_AWAY, CONTEXT_WINDOW)
                    for s_r in s.relationships:
                        if r.ent1.decode("utf8") == s_r.ent1 and \
                                        r.ent2.decode("utf8") == s_r.ent2:
                            unigrams_bef_words = word_tokenize(s_r.before)
                            unigrams_bet_words = word_tokenize(s_r.between)
                            unigrams_aft_words = word_tokenize(s_r.after)
                            bigrams_rel_words = extract_bigrams(s_r.between)

                            if fact_bet_words_tokens == unigrams_bet_words:
                                hits_with_r += 1

                            elif any(x in rel_words_unigrams
                                     for x in unigrams_bef_words):
                                hits_with_r += 1

                            elif any(x in rel_words_unigrams
                                     for x in unigrams_bet_words):
                                hits_with_r += 1

                            elif any(x in rel_words_unigrams
                                     for x in unigrams_aft_words):
                                hits_with_r += 1

                            elif rel_words_bigrams == bigrams_rel_words:
                                hits_with_r += 1
                            else:
                                hits_without_r += 1

                if hits_with_r > 0 and hits_without_r > 0:
                    pmi = float(hits_with_r) / float(hits_without_r)
                    if pmi >= PMI:
                        results.append(r)

                    else:
                        not_found.append(r)

                else:
                    not_found.append(r)
                count += 1

            except queue.Empty:
                break


def main():
    # "Automatic Evaluation of Relation Extraction Systems on Large-scale"
    # https://akbcwekex2012.files.wordpress.com/2012/05/8_paper.pdf
    #
    # S  - system output
    # D  - database (freebase)
    # G  - will be the resulting ground truth
    # G' - superset, contains true facts, and wrong facts
    # a  - contains correct facts from the system output
    #
    # b  - intersection between the system output and the
    #      database (i.e., freebase),
    #      it is assumed that every fact in this region is correct
    # c  - contains the database facts described in the corpus
    #      but not extracted by the system
    # d  - contains the facts described in the corpus that are not
    #      in the system output nor in the database
    #
    # Precision = |a|+|b| / |S|
    # Recall    = |a|+|b| / |a| + |b| + |c| + |d|
    # F1        = 2*P*R / P+R

    if len(sys.argv) == 1:
        print("No arguments")
        print("Use: evaluation.py threshold system_output rel_type database")
        print("\n")
        sys.exit(0)

    threhsold = float(sys.argv[1])
    rel_type = sys.argv[3]

    # load relationships extracted by the system
    system_output = process_output(sys.argv[2], threhsold, rel_type)
    print("Relationships score threshold :", threhsold)
    print("System output relationships   :", len(system_output))

    # load freebase relationships as the database
    database_1, database_2, database_3 = process_freebase(sys.argv[4], rel_type)
    print("Freebase relationships loaded :", len(list(database_1.keys())))

    # corpus from which the system extracted relationships
    corpus = "/home/dsbatista/gigaword/automatic-evaluation/" \
             "sentences_matched_freebase_added_tags.txt"

    # index to be used to estimate proximity PMI
    index = "/home/dsbatista/gigaword/automatic-evaluation/index_full"

    # entities semantic type
    rel_words_unigrams = None
    rel_words_bigrams = None

    if rel_type == 'founder':
        e1_type = "ORG"
        e2_type = "PER"
        rel_words_unigrams = founded_unigrams
        rel_words_bigrams = founded_bigrams

    elif rel_type == 'acquired':
        e1_type = "ORG"
        e2_type = "ORG"
        rel_words_unigrams = acquired_unigrams
        rel_words_bigrams = acquired_unigrams

    elif rel_type == 'headquarters':
        # load dbpedia relationships
        print("Loading extra DBPedia relationships for", rel_type)
        load_dbpedia(sys.argv[5], database_1, database_2)
        e1_type = "ORG"
        e2_type = "LOC"
        rel_words_unigrams = headquarters_unigrams
        rel_words_bigrams = headquarters_bigrams

    elif rel_type == 'contained_by':
        e1_type = "LOC"
        e2_type = "LOC"

    elif rel_type == 'employer':
        e1_type = "ORG"
        e2_type = "PER"
        rel_words_unigrams = employment_unigrams
        rel_words_bigrams = employment_bigrams

    else:
        print("Invalid relationship type", rel_type)
        print("Use: founder, acquired, headquarters, employer")
        sys.exit(0)

    print("\nRelationship Type:", rel_type)
    print("Arg1 Type:", e1_type)
    print("Arg2 Type:", e2_type)

    print("\nCalculating set B: intersection between system output and database")
    b, not_in_database = calculate_b(system_output, database_1, database_2,
                                     database_3, e1_type, e2_type)

    print("System output      :", len(system_output))
    print("Found in database  :", len(b))
    print("Not found          :", len(not_in_database))
    assert len(system_output) == len(not_in_database) + len(b)

    print("\nCalculating set A: correct facts from system output not in " \
          "the database (proximity PMI)")
    a, not_found = calculate_a(not_in_database, e1_type, e2_type, index,
                               rel_words_unigrams, rel_words_bigrams)

    print("System output      :", len(system_output))
    print("Found in database  :", len(b))
    print("Correct in corpus  :", len(a))
    print("Not found          :", len(not_found))
    print("\n")
    assert len(system_output) == len(a) + len(b) + len(not_found)

    # Estimate G \intersected D = |b| + |c|, looking for relationships in G'
    # that match a relationship in D, once we have G \in D and |b|, |c| can be
    # derived by: |c| = |G \in D| - |b| G' = superset of G, cartesian product
    # of all possible entities and relations (i.e., G' = E x R x E)
    print("\nCalculating set C: database facts in the corpus but not " \
          "extracted by the system")
    c, g_minus_d = calculate_c(corpus, database_1, database_2, database_3, b,
                               e1_type, e2_type, rel_type, rel_words_unigrams,
                               rel_words_bigrams)
    assert len(c) > 0

    uniq_c = set()
    for r in c:
        uniq_c.add((r.ent1, r.ent2))

    # By applying the PMI of the facts not in the database (i.e., G' \in D)
    # we determine |G \ D|, then we can estimate |d| = |G \ D| - |a|
    print("\nCalculating set D: facts described in the corpus not in " \
          "the system output nor in the database")
    d = calculate_d(g_minus_d, a, e1_type, e2_type, index, rel_type,
                    rel_words_unigrams, rel_words_bigrams)

    print("System output      :", len(system_output))
    print("Found in database  :", len(b))
    print("Correct in corpus  :", len(a))
    print("Not found          :", len(not_found))
    print("\n")
    assert len(d) > 0

    uniq_d = set()
    for r in d:
        uniq_d.add((r.ent1, r.ent2))

    print("|a| =", len(a))
    print("|b| =", len(b))
    print("|c| =", len(c), "(", len(uniq_c), ")")
    print("|d| =", len(d), "(", len(uniq_d), ")")
    print("|S| =", len(system_output))
    print("|G| =", len(set(a).union(set(b).union(set(c).union(set(d))))))
    print("Relationships not found:", len(set(not_found)))

    # Write relationships not found in the Database nor with high PMI
    # relational words to disk
    f = open(rel_type + "_" + sys.argv[2][-11:][:-4] + "_negative.txt", "w")
    for r in sorted(set(not_found), reverse=True):
        f.write('instance :' + r.ent1 + '\t' + r.ent2 + '\t' + str(r.score) +
                '\n')
        f.write('sentence :' + r.sentence + '\n')
        f.write('bef_words:' + r.bef_words + '\n')
        f.write('bet_words:' + r.bet_words + '\n')
        f.write('aft_words:' + r.aft_words + '\n')
        f.write('\n')
    f.close()

    # Write all correct relationships (sentence, entities and score) to file
    f = open(rel_type + "_" + sys.argv[2][-11:][:-4] + "_positive.txt", "w")
    for r in sorted(set(a).union(b), reverse=True):
        f.write('instance :' + r.ent1 + '\t' + r.ent2 + '\t' + str(r.score) +
                '\n')
        f.write('sentence :' + r.sentence + '\n')
        f.write('bef_words:' + r.bef_words + '\n')
        f.write('bet_words:' + r.bet_words + '\n')
        f.write('aft_words:' + r.aft_words + '\n')
        f.write('\n')
    f.close()

    a = set(a)
    b = set(b)
    output = set(system_output)
    if len(output) == 0:
        print("\nPrecision   : 0.0")
        print("Recall      : 0.0")
        print("F1          : 0.0")
        print("\n")
    elif float(len(a) + len(b)) == 0:
        print("\nPrecision   : 0.0")
        print("Recall      : 0.0")
        print("F1          : 0.0")
        print("\n")
    else:
        precision = float(len(a) + len(b)) / float(len(output))
        recall = float(len(a) + len(b)) / float(len(a) + len(b) + len(uniq_c) +
                                                len(uniq_d))
        f1 = 2 * (precision * recall) / (precision + recall)
        print("\nPrecision   : ", precision)
        print("Recall      : ", recall)
        print("F1          : ", f1)
        print("\n")

if __name__ == "__main__":
    main()
