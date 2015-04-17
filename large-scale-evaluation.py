#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import fileinput
import functools
import multiprocessing
import re
import time
import sys
import cPickle
import jellyfish
import Queue

from whoosh.index import open_dir, os
from whoosh.query import spans
from whoosh import query
from nltk import PunktWordTokenizer
from nltk.corpus import stopwords
from collections import defaultdict
from Common.Sentence import Sentence

# relational words to be used in calculating the set D with the proximity PMI
founded = ['founder', 'co-founder', 'cofounder', 'cofounded', 'founded']
acquired = ['owns', 'acquired', 'bought']
headquarters = ['headquarters', 'headquartered', 'based', 'located', 'offices']
employment = ['chief', 'scientist', 'professor', 'biologist', 'ceo']

# PMI value for proximity
PMI = 0.7

# Parameters for relationship extraction from Sentence
# 9 tokens away, cause it was indexed was done without removing stop-words...
MAX_TOKENS_AWAY = 9
MIN_TOKENS_AWAY = 1
CONTEXT_WINDOW = 2

# DEBUG stuff
PRINT_NOT_FOUND = False

# stores all variations matched with database
manager = multiprocessing.Manager()
all_in_freebase = manager.dict()


class ExtractedFact(object):
    def __init__(self, _e1, _e2, _score, _bef, _bet, _aft, _sentence, _passive_voice):
        self.ent1 = _e1
        self.ent2 = _e2
        self.score = _score.strip()
        self.bef_words = _bef
        self.bet_words = _bet
        self.aft_words = _aft
        self.sentence = _sentence
        self.passive_voice = _passive_voice

    def __hash__(self):
        sig = hash(self.ent1) ^ hash(self.ent2) ^ hash(self.bef_words) ^ hash(self.bet_words) ^ hash(self.aft_words) ^ \
              hash(self.score) ^ hash(self.sentence)
        return sig

    def __eq__(self, other):
        if self.ent1 == other.ent1 and self.ent2 == other.ent2 and self.score == other.score and self.patterns == \
                other.patterns and self.sentence == other.sentence:
            return True
        else:
            return False


############################################
# Misc., Utils, parsing corpus into memory #
############################################

def timecall(f):
    @functools.wraps(f)
    def wrapper(*args, **kw):
        start = time.time()
        result = f(*args, **kw)
        end = time.time()
        #print "%s %.2f seconds" % (f.__name__, end - start)
        print "Time taken: %.2f seconds" % (end - start)
        return result
    return wrapper


def is_acronym(entity):
    if len(entity.split()) == 1 and entity.isupper():
        return True
    else:
        return False


def process_corpus(queue, g_dash, e1_type, e2_type):
    while True:
        try:
            line = queue.get_nowait()
            s = Sentence(line.strip(), e1_type, e2_type, MAX_TOKENS_AWAY, MIN_TOKENS_AWAY, CONTEXT_WINDOW)
            for r in s.relationships:
                if r.between == " , " or r.between == " ( " or r.between == " ) ":
                    continue
                else:
                    g_dash.append(r)
            if queue.empty() is True:
                break

        except Queue.Empty:
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
            if passive_voice is True and rel_type in ['acquired', 'headquarters']:
                r = ExtractedFact(e2, e1, score, bef, bet, aft, sentence, passive_voice)
            else:
                r = ExtractedFact(e1, e2, score, bef, bet, aft, sentence, passive_voice)
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
    numbered = re.compile('#[0-9]+$')

    # for the 'founder' relationships don't load those from freebase, as it lists countries as founders and not persons
    founder_to_ignore = ['UNESCO', 'World Trade Organization', 'European Union', 'United Nations']

    for line in fileinput.input(data):
        e1, r, e2 = line.split('\t')
        # ignore some entities, which are Freebase identifiers or which are ambigious
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

            if rel_type == 'founded' or rel_type == 'employment':
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


#########################################
# Estimations of sets and intersections #
#########################################
@timecall
def calculate_a(output, e1_type, e2_type, index):

    m = multiprocessing.Manager()
    queue = m.Queue()
    num_cpus = multiprocessing.cpu_count()
    results = [m.list() for _ in range(num_cpus)]
    not_found = [m.list() for _ in range(num_cpus)]

    for r in output:
        queue.put(r)

    processes = [multiprocessing.Process(target=proximity_pmi_a, args=(e1_type, e2_type, queue, index, results[i],
                                                                       not_found[i])) for i in range(num_cpus)]
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
    # intersection between the system output and the database (i.e., freebase),
    # it is assumed that every fact in this region is correct
    m = multiprocessing.Manager()
    queue = m.Queue()
    num_cpus = multiprocessing.cpu_count()
    results = [m.list() for _ in range(num_cpus)]
    no_matches = [m.list() for _ in range(num_cpus)]

    for r in output:
        queue.put(r)

    processes = [multiprocessing.Process(target=string_matching_parallel, args=(results[i], no_matches[i], database_1,
                                                                                database_2, database_3, queue,
                                                                                e1_type, e2_type))
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
def calculate_c(corpus, database_1, database_2, database_3, b, e1_type, e2_type, rel_type):
    # contains the database facts described in the corpus but not extracted by the system
    #
    # G' = superset of G, cartesian product of all possible entities and relations (i.e., G' = E x R x E)
    # for now, all relationships from a sentence
    print "Building G', a superset of G"
    m = multiprocessing.Manager()
    queue = m.Queue()
    g_dash = m.list()
    num_cpus = multiprocessing.cpu_count()

    # check if superset G' for e1_type, e2_type already exists
    # if it exists load into g_dash_set
    if os.path.isfile("superset_"+e1_type+"_"+e2_type+".pkl"):
        f = open("superset_"+e1_type+"_"+e2_type+".pkl")
        print "\nLoading superset G'", "superset_"+e1_type+"_"+e2_type+".pkl"
        g_dash_set = cPickle.load(f)
        f.close()

    # else generate G'
    else:
        with open(corpus) as f:
            print "Reading corpus into memory"
            data = f.readlines()
            print "Storing in shared Queue"
            for l in data:
                queue.put(l)
        print "Queue size:", queue.qsize()

        processes = [multiprocessing.Process(target=process_corpus, args=(queue, g_dash, e1_type, e2_type))
                     for _ in range(num_cpus)]
        print "Extracting all possible "+e1_type+","+e2_type+" relationships from the corpus"
        print "Running", len(processes), "threads"

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        print len(g_dash), "relationships built"
        g_dash_set = set(g_dash)
        print len(g_dash_set), "unique relationships"
        print "Dumping into file", "superset_"+e1_type+"_"+e2_type+".pkl"
        f = open("superset_"+e1_type+"_"+e2_type+".pkl", "wb")
        cPickle.dump(g_dash_set, f)
        f.close()

    # Estimate G \in D, look for facts in G' that a match a fact in the database
    # check if already exists for this particular relationship
    if os.path.isfile(rel_type+"_g_intersection_d.pkl"):
        f = open(rel_type+"_g_intersection_d.pkl", "r")
        print "\nLoading G intersected with D'", rel_type+"_g_intersection_d.pkl"
        g_intersect_d = cPickle.load(f)
        f.close()

    else:
        print "Estimating G intersection with D"
        g_intersect_d = set()
        print "G':", len(g_dash_set)
        print "Database:", len(database_1.keys())

        queue = manager.Queue()
        results = [manager.list() for _ in range(num_cpus)]
        no_matches = [manager.list() for _ in range(num_cpus)]

        # Load everything into a shared queue
        for r in g_dash_set:
            queue.put(r)

        processes = [multiprocessing.Process(target=string_matching_parallel, args=(results[i], no_matches[i],
                                                                                    database_1, database_2,
                                                                                    database_3, queue, e1_type,
                                                                                    e2_type))
                     for i in range(num_cpus)]

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        for l in results:
            g_intersect_d.update(l)

        if len(g_intersect_d) > 0:
            # dump G intersected with D to file
            f = open(rel_type+"_g_intersection_d.pkl", "wb")
            cPickle.dump(g_intersect_d, f)
            f.close()

    # having B and G_intersect_D => C = G_intersect_D - B
    c = g_intersect_d.difference(set(b))
    return c, g_dash_set


@timecall
def calculate_d(g_dash, a, e1_type, e2_type, index, rel_type):
    # contains facts described in the corpus that are not in the system output nor in the database
    #
    # by applying the PMI of the facts not in the database (i.e., G' \in D)
    # we determine |G \ D|, then we can estimate |d| = |G \ D| - |a|
    #
    # |G' \ D|
    # determine facts not in the database, with high PMI, that is, facts that are true and are not in the database

    if rel_type == "founded":
        rel_words = founded
    elif rel_type == "acquired":
        rel_words = acquired
    elif rel_type == 'headquarters':
        rel_words = headquarters
    elif rel_type == 'employment':
        rel_words = employment
    else:
        print rel_type, " is invalid"
        sys.exit(0)

    # check if it was already calculated and stored in disk
    if os.path.isfile(rel_type+"_high_pmi_not_in_database.pkl"):
        f = open(rel_type+"_high_pmi_not_in_database.pkl")
        print "\nLoading high PMI facts not in the database", rel_type+"_high_pmi_not_in_database.pkl"
        g_minus_d = cPickle.load(f)
        f.close()

    else:
        m = multiprocessing.Manager()
        queue = m.Queue()
        num_cpus = multiprocessing.cpu_count()
        results = [m.list() for _ in range(num_cpus)]

        for r in g_dash:
            queue.put(r)

        # calculate PMI for r not in database
        processes = [multiprocessing.Process(target=proximity_pmi_rel_word, args=(e1_type, e2_type, queue, index,
                                                                                  results[i], rel_words))
                     for i in range(num_cpus)]
        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        g_minus_d = set()
        for l in results:
            g_minus_d.update(l)

        # dump high PMI facts not in the database
        if len(g_minus_d) > 0:
            f = open(rel_type+"_high_pmi_not_in_database.pkl", "wb")
            print "Dumping high PMI facts not in the database to", rel_type+"_high_pmi_not_in_database.pkl"
            cPickle.dump(g_minus_d, f)
            f.close()

    return g_minus_d.difference(a)


########################################################################
# Paralelized functions: each function will run as a different process #
########################################################################
def proximity_pmi_rel_word(e1_type, e2_type, queue, index, results, rel_words):
    """
    #TODO: proximity_pmi with relation specific given relational words
    :param e1_type:
    :param e2_type:
    :param queue:
    :param index:
    :param results:
    :param rel_words:
    :return:
    """
    """
    sentences with tagged entities are indexed in whoosh
    perform the following query
    ent1 NEAR:X r NEAR:X ent2
    X is the maximum number of words between the query elements.
    """
    idx = open_dir(index)
    count = 0
    distance = MAX_TOKENS_AWAY
    q_limit = 500
    with idx.searcher() as searcher:
        while True:
            try:
                r = queue.get_nowait()
                count += 1
                if count % 100 == 0:
                    print multiprocessing.current_process(), "In Queue", queue.qsize(), "Total Matched: ", len(results)
                if (r.ent1, r.ent2) not in all_in_freebase:
                    # if its not in the database calculate the PMI
                    entity1 = "<"+e1_type+">"+r.ent1+"</"+e1_type+">"
                    entity2 = "<"+e2_type+">"+r.ent2+"</"+e2_type+">"
                    t1 = query.Term('sentence', entity1)
                    t3 = query.Term('sentence', entity2)

                    # Entities proximity query without relational words
                    q1 = spans.SpanNear2([t1, t3], slop=distance, ordered=True, mindist=1)
                    hits = searcher.search(q1, limit=q_limit)

                    # Entities proximity considering relational words
                    # From the results above count how many contain a relational word
                    #TODO: maybe the relational words can be in the BEF or AFT context
                    hits_with_r = 0
                    for s in hits:
                        sentence = s.get("sentence")
                        s = Sentence(sentence, e1_type, e2_type, MAX_TOKENS_AWAY, MIN_TOKENS_AWAY, CONTEXT_WINDOW)
                        for s_r in s.relationships:
                            if r.ent1.decode("utf8") == s_r.ent1 and r.ent2.decode("utf8") == s_r.ent2:
                                for rel in rel_words:
                                    if rel in r.between:
                                        hits_with_r += 1
                                        break

                        if not len(hits) >= hits_with_r:
                            print "ERROR!"
                            print "hits", len(hits)
                            print "hits_with_r", hits_with_r
                            print entity1, '\t', entity2
                            print "\n"
                            sys.exit(0)

                    if len(hits) > 0:
                        pmi = float(hits_with_r) / float(len(hits))
                        if pmi > PMI:
                            results.append(r)
                            """
                            if isinstance(r, ExtractedFact):
                                print r.ent1, '\t', r.patterns, '\t', r.ent2, pmi
                            elif isinstance(r, Relationship):
                                print r.ent1, '\t', r.between, '\t', r.ent2, pmi
                            """
                    if queue.empty() is True:
                        break

            except Queue.Empty:
                break


def string_matching_parallel(matches, no_matches, database_1, database_2, database_3, queue, e1_type, e2_type):
    count = 0
    while True:
        try:
            r = queue.get_nowait()
            found = False
            count += 1
            if count % 500 == 0:
                print multiprocessing.current_process(), "In Queue", queue.qsize()

            # check if its in cache, i.e., if tuple was already matched
            if (r.ent1, r.ent2) in all_in_freebase:
                matches.append(r)
                found = True

            # check for a relationship with a direct string matching
            if found is False:
                if len(database_1[(r.ent1.decode("utf8"), r.ent2.decode("utf8"))]) > 0:
                    matches.append(r)
                    all_in_freebase[(r.ent1, r.ent2)] = "Found"
                    found = True

            if found is False:
                # database_2: arg_1 rel list(arg_2)
                # check for a direct string matching with all possible arg2 entities
                # FOUNDER   : r.ent1:ORG   r.ent2:PER
                # DATABASE_1: (ORG,PER)
                # DATABASE_2: ORG   list<PER>
                # DATABASE_3: PER   list<ORG>

                ent2 = database_2[r.ent1.decode("utf8")]
                if len(ent2) > 0:
                    if r.ent2 in ent2:
                        matches.append(r)
                        all_in_freebase[(r.ent1, r.ent2)] = "Found"
                        found = True

            # if a direct string matching occur with arg_2, check for a direct string matching
            # with all possible arg1 entities
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
                        jaccardi = float(len(set_1.intersection(set_2))) / float(len(set_1.union(set_2)))
                        if jaccardi >= 0.5:
                            matches.append(r)
                            all_in_freebase[(r.ent1, r.ent2)] = "Found"
                            found = True

                        # Jaro Winkler
                        elif jaccardi <= 0.5:
                            score = jellyfish.jaro_winkler(new_arg1.upper(), r.ent1.upper())
                            if score >= 0.9:
                                matches.append(r)
                                all_in_freebase[(r.ent1, r.ent2)] = "Found"
                                found = True

            # if a direct string matching occur with arg_1, check for a direct string matching
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
                        jaccardi = float(len(set_1.intersection(set_2))) / float(len(set_1.union(set_2)))
                        if jaccardi >= 0.5:
                            matches.append(r)
                            all_in_freebase[(r.ent1, r.ent2)] = "Found"
                            found = True

                        # Jaro Winkler
                        elif jaccardi <= 0.5:
                            score = jellyfish.jaro_winkler(new_arg2.upper(), r.ent2.upper())
                            if score >= 0.9:
                                matches.append(r)
                                all_in_freebase[(r.ent1, r.ent2)] = "Found"
                                found = True

            if found is False:
                no_matches.append(r)
                if PRINT_NOT_FOUND is True:
                    print r.ent1, '\t', r.ent2

            if queue.empty() is True:
                break

        except Queue.Empty:
            break


def proximity_pmi_a(e1_type, e2_type, queue, index, results, not_found):
    """
    sentences with tagged entities are indexed in whoosh
    perform the following query
    ent1 NEAR:X r NEAR:X ent2
    X is the maximum number of words between the query elements.
    """
    idx = open_dir(index)
    count = 0
    q_limit = 500
    with idx.searcher() as searcher:
        while True:
            try:
                r = queue.get_nowait()
                count += 1
                if count % 50 == 0:
                    print multiprocessing.current_process(), "To Process", queue.qsize(), "Correct found:", len(results)

                # if its not in the database calculate the PMI
                entity1 = "<"+e1_type+">"+r.ent1+"</"+e1_type+">"
                entity2 = "<"+e2_type+">"+r.ent2+"</"+e2_type+">"
                t1 = query.Term('sentence', entity1)
                t3 = query.Term('sentence', entity2)

                # First count the proximity (MAX_TOKENS_AWAY) occurrences of entities r.e1 and r.e2
                q1 = spans.SpanNear2([t1, t3], slop=MAX_TOKENS_AWAY, ordered=True, mindist=1)
                hits = searcher.search(q1, limit=q_limit)
                # TODO: maybe use other contexts for evaluation: rel.bef_words, rel.bet_words, rel.aft_words
                rel_words = [word for word in PunktWordTokenizer().tokenize(r.bet_words) if word
                             not in stopwords.words('english')]
                rel_words = set(rel_words)

                """
                print q1
                print "hits", len(hits)
                print r.ent1, '\t', r.ent2
                print "\n"
                print rel_words
                """

                # Using all the hits above from the query above, count how many have between the entities
                # the relational word(s) r.bef, r.bet, r.aft. That is the same word(s) as extracted by the system
                #
                # If there more hits for the two entities containing the same relational word(s) as extracted by a
                # system than any other words, we consider this extraction positive.
                hits_with_r = 0
                for s in hits:
                    sentence = s.get("sentence")
                    s = Sentence(sentence, e1_type, e2_type, MAX_TOKENS_AWAY, MIN_TOKENS_AWAY, CONTEXT_WINDOW)
                    for s_r in s.relationships:
                        if r.ent1.decode("utf8") == s_r.ent1 and r.ent2.decode("utf8") == s_r.ent2:
                            for rel in rel_words:
                                if rel in s_r.between:
                                    hits_with_r += 1
                                    break

                    if not len(hits) >= hits_with_r:
                        print "ERROR!"
                        print "hits", len(hits)
                        print "hits_with_r", hits_with_r
                        print entity1, '\t', entity2
                        print "\n"
                        sys.exit(0)

                if len(hits) > 0:
                    pmi = float(hits_with_r) / float(len(hits))
                    if pmi >= PMI:
                        results.append(r)
                        """
                        if isinstance(r, ExtractedFact):
                            print r.ent1, '\t', r.patterns, '\t', r.ent2, pmi
                        elif isinstance(r, Relationship):
                            print r.ent1, '\t', r.between, '\t', r.ent2, pmi
                        """
                    else:
                        not_found.append((r, pmi))
                else:
                    not_found.append((r, None))

                if queue.empty() is True:
                    break

            except Queue.Empty:
                break


def main():
    # "Automatic Evaluation of Relation Extraction Systems on Large-scale"
    # https://akbcwekex2012.files.wordpress.com/2012/05/8_paper.pdf
    #
    # S  - system output
    # D  - database (freebase)
    # G  - will be the resulting ground truth
    # G' - superset, contains true facts, and wrong facts
    #
    # a  - contains correct facts from the system output
    # b  - intersection between the system output and the database (i.e., freebase),
    #      it is assumed that every fact in this region is correct
    # c  - contains the database facts described in the corpus but not extracted by the system
    # d  - contains the facts described in the corpus that are not in the system output nor in the database
    #
    # Precision = |a|+|b| / |S|
    # Recall    = |a|+|b| / |a| + |b| + |c| + |d|
    # F1        = 2*P*R / P+R

    if len(sys.argv) == 1:
        print "No arguments"
        print "Use: evaluation.py threshold system_output rel_type database"
        print "\n"
        sys.exit(0)

    threhsold = float(sys.argv[1])
    rel_type = sys.argv[3]

    # load relationships extracted by the system
    system_output = process_output(sys.argv[2], threhsold, rel_type)
    print "Relationships score threshold :", threhsold
    print "System output relationships   :", len(system_output)

    # load freebase relationships as the database
    database_1, database_2, database_3 = process_freebase(sys.argv[4], rel_type)
    print "Freebase relationships loaded :", len(database_1.keys())

    # corpus from which the system extracted relationships
    #corpus = "/home/dsbatista/gigaword/automatic-evaluation/corpus.txt"
    corpus = "/home/dsbatista/gigaword/automatic-evaluation/sentences_matched_freebase.txt"

    # index to be used to estimate proximity PMI
    #index = "/home/dsbatista/gigaword/automatic-evaluation/index_2005_2010/"
    # index = "/home/dsbatista/gigaword/automatic-evaluation/index_2000_2010/"
    index = "/home/dsbatista/gigaword/automatic-evaluation/index_full"

    # entities semantic type
    if rel_type == 'founded' or rel_type == 'employment':
        e1_type = "ORG"
        e2_type = "PER"
    elif rel_type == 'acquired':
        e1_type = "ORG"
        e2_type = "ORG"
    elif rel_type == 'headquarters':
        e1_type = "ORG"
        e2_type = "LOC"
        # load dbpedia relationships
        load_dbpedia(sys.argv[5], database_1, database_2)
        print "Total relationships loaded    :", len(database_1.keys())

    elif rel_type == 'contained_by':
        e1_type = "LOC"
        e2_type = "LOC"
    else:
        print "Invalid relationship type", rel_type
        print "Use: founded, acquired, headquarters, contained_by"
        sys.exit(0)

    print "\nRelationship Type:", rel_type
    print "Arg1 Type:", e1_type
    print "Arg2 Type:", e2_type

    print "\nCalculating set B: intersection between system output and database"
    b, not_in_database = calculate_b(system_output, database_1, database_2, database_3, e1_type, e2_type)

    print "System output      :", len(system_output)
    print "Found in database  :", len(b)
    print "Not found          :", len(not_in_database)
    assert len(system_output) == len(not_in_database) + len(b)

    print "\nCalculating set A: correct facts from system output not in the database (proximity PMI)"
    a, not_found = calculate_a(not_in_database, e1_type, e2_type, index)
    print "System output      :", len(system_output)
    print "Found in database  :", len(b)
    print "Correct in corpus  :", len(a)
    print "Not found          :", len(not_found)
    print "\n"
    assert len(system_output) == len(a) + len(b) + len(not_found)

    if PRINT_NOT_FOUND is True:
        for r in sorted(set(not_found)):
            print r.ent1, '\t', r.patterns, '\t', r.ent2

    # Estimate G \intersected D = |b| + |c|, looking for relationships in G' that match a relationship in D
    # once we have G \in D and |b|, |c| can be derived by: |c| = |G \in D| - |b|
    #  G' = superset of G, cartesian product of all possible entities and relations (i.e., G' = E x R x E)
    print "\nCalculating set C: database facts in the corpus but not extracted by the system"
    c, superset = calculate_c(corpus, database_1, database_2, database_3, b, e1_type, e2_type, rel_type)
    assert len(c) > 0

    uniq_c = set()
    for r in c:
        uniq_c.add((r.ent1, r.ent2))

    # By applying the PMI of the facts not in the database (i.e., G' \in D)
    # we determine |G \ D|, then we can estimate |d| = |G \ D| - |a|
    print "\nCalculating set D: facts described in the corpus not in the system output nor in the database"
    d = calculate_d(superset, a, e1_type, e2_type, index, rel_type)
    print "System output      :", len(system_output)
    print "Found in database  :", len(b)
    print "Correct in corpus  :", len(a)
    print "Not found          :", len(not_found)
    print "\n"
    assert len(d) > 0

    uniq_d = set()
    for r in d:
        uniq_d.add((r.ent1, r.ent2))

    print "|a| =", len(a)
    print "|b| =", len(b)
    print "|c| =", len(c), "(", len(uniq_c), ")"
    print "|d| =", len(d), "(", len(uniq_d), ")"
    print "|S| =", len(system_output)
    print "Relationships not evaluated", len(set(not_found))

    # Write incorrect (i.e. not found) relationship sentences to disk
    f = open(rel_type+"_not_found.txt", "w")
    for r in set(not_found):
        f.write(r[0].ent1+'\t'+r[0].patterns+'\t'+r[0].ent2+'\t'+str(r[1])+'\n')
    f.close()

    # Write all correct relationships (sentence, entities and score) to file
    f = open(rel_type+"_correct_extractions.txt", "w")
    for r in set(a).union(b):
        f.write('instance:\t'+r.ent1+'\t'+r.patterns+'\t'+r.ent2+'\t'+r.score+'\n')
        f.write('pattern:' + r.patterns+'\n')
        f.write('sentence:' + r.sentence+'\n')
        f.write('\n')
    f.close()

    a = set(a)
    b = set(b)
    output = set(system_output)
    precision = float(len(a) + len(b)) / float(len(output))
    recall = float(len(a) + len(b)) / float(len(a) + len(b) + len(uniq_c) + len(uniq_d))
    f1 = 2*(precision*recall)/(precision+recall)

    print "\nPrecision: ", precision
    print "Recall   : ", recall
    print "F1   : ", f1
    print "\n"

if __name__ == "__main__":
    main()