#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fileinput
import functools
import multiprocessing
import re
import time
import sys
import cPickle
import itertools
import jellyfish

from whoosh.index import open_dir, os
from whoosh.query import spans
from whoosh import query
from Snowball.Sentence import Sentence
from Snowball.Sentence import Relationship
from collections import defaultdict

# relational words to be used in calculating the set D with the proximity PMI
founder = ['founder', 'co-founder', 'cofounder', 'founded by', 'started by']
acquired = ['bought', 'shares', 'holds', 'owns', 'acquired']
headquarters = ['headquarters', 'compund', 'offices']
contained_by = ['capital', 'located']

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
all_in_freebase = manager.dict()


class ExtractedFact(object):
    def __init__(self, _e1, _e2, _score, _patterns, _sentence):
        self.ent1 = _e1
        self.ent2 = _e2
        self.score = _score
        self.patterns = _patterns
        self.sentence = _sentence

    def __hash__(self):
        return hash(self.ent1) ^ hash(self.ent2) ^ hash(self.patterns) ^ hash(self.score) ^ hash(self.sentence)

    def __eq__(self, other):
        if self.ent1 == other.ent1 and self.ent2 == other.ent2 and self.score == other.score and self.patterns == other.patterns and self.sentence == other.sentence:
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
        print "%s %.2f seconds" % (f.__name__, end - start)
        return result
    return wrapper


def is_acronym(entity):
    if len(entity.split()) == 1 and entity.isupper():
        return True
    else:
        return False


def process_corpus(queue, g_dash, e1_type, e2_type):
    while True:
        line = queue.get_nowait()
        s = Sentence(line.strip(), e1_type, e2_type, MAX_TOKENS_AWAY, MIN_TOKENS_AWAY, CONTEXT_WINDOW)
        for r in s.relationships:
            if r.between == " , " or r.between == " ( " or r.between == " ) ":
                continue
            else:
                g_dash.append(r)
        if queue.empty() is True:
            break


def process_output(data):
    """
    parses the file with the relationships extracted by the system
    each relationship is transformed into a ExtracteFact class
    :param data:
    :return:
    """

    system_output = list()
    for line in fileinput.input(data):

        if line.startswith('instance'):
            instance_parts, score = line.split("score:")
            e1, e2 = instance_parts.split("instance:")[1].strip().split('\t')

        if line.startswith('sentence'):
            sentence = line.split("sentence:")[1].strip()

        if line.startswith('pattern'):
            patterns = line.split("pattern:")[1].strip()

        if line.startswith('\n'):
            r = ExtractedFact(e1, e2, score, patterns, sentence)
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
        # for founder don't load those from freebase, lists countries as founders not persons
        if e2.strip() in founder_to_ignore:
            continue
        else:
            if "(" in e1:
                e1 = re.sub(r"\(.*\)", "", e1).strip()
            if "(" in e2:
                e2 = re.sub(r"\(.*\)", "", e2).strip()

            if rel_type == 'founder':
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
    return acronyms
    fileinput.close()


#########################################
# Estimations of sets and intersections #
#########################################

@timecall
def calculate_a(output, e1_type, e2_type, index):
    m = multiprocessing.Manager()
    queue = m.Queue()
    num_cpus = multiprocessing.cpu_count()
    results = [m.list() for _ in range(num_cpus)]

    # put output in a processed shared queue
    for r in output:
        queue.put(r)

    processes = [multiprocessing.Process(target=proximity_pmi, args=(e1_type, e2_type, queue, index, results[i])) for i in range(num_cpus)]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()

    a = list()
    for l in results:
        a.extend(l)
    return a


@timecall
def calculate_b(output, database_1, database_2, database_3, acronyms):
    # intersection between the system output and the database (i.e., freebase),
    # it is assumed that every fact in this region is correct
    m = multiprocessing.Manager()
    queue = m.Queue()
    num_cpus = multiprocessing.cpu_count()
    results = [m.list() for _ in range(num_cpus)]
    no_matches = [m.list() for _ in range(num_cpus)]

    # passar tudo para a queue
    for r in output:
        queue.put(r)

    processes = [multiprocessing.Process(target=string_matching_parallel, args=(acronyms, results[i], no_matches[i], database_1, database_2, database_3, queue)) for i in range(num_cpus)]

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
def calculate_c(corpus, acronyms, database_1, database_2, database_3, b, e1_type, e2_type, rel_type):
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

        processes = [multiprocessing.Process(target=process_corpus, args=(queue, g_dash, e1_type, e2_type)) for _ in range(num_cpus)]
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
        print "Freebase:", len(database_1.keys())

        queue = manager.Queue()
        results = [manager.list() for _ in range(num_cpus)]
        no_matches = [manager.list() for _ in range(num_cpus)]

        # passar tudo para a queue
        for r in g_dash_set:
            queue.put(r)

        processes = [multiprocessing.Process(target=string_matching_parallel, args=(acronyms, results[i], no_matches[i], database_1, database_2, database_3, queue)) for i in range(num_cpus)]

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
    # |G \ D|
    # determine facts not in the database, with high PMI, that is, facts that are true and are not in the database

    m = multiprocessing.Manager()
    queue = m.Queue()
    num_cpus = multiprocessing.cpu_count()
    results = [m.list() for _ in range(num_cpus)]

    print "Storing g_dash in a shared Queue"
    for r in g_dash:
        queue.put(r)
    print "queue size", queue.qsize()

    if rel_type == "founder":
        rel_words = founder
    elif rel_type == "acquired":
        rel_words = acquired
    elif rel_type == 'headquarters':
        rel_words = headquarters
    elif rel_type == "contained_by":
        rel_words = contained_by

    # calculate PMI for r not in database
    processes = [multiprocessing.Process(target=proximity_pmi_rel_word, args=(e1_type, e2_type, queue, index, results[i], rel_words)) for i in range(num_cpus)]
    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()

    g_minus_d = set()
    for l in results:
        g_minus_d.update(l)

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
    :param rel_word:
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
    distance = 9
    q_limit = 500
    with idx.searcher() as searcher:
        while True:
            r = queue.get_nowait()
            count += 1
            if count % 500 == 0:
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

                if float(len(hits)) > 0:
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


def proximity_pmi(e1_type, e2_type, queue, index, results):
    """
    sentences with tagged entities are indexed in whoosh
    perform the following query
    ent1 NEAR:X r NEAR:X ent2
    X is the maximum number of words between the query elements.
    """
    tokenize = re.compile('\w+(?:-\w+)+|<[A-Z]+>[^<]+</[A-Z]+>|\w+', re.U)
    entity = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)
    idx = open_dir(index)
    count = 0
    distance = 9
    q_limit = 500
    with idx.searcher() as searcher:
        while True:
            count += 1
            if count % 50 == 0:
                print multiprocessing.current_process(), "In Queue", queue.qsize(), "Total Matched: ", len(results)
            r = queue.get_nowait()
            n_1 = set()
            n_2 = set()
            n_3 = set()
            # if its not in the database calculate the proximity PMI
            if (r.ent1, r.ent2) not in all_in_freebase:
                entity1 = "<"+e1_type+">"+r.ent1+"</"+e1_type+">"
                entity2 = "<"+e2_type+">"+r.ent2+"</"+e2_type+">"
                t1 = query.Term('sentence', entity1)
                t3 = query.Term('sentence', entity2)

                # Entities proximity query without relational words
                q1 = spans.SpanNear2([t1, t3], slop=distance, ordered=True, mindist=1)
                hits_1 = searcher.search(q1, limit=q_limit)

                # Entities proximity considering relational words
                if isinstance(r, ExtractedFact):
                    tokens_rel = re.findall(tokenize, r.patterns)

                elif isinstance(r, Relationship):
                    tokens_rel = re.findall(tokenize, r.between)

                token_terms = list()
                for t in tokens_rel:
                    if re.search(entity, t) is None:
                        token_terms.append(query.Term('sentence', t))

                l1 = [t for t in token_terms]
                l1.insert(0, t1)
                l2 = [t for t in token_terms]
                l2.append(t3)

                q2 = spans.SpanNear2(l1, slop=distance-1, ordered=True, mindist=1)
                hits_2 = searcher.search(q2, limit=q_limit)

                q3 = spans.SpanNear2(l2, slop=distance-1, ordered=True, mindist=1)
                hits_3 = searcher.search(q3, limit=q_limit)

                for d in hits_1:
                    n_1.add(d.get("sentence"))

                for d in hits_2:
                    n_2.add(d.get("sentence"))

                for d in hits_3:
                    n_3.add(d.get("sentence"))

                entities_occurr = len(hits_1)
                entities_occurr_with_r = len(n_1.intersection(n_2).intersection(n_3))

                try:
                    assert not entities_occurr_with_r > entities_occurr
                except AssertionError, e:
                    print e
                    print r.sentence
                    print r.ent1
                    print r.ent2
                    print q1, len(hits_1)
                    print q2, len(hits_2)
                    print q3, len(hits_3)
                    print "intersection", len(n_1.intersection(n_2).intersection(n_3))
                    sys.exit(0)

                if float(entities_occurr) > 0:
                    if float(entities_occurr) > 1 and entities_occurr_with_r > 1:
                        pmi = float(entities_occurr_with_r) / float(entities_occurr)
                        if pmi > PMI:
                            """
                            # TODO: há coisas aqui que sao falsas, por exemplo: 'chief'
                            if isinstance(r, ExtractedFact):
                                print r.ent1, '\t', r.patterns, '\t', r.ent2, pmi
                            elif isinstance(r, Relationship):
                                print r.ent1, '\t', r.between, '\t', r.ent2, pmi
                            """
                            results.append(r)

                if queue.empty() is True:
                    break


def string_matching_parallel(acronyms, matches, no_matches, database_1, database_2, database_3, queue):
    count = 0
    while True:
        r = queue.get_nowait()
        found = False
        count += 1
        if count % 500 == 0:
            print multiprocessing.current_process(), "In Queue", queue.qsize()

        # check if its in cache, i.e., if tuple was already matched
        if (r.ent1, r.ent2) in all_in_freebase:
            matches.append(r)
            found = True

        if found is False:
            # check both entities for a direct string matching
            if len(database_1[(r.ent1.decode("utf8"), r.ent2.decode("utf8"))]) > 0:
                matches.append(r)
                all_in_freebase[(r.ent1, r.ent2)] = "Found"
                found = True

        if found is False:
            # if e1 and e2 are both acronyms
            if is_acronym(r.ent1) and is_acronym(r.ent2):
                expansions_e1 = acronyms.get(r.ent1)
                expansions_e2 = acronyms.get(r.ent2)
                if expansions_e1 is not None and expansions_e2 is not None:
                    for i in itertools.product(expansions_e1, expansions_e2):
                        e1 = i[0]
                        e2 = i[1]
                        if e2 in database_2[e1]:
                            matches.append(r)
                            all_in_freebase[(r.ent1, r.ent2)] = "Found"
                            found = True

        if found is False:
            # if e1 is acronym
            if is_acronym(r.ent1) and not is_acronym(r.ent2):
                expansions = acronyms.get(r.ent1)
                if expansions is not None:
                    for e in expansions:
                        if r.ent2 in database_2[e]:
                            matches.append(r)
                            all_in_freebase[(r.ent1, r.ent2)] = "Found"
                            found = True

        if found is False:
            # if e2 is acronym
            if is_acronym(r.ent2) and not is_acronym(r.ent1):
                expansions = acronyms.get(r.ent2)
                if expansions is not None:
                    for e in expansions:
                        if r.ent1 in database_3[e]:
                            matches.append(r)
                            all_in_freebase[(r.ent1, r.ent2)] = "Found"
                            found = True

        # FOUNDER   : r.ent1:ORG   r.ent2:PER
        # DATABASE_1: (ORG,PER)
        # DATABASE_2: ORG   list<PER>
        # DATABASE_3: PER   list<ORG>

        # direct matching with person name
        if found is False:
            organisations = database_3[r.ent2]
            if organisations is not None:
                for o in organisations:
                    new_o = re.sub(r" Corporation| Inc\.", "", o)
                    # person name matched 100% check if organisation match with jaccardi
                    set_1 = set(new_o.split())
                    set_2 = set(r.ent1.split())
                    jaccardi = float(len(set_1.intersection(set_2))) / float(len(set_1.union(set_2)))
                    if jaccardi >= 0.5:
                        #print r.ent1, '\t', database_3[r.ent2], "MATCHED"
                        """
                        print "Set1", set(new_o.split())
                        print "Set2", set(r.ent1.split())
                        print float(len(set_1.intersection(set_2))) / float(len(set_1.union(set_2)))
                        print "\n"
                        """
                        matches.append(r)
                        all_in_freebase[(r.ent1, r.ent2)] = "Found"
                        found = True

                    else:
                        score = jellyfish.jaro_winkler(new_o.upper(), r.ent1.upper())
                        if score >= 0.9:
                            matches.append(r)
                            all_in_freebase[(r.ent1, r.ent2)] = "Found"
                            found = True

        # direct matching with organisation name
        if found is False:
            names = database_2[r.ent1]
            if names is not None:
                for n in names:
                    # organisation name matched 100% check if names match with jaccardi
                    set_1 = set(n.split())
                    set_2 = set(r.ent2.split())
                    jaccardi = float(len(set_1.intersection(set_2))) / float(len(set_1.union(set_2)))
                    if jaccardi >= 0.5:
                        matches.append(r)
                        all_in_freebase[(r.ent1, r.ent2)] = "Found"
                        found = True
                    else:
                        score = jellyfish.jaro_winkler(n.upper(), r.ent2.upper())
                        if score >= 0.9:
                            matches.append(r)
                            all_in_freebase[(r.ent1, r.ent2)] = "Found"
                            found = True

        # approximate string similarity
        # TODO: usar similaridade de Jaccardi aqui também
        if found is False:
            for k in database_2.keys():
                # remove 'Corporation' and 'Inc.' from companies names to ease the string machting
                new_k = re.sub(r" Corporation| Inc\.", "", k)
                score_1 = jellyfish.jaro_winkler(new_k.upper(), r.ent1.upper())
                if score_1 >= 0.9:
                    for e1 in database_2[k]:
                        score_2 = jellyfish.jaro_winkler(e1.upper(), r.ent2.upper())
                        if score_2 >= 0.9:
                            matches.append(r)
                            all_in_freebase[(r.ent1, r.ent2)] = "Found"
                            found = True
                            break

                        else:
                            pass
                            """
                            print k, r.ent1, score_1
                            # if company matched with high score, try to match name
                            print e1, '\t', r.ent2, score_2
                            # TODO: look for how many common names, jaccardi
                            print "\n"
                            """

                    if found is True:
                        break

        if found is False:
            #TODO: "FARC Revolutionary Armed Forces of Colombia" -> "FARC"
            #TODO: cache para o que não fez match
            no_matches.append(r)
            print r.ent1, '\t', r.ent2

        if queue.empty() is True:
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
        print "Use: evaluation.py system_output database corpus index acronyms rel_type"
        print "\n"
        sys.exit(0)

    # load relationships extracted by the system
    system_output = process_output(sys.argv[1])
    print len(system_output), "system output relationships loaded"

    # relationship type
    rel_type = sys.argv[6]

    # load freebase relationships as the database
    database_1, database_2, database_3 = process_freebase(sys.argv[2], rel_type)
    print len(database_1.keys()), "freebase relationships loaded"

    # corpus from which the system extracted relationships
    corpus = sys.argv[3]

    # index to be used to estimate proximity PMI
    index = sys.argv[4]

    acronyms = load_acronyms(sys.argv[5])
    print len(acronyms), "acronyms loaded"

    # entities semantic type
    if rel_type == 'founder':
        e1_type = "ORG"
        e2_type = "PER"
    elif rel_type == 'acquired':
        e1_type = "ORG"
        e2_type = "ORG"
    elif rel_type == 'headquarters':
        e1_type = "ORG"
        e2_type = "LOC"
    elif rel_type == 'contained_by':
        e1_type = "LOC"
        e2_type = "LOC"
    else:
        print "Invalid relationship type", rel_type
        print "Use: founder, acquired, headquarters, contained_by"
        sys.exit(0)

    print "\nRelationship Type:", rel_type
    print "Arg1 Type:", e1_type
    print "Arg2 Type:", e2_type

    print "\nCalculating set B: intersection between system output and database"
    b, not_in_database = calculate_b(system_output, database_1, database_2, database_3, acronyms)
    assert len(b) > 0
    assert len(system_output) == len(not_in_database) + len(b)
    print "\nTotal output", len(system_output)
    print "Found in Freebase", len(b)
    print "Not in Freebase", len(not_in_database)
    print "\n"

    print "\nCalculation set A: correct facts from system output not in the database (proximity PMI)"
    a = calculate_a(not_in_database, e1_type, e2_type, index)
    assert len(a) > 0

    print "Total output", len(system_output)
    print "Found in Freebase", len(b)
    print "Not in Freebase", len(not_in_database)
    print "Found in Corpus", len(a)
    print "Not Found", len(not_in_database)-len(a)
    print "\n"
    not_found = list()
    for r in not_in_database:
        if r not in a:
            not_found.append(r)

    if PRINT_NOT_FOUND is True:
        for r in sorted(set(not_found)):
            print r.ent1, '\t', r.patterns, '\t', r.ent2

    # Estimate G \intersected D = |b| + |c|, looking for relationships in G' that match a relationship in D
    # once we have G \in D and |b|, |c| can be derived by: |c| = |G \in D| - |b|
    #  G' = superset of G, cartesian product of all possible entities and relations (i.e., G' = E x R x E)
    print "\nCalculating set C: database facts in the corpus but not extracted by the system"
    c, superset = calculate_c(corpus, acronyms, database_1, database_2, database_3, b, e1_type, e2_type, rel_type)
    assert len(c) > 0

    # By applying the PMI of the facts not in the database (i.e., G' \in D)
    # we determine |G \ D|, then we can estimate |d| = |G \ D| - |a|
    print "\nCalculating set D: facts described in the corpus not in the system output nor in the database"
    d = calculate_d(superset, a, e1_type, e2_type, index, rel_type)
    assert len(d) > 0

    print "|a| =", len(a)
    print "|b| =", len(b)
    print "|c| =", len(c)
    print "|d| =", len(d)
    print "|S| =", len(system_output)
    print "Relationships not evaluated", len(set(not_found))

    f = open(rel_type+"_not_found.txt", "w")
    for r in set(not_found):
        f.write(r.ent1+'\t'+r.patterns+'\t'+r.ent2+'\n')
    f.close()

    a = set(a)
    b = set(b)
    output = set(system_output)

    precision = float(len(a) + len(b)) / float(len(output))
    recall = float(len(a) + len(b)) / float(len(a) + len(b) + len(c) + len(d))
    f1 = 2*(precision*recall)/(precision+recall)
    print "\nPrecision: ", precision
    print "Recall   : ", recall
    print "F1   : ", f1
    print "\n"

if __name__ == "__main__":
    main()

