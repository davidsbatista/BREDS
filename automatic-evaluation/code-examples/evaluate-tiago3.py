#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fileinput
import functools
import multiprocessing
import re
import time
import sys
import _multiprocessing
import easy_freebase_clean

from whoosh.analysis import RegexTokenizer, StopFilter
from whoosh.index import open_dir
from whoosh.query import spans
from whoosh import query
from Sentence_old import Sentence


class ExtractedFact(object):

    def __init__(self, _e1, _e2, _score, _patterns, _sentence):
        self.e1 = _e1
        self.e2 = _e2
        self.score = _score
        self.patterns = _patterns
        self.sentence = _sentence


def timecall(f):
    @functools.wraps(f)
    def wrapper(*args, **kw):
        start = time.time()
        result = f(*args, **kw)
        end = time.time()
        print "%s %.2f seconds" % (f.__name__, end - start)
        return result
    return wrapper


POOL = multiprocessing.Pool()


def map_reduce(proc_elems, elems, pool=POOL):
    result = set()
    for r in pool.imap_unordered(proc_elems, elems):
        if r is not None:
            result.add(r)
    return result


@timecall
def calculate_a(output, e1_type, e2_type, index):
    # NOTE: this queries an index build over the whole AFP corpus
    """
    # sentences with tagged entities are indexed in whoosh, perform the following query
    # ent1 NEAR:X r NEAR:X ent2
    # X is the maximum number of words between the query elements.
    """
    return map_reduce(lambda r: query_a(r, e1_type, e2_type, index), output)


def query_a(r, e1_type, e2_type, index):
    idx = open_dir(index)
    entity1 = "<"+e1_type+">"+r.e1+"</"+e1_type+">"
    entity2 = "<"+e2_type+">"+r.e2+"</"+e2_type+">"
    t1 = query.Term("sentence", entity1)
    t2 = query.Term("sentence", r.patterns)
    t3 = query.Term("sentence", entity2)
    q1 = spans.SpanNear2([t1, t2, t3], slop=5, ordered=True)
    q2 = spans.SpanNear2([t1, t3], slop=5, ordered=True)

    with idx.searcher() as searcher:
        entities_r = searcher.search(q1)
        entities = searcher.search(q2)
        # TODO: fazer stemming ou normalização da palavra a usar no query
        if len(entities) > 0:
            pmi = float(len(entities_r)) / float(len(entities))
            # TODO: qual o melhor valor de threshold ?
            if pmi >= 0.5:
                #print entity1, '\t', r.patterns, '\t', entity2, pmi
                return r


@timecall
def calculate_b(output, database):
    # intersection between the system output and the database (i.e., freebase),
    # it is assumed that every fact in this region is correct
    # relationships in database are in the form of
    # (e1,e2) is a tuple
    # database is a dictionary of lists
    # each key is a tuple (e1,e2), and each value is a list containing the relationships
    # between the e1 and e2 entities

    b = set()
    for system_r in output:
        for k in database.keys():
            """
            Freebase represents the 'founder-of' relationship has:
            PER 'Organization founder' ORG
            The system extracted in a different order: ORG founded-by PER
            Swap the entities order, in comparision
            """
            # TODO: usar string matching entre as entidades: https://www.cs.cmu.edu/~pradeepr/papers/ijcai03.pdf
            # ver biblioteca de python jellyfish
            # TODO: usar a expansão de acrónimos, usar apenas acrónimos da lista não âmbiguos
            if system_r.e1.decode("utf8") == k[1].decode("utf8") and system_r.e2.decode("utf8") == k[0].decode("utf8"):
                if len(database[(k[0].encode("utf8"), k[1].encode("utf8"))]) == 1:
                    b.add(system_r)
                else:
                    for r in database[(k[0].encode("utf8"), k[1].encode("utf8"))]:
                        print r
    return b


@timecall
def calculate_c(corpus, database, b):
    # contains the database facts described in the corpus but not extracted by the system
    #
    # G' = superset of G, cartesian product of all possible entities and relations (i.e., G' = E x R x E)
    # for now, all relationships from a sentence
    print "Building G', a superset of G"
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    g_dash = manager.list()
    num_cpus = multiprocessing.cpu_count()

    with open(corpus) as f:
        print "Reading corpus into memory"
        data = f.readlines()
        print "Storing in shared Queue"
        for l in data:
            queue.put(l)

    processes = [multiprocessing.Process(target=process_corpus, args=(queue, g_dash)) for i in range(num_cpus)]
    print "Running", len(processes), "threads"

    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()

    # TODO: isto dá igual: implementar o __eq__ numa relationships
    print len(g_dash), "relationships built"
    g_dash_set = set(g_dash)
    print len(g_dash_set), "unique relationships"

    # estimate G \in D, look for facts in G' that a match a fact in the database
    print "Estimating G intersection with D"
    g_intersect_d = set()
    for r in g_dash_set:
        if len(database[(r.ent1, r.ent2)]) > 0:
            for relation in database[(r.ent1, r.ent2)]:
                # TODO: está hard-coded para relação: founder, para caso geral, aplicar uma medida de similaridade
                # entre a palvra da na frase e a relação do freebase
                if relation == 'Organization founded':
                    g_intersect_d.add(r)

    # having b and g_intersect_d => |c| = g_intersect_d - b
    # TODO: só para uma relação pode-se fazer um dump e evitar andar sempre a calcular
    # TODO: testar o difference
    c = g_intersect_d.difference(b)
    return c, g_dash_set


def process_corpus(queue, g_dash):
    while True:
        line = queue.get_nowait()
        s = Sentence(line.strip())
        for r in s.relationships:
            g_dash.append(r)
        if queue.empty() is True:
            break


@timecall
def calculate_d(g_dash, database, a, e1_type, e2_type, index):
    # contains facts described in the corpus that are not in the system output nor in the database
    #
    # by applying the PMI of the facts not in the database (i.e., G' \in D)
    # we determine |G \ D|, then we can estimate |d| = |G \ D| - |a|
    #
    # |G \ D|
    # determine facts not in the database, with high PMI, that is, facts that are true and are not in the database

    manager = multiprocessing.Manager()
    queue = manager.Queue()
    g_minus_d = manager.list()
    num_cpus = multiprocessing.cpu_count()

    print len(g_dash)
    c = 0
    print "Storing g_dash in a shared Queue"
    for r in g_dash:
        c += 1
        if c % 25000 == 0:
            print c
        queue.put(r)

    print "queue size", queue.qsize()

    # calculate PMI for r not in database
    processes = [multiprocessing.Process(target=query_thread, args=(queue, database, g_minus_d, e1_type, e2_type, index)) for i in range(num_cpus)]
    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()

    print "Relationships with high PMI", len(g_minus_d)
    g_minus_d_set = set(g_minus_d)
    return g_minus_d_set.difference(a)


def query_thread(queue, database, g_minus_d, e1_type, e2_type, index):
    idx = open_dir(index)
    regex_tokenize = re.compile('\w+|-|<[A-Z]+>[^<]+</[A-Z]+>', re.U)
    tokenizer = RegexTokenizer(regex_tokenize)
    stopper = StopFilter()
    count = 0

    with idx.searcher() as searcher:
        while True:
            r = queue.get_nowait()
            count += 1
            if count % 25000 == 0:
                print multiprocessing.current_process(), count, queue.qsize()

            if len(database[(r.ent1, r.ent2)]) == 0:
                # if its not in the database calculate the PMI
                entity1 = "<"+e1_type+">"+r.ent1+"</"+e1_type+">"
                entity2 = "<"+e2_type+">"+r.ent2+"</"+e2_type+">"
                terms = list()
                for token in stopper(tokenizer((r.between.decode("utf8")), renumber=True)):
                    terms.append(query.Term("sentence", token.text))

                #print terms
                t1 = query.Term("sentence", entity1)
                t3 = query.Term("sentence", entity2)

                query_terms = list()
                query_terms.append(t1)
                for t in terms:
                    query_terms.append(t)
                query_terms.append(t3)

                q1 = spans.SpanNear2(query_terms, slop=2, ordered=True)
                q2 = spans.SpanNear2([t1, t3], slop=8, ordered=True)
                entities_r = searcher.search(q1)
                entities = searcher.search(q2)

                """
                print query_terms, len(entities_r)
                print [t1, t3], len(entities)
                print "\n"
                """

                #print entity1, '\t', r.between, '\t', entity2, len(entities_r), len(entities)

                try:
                    assert not len(entities_r) > len(entities)
                except AssertionError, e:
                    print e
                    print r.sentence
                    print r.ent1
                    print r.ent2
                    print query_terms
                    print [t1, t3]

                if len(entities) > 0:
                    pmi = float(len(entities_r)) / float(len(entities))
                    if pmi >= 0.5:
                        #print entity1, '\t', r.between, '\t', entity2, pmi
                        g_minus_d.append(r)

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


def main():
    # Implements the paper: "Automatic Evaluation of Relation Extraction Systems on Large-scale"
    # https://akbcwekex2012.files.wordpress.com/2012/05/8_paper.pdf
    #
    # S  - system output
    # D  - database (freebase)
    # G  - will be the resulting ground truth
    # G' - superset, contains true facts, and wrong facts
    #
    # a - contains correct facts from the system output
    # b - intersection between the system output and the database (i.e., freebase),
    #     it is assumed that every fact in this region is correct
    # c - contains the database facts described in the corpus but not extracted by the system
    # d - contains the facts described in the corpus that are not in the system output nor in the database
    #
    # Precision = |a|+|b| / |S|
    # Recall = |a|+|b| / |a| + |b| + |c| + |d|
    # F1 = 2*P*R / P+R

    # load relationships extracted by the system
    system_output = process_output(sys.argv[1])
    print len(system_output), "system output relationships loaded"

    # load freebase relationships as the database
    database = easy_freebase_clean.collect_relationships(sys.argv[2], 'Organization founded')
    print len(database.keys()), "freebase relationships loaded"

    # corpus from which the system extracted relationships
    corpus = sys.argv[3]

    # index to be used to estimate proximity PMI
    index = sys.argv[4]

    e1_type = "ORG"
    e2_type = "PER"

    print "\nCalculation set A: correct facts from system output not in the database (proximity PMI)"
    a = calculate_a(system_output, e1_type, e2_type, index)
    assert len(a) > 0

    print "\nCalculating set B: intersection between system output and database (direct string matching)"
    b = calculate_b(system_output, database)
    assert len(b) > 0

    # Estimate G \intersected D = |b| + |c|, looking for relationships in G' that match a relationship in D
    # once we have G \in D and |b|, |c| can be derived by: |c| = |G \in D| - |b|
    #  G' = superset of G, cartesian product of all possible entities and relations (i.e., G' = E x R x E)
    print "\nCalculating set C: database facts in the corpus but not extracted by the system"
    c, g_dash = calculate_c(corpus, database, b)
    assert len(c) > 0

    # By applying the PMI of the facts not in the database (i.e., G' \in D)
    # we determine |G \ D|, then we can estimate |d| = |G \ D| - |a|
    print "\nCalculating set D: facts described in the corpus not in the system output nor in the database"
    d = calculate_d(g_dash, database, a, e1_type, e2_type, index)
    assert len(d) > 0

    print "|a| =", len(a)
    print "|b| =", len(b)
    print "|c| =", len(c)
    print "|d| =", len(d)

    precision = float(len(a) + len(b)) / float(len(system_output))
    recall = float(len(a) + len(b)) / float(len(a) + len(b) + len(c) + len(d))
    f1 = 2*(precision*recall)/(precision+recall)
    print "\nPrecision: ", precision
    print "Recall   : ", recall
    print "F1   : ", f1
    print "\n"

if __name__ == "__main__":
    main()
