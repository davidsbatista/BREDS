#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fileinput
import functools
import time
import sys
import freebase

from whoosh.index import open_dir
from whoosh.query import spans
from whoosh import query
from Sentence import Sentence


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


def calculate_a(output, e1_type, e2_type):
    # NOTE: this queries an index build over the whole AFT corpus
    """
    # sentences with tagged entities are indexed in whoosh, perform the following query
    # ent1 NEAR:X r NEAR:X ent2
    # X is the maximum number of words between the query elements.
    #
    """
    a = set()
    idx = open_dir("index")

    for r in output:
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
            #TODO: fazer stemming ou normalização da palavra
            """
            print entity1, '\t', r.patterns, '\t', entity2, len(entities_r)
            print entity1, '\t', entity2, len(entities)
            print len(entities_r), len(entities)
            print "\n"
            """
            if len(entities) > 0:
                pmi = float(len(entities_r)) / float(len(entities))
                # TODO: ver como calcular o threshold
                if pmi >= 0.5:
                    print entity1, '\t', r.patterns, '\t', entity2, pmi
                    a.add(r)

    idx.close()
    return a


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
        if len(database[(system_r.e1, system_r.e2)]) > 0:
            #print database[(system_r.ent1, system_r.ent2)]
            for relation in database[(system_r.e1, system_r.e2)]:
                #print "Output  :", system_r.e1, '\t', system_r.e2, system_r.patterns
                #print "Freebase:", relation
                # TODO: aplicar uma medida de similaridade entre entre a palvra da na frase e a relação do freebase
                b.add(system_r)
    return b


def calculate_c(corpus, database, b):
    # contains the database facts described in the corpus but not extracted by the system
    #
    # G' = superset of G, cartesian product of all possible entities and relations (i.e., G' = E x R x E)
    # for now, all relationships from a sentence
    g_dash = set()
    for line in fileinput.input(corpus):
        s = Sentence(line.strip())
        for r in s.relationships:
            g_dash.add(r)
    fileinput.close()

    # estimate G \in D, look for facts in G' that a match a fact in the database
    g_intersect_d = set()
    for r in g_dash:
        if len(database[(r.ent1, r.ent2)]) > 0:
            for relation in database[(r.ent1, r.ent2)]:
                # TODO: aplicar uma medida de similaridade entre entre a palvra da na frase e a relação do freebase
                # TODO: está hard-coded para relação: founder
                if relation == 'Organization founded':
                    g_intersect_d.add(r)

    # having b and g_intersect_d => |c| = g_intersect_d - b
    # TODO: testar o difference
    c = g_intersect_d.difference(b)
    return c, g_dash


def calculate_d(g_dash, database, a, e1_type, e2_type):
    # contains facts described in the corpus that are not in the system output nor in the database
    #
    # by applying the PMI of the facts not in the database (i.e., G' \in D)
    # we determine |G \ D|, then we can estimate |d| = |G \ D| - |a|
    #
    # |G \ D|
    # determine facts not in the database, with high PMI, that is, facts that are true and are not in the database
    g_minus_d = set()
    idx = open_dir("index")
    with idx.searcher() as searcher:
        # precorrer o g_dash, e para cada par que não está na base de dados
        # calcular o PMI, se o PMI for alto, considerar um true fact
        print len(g_dash)
        c = 0
        cache = set()
        for r in g_dash:
            c += 1
            if c % 25000 == 0:
                print c, len(cache)

            if len(database[(r.ent1, r.ent2)]) == 0:
                # PMI
                entity1 = "<"+e1_type+">"+r.ent1+"</"+e1_type+">"
                entity2 = "<"+e2_type+">"+r.ent2+"</"+e2_type+">"
                t1 = query.Term("sentence", entity1)
                t2 = query.Term("sentence", r.between)  # TODO: remover stopwords do r.between
                t3 = query.Term("sentence", entity2)
                if (t1, t2, t3) not in cache:
                    q1 = spans.SpanNear2([t1, t2, t3], slop=5, ordered=True)
                    q2 = spans.SpanNear2([t1, t3], slop=5, ordered=True)
                    entities_r = searcher.search(q1)
                    entities = searcher.search(q2)
                    """
                    print entity1, '\t', r.patterns, '\t', entity2, len(entities_r)
                    print entity1, '\t', entity2, len(entities)
                    """
                    if len(entities) > 0:
                        pmi = float(len(entities_r)) / float(len(entities))
                        # TODO: ver como calcular o threshold
                        if pmi >= 0.5:
                            print entity1, '\t', r.patterns, '\t', entity2, len(entities_r)
                            g_minus_d.add(r)

                    cache.add((t1, t2, t3))

    return g_minus_d.difference(a)


def process_output(data):
    # process the relationships extracted by the system
    system_output = list()
    for line in fileinput.input(data):

        if line.startswith('instance'):
            instance_parts, score = line.split("score:")
            e1, e2 = instance_parts.split("instance:")[1].strip().split('\t')

        if line.startswith('sentence'):
            sentence = line.split("sentence:")[1].strip()

        if line.startswith('pattern'):
            patterns = line.split("pattern :")[1].strip()

        if line.startswith('\n'):
            r = ExtractedFact(e1, e2, score, patterns, sentence)
            system_output.append(r)

    fileinput.close()
    return system_output


def main():
    # S - system output
    # D - database (freebase)
    # G - will be the resulting ground truth
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
    database = freebase.collect_relationships(sys.argv[2], 'Organization founded')
    print len(database.keys()), "freebase relationships loaded"

    # corpus from which the system extracted relationships
    corpus = sys.argv[3]

    e1_type = "ORG"
    e2_type = "PER"

    print "\nCalculating |a| (proximity PMI)"
    a = calculate_a(system_output, e1_type, e2_type)

    print "Calculating |b|"
    b = calculate_b(system_output, database)

    # Estimate G \intersected D = |b| + |c|, looking for relationships in G' that match a relationship in D
    # once we have G \in D and |b|, |c| can be derived by: |c| = |G \in D| - |b|
    #  G' = superset of G, cartesian product of all possible entities and relations (i.e., G' = E x R x E)
    print "Calculating |c|"
    c, g_dash = calculate_c(corpus, database, b)

    # By applying the PMI of the facts not in the database (i.e., G' \in D)
    # we determine |G \ D|, then we can estimate |d| = |G \ D| - |a|
    print "Calculating |d|"
    d = calculate_d(g_dash, database, a, e1_type, e2_type)

    print "|a| =", len(a)
    print "|b| =", len(b)
    print "|c| =", len(c)
    print "|d| =", len(d)

    print "\nPrecision: ", float(len(a) + len(b)) / float(len(system_output))
    print "Recall   : ", float(len(a) + len(b)) / float(len(a) + len(b) + len(c) + len(d))
    print "\n"

if __name__ == "__main__":
    main()


