#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re

__author__ = 'dsbatista'

from whoosh.analysis import StopFilter, RegexTokenizer
import multiprocessing
import fileinput
import functools
import time
import sys

from whoosh.index import open_dir
from whoosh.query import spans
from whoosh import query
from Sentence import Sentence


def timecall(f):
    @functools.wraps(f)
    def wrapper(*args, **kw):
        start = time.time()
        result = f(*args, **kw)
        end = time.time()
        print "%s %.2f seconds" % (f.__name__, end - start)
        return result
    return wrapper


def multi_thread_query(relationships, results, e1_type, e2_type):
    idx = open_dir("index_test")
    regex_tokenize = re.compile('\w+|-|<[A-Z]+>[^<]+</[A-Z]+>', re.U)
    tokenizer = RegexTokenizer(regex_tokenize)
    stopper = StopFilter()

    with idx.searcher() as searcher:
        while True:
            r = relationships.get_nowait()
            # calculate the PMI
            entity1 = "<"+e1_type+">"+r.ent1+"</"+e1_type+">"
            entity2 = "<"+e2_type+">"+r.ent2+"</"+e2_type+">"

            terms = list()
            for token in stopper(tokenizer(unicode(r.between), renumber=True)):
                terms.append(query.Term("sentence", token.text))

            #print terms
            t1 = query.Term("sentence", entity1)
            #t2 = query.Term("sentence", r.between)  # TODO: remover stopwords do r.between
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
                    print entity1, '\t', r.between, '\t', entity2, pmi
                    results.append(r)

            if relationships.empty() is True:
                break


@timecall
def main():
    corpus = sys.argv[1]
    manager = multiprocessing.Manager()
    relationships = manager.Queue()
    results = manager.list()
    print "Reading relationships from", sys.argv[1]
    for line in fileinput.input(corpus):
        s = Sentence(line.strip())
        for r in s.relationships:
            relationships.put(r)
    fileinput.close()

    e1_type = "ORG"
    e2_type = "PER"
    #num_cpus = multiprocessing.cpu_count()
    num_cpus = 1

    processes = [multiprocessing.Process(target=multi_thread_query, args=(relationships, results, e1_type, e2_type, )) for i in range(num_cpus)]
    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()

    print "Relationships with high PMI", len(results)

if __name__ == "__main__":
    main()
