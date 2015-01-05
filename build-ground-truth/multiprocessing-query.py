#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'dsbatista'

from multiprocessing.dummy import Process
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
    idx = open_dir("index")
    with idx.searcher() as searcher:
        while True:
            r = relationships.get_nowait()
            # calculate the PMI
            entity1 = "<"+e1_type+">"+r.ent1+"</"+e1_type+">"
            entity2 = "<"+e2_type+">"+r.ent2+"</"+e2_type+">"
            t1 = query.Term("sentence", entity1)
            t2 = query.Term("sentence", r.between)  # TODO: remover stopwords do r.between
            t3 = query.Term("sentence", entity2)
            q1 = spans.SpanNear2([t1, t2, t3], slop=1, ordered=True)
            q2 = spans.SpanNear2([t1, t3], slop=1, ordered=True)
            entities_r = searcher.search(q1)
            entities = searcher.search(q2)

            print entity1, '\t', r.between, '\t', entity2, len(entities_r), len(entities)

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
    num_cpus = 5

    processes = [multiprocessing.Process(target=multi_thread_query, args=(relationships, results, e1_type, e2_type, )) for i in range(num_cpus)]
    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()

    print "Relationships with high PMI", len(results)

if __name__ == "__main__":
    main()
