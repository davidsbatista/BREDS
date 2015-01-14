#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import time

from whoosh.index import open_dir
from whoosh.query import spans
from whoosh import query


def timecall(f):
    @functools.wraps(f)
    def wrapper(*args, **kw):
        start = time.time()
        result = f(*args, **kw)
        end = time.time()
        print "%s %.2f seconds" % (f.__name__, end - start)
        return result
    return wrapper


def calculate_pmi(e1, e2, r):
    # NOTE: this queries an index build over the whole AFT corpus
    """
    # sentences with tagged entities are indexed in whoosh, perform the following query
    # ent1 NEAR:X r NEAR:X ent2
    # X is the maximum number of words between the query elements.
    #
    """
    idx = open_dir("index")
    t1 = query.Term("sentence", e1)
    t2 = query.Term("sentence", r)
    t3 = query.Term("sentence", e2)
    q1 = spans.SpanNear2([t1, t2, t3], slop=5, ordered=False)
    q2 = spans.SpanNear2([t1, t3], slop=5, ordered=False)

    with idx.searcher() as searcher:
        entities_r = searcher.search(q1)
        entities = searcher.search(q2)
        print len(entities_r)
        print len(entities)
        pmi = float(len(entities_r)) / float(len(entities))

    idx.close()
    return pmi


@timecall
def main():
    #pmi = calculate_pmi('<ORG>Microsoft</ORG>', '<LOC>Seattle</LOC>', "headquarters")
    #print pmi

    # To find documents where “<ORG>Microsoft</ORG>” occurs at most X positions before “headquarters”:
    # q1t1 = query.Term("sentence", "<ORG>Microsoft</ORG>")
    # q1t2 = query.Term("sentence", "accused")
    # q = spans.SpanNear2((q1t1, q1t2), slop=5)

    # To match documents where “apple” occurs at most 10 places before “bear” in the “text” field and “cute” is between
    # X = 5
    # t1 = query.Term("sentence", "<ORG>Microsoft</ORG>")
    # t2 = query.Term("sentence", "<PER>Bill Gates</PER>")
    # near = spans.SpanNear(t1, t2, slop=2)
    # q = spans.SpanContains(near, query.Term("sentence", "founder"))

    """
    t1 = query.Term("sentence", "<ORG>Microsoft</ORG>")
    t2 = query.Term("sentence", 'headquarters')
    t3 = query.Term("sentence", "<LOC>Seattle</LOC>")
    q = spans.SpanNear2([t1, t2, t3], slop=5, ordered=True)

    #parser = QueryParser("sentence", schema=idx.schema)
    #myquery = parser.parse(q)
    with idx.searcher() as searcher:
        results = searcher.search(q)
        for hit in results:
            print hit['sentence']
            print "\n"
        print len(results)
    """

    idx.close()

if __name__ == "__main__":
    main()