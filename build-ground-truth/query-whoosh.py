#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import time

from whoosh.index import open_dir
from whoosh.qparser import QueryParser
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

@timecall
def main():
    # ent1 NEAR:X r NEAR:X ent2
    # X is the maximum number of words between the query elements.
    # X = 5
    # myquery = whoosh.query.Every()

    idx = open_dir("index")

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

    t1 = query.Term("sentence", "<ORG>Microsoft</ORG>")
    t2 = query.Term("sentence", 'headquarters')
    t3 = query.Term("sentence", "<LOC>Seattle</LOC>")
    q = spans.SpanNear2([t1, t2, t3], slop=5, ordered=True)

    #parser = QueryParser("sentence", schema=idx.schema)
    #myquery = parser.parse(q)
    with idx.searcher() as searcher:
        results = searcher.search(q)
        for hit in results:
            print "ent1:", hit['entity1']
            print "ent2:", hit['entity2']
            print "s:   ", hit['sentence']
            print "\n"
        print len(results)

    idx.close()

if __name__ == "__main__":
    main()