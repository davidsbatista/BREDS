#!/usr/bin/env python
# -*- coding: utf-8 -*-

from whoosh.index import open_dir
from whoosh.query import spans
from whoosh import query

MAX_TOKENS_AWAY = 6

e1_type = "ORG"
e2_type = "ORG"

ent1 = "AOL"
ent2 = "Time Warner"

#ent1 = "IKEA"
#ent2 = "Sweden"


def main():
    q_limit = 500
    idx = open_dir("index_2000_2010")
    with idx.searcher() as searcher:
        entity1 = "<"+e1_type+">"+ent1+"</"+e1_type+">"
        entity2 = "<"+e2_type+">"+ent2+"</"+e2_type+">"
        t1 = query.Term('sentence', entity1)
        t3 = query.Term('sentence', entity2)

        """
        hits = searcher.search(t1, limit=q_limit)
        print t1
        print len(hits)

        for h in hits:
            print h.get("sentence")
        """

        # Entities proximity query without relational words
        q2 = spans.SpanNear2([t1, t3], slop=MAX_TOKENS_AWAY, ordered=True, mindist=1)
        hits = searcher.search(q2, limit=q_limit)

        print q2
        print "HITS", len(hits)
        for h in hits:
            print h.get("sentence")

if __name__ == "__main__":
    main()