#!/usr/bin/env python
# -*- coding: utf-8 -
import re

__author__ = 'dsbatista'

from whoosh.index import open_dir
from whoosh.query import spans
from whoosh import query


def proximity_pmi(entity1, entity2, rel, index, distance, q_limit):
    tokenize = re.compile('\w+(?:-\w+)+|<[A-Z]+>[^<]+</[A-Z]+>|\w+', re.U)
    entity = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)
    idx = open_dir(index)

    n_1 = set()
    n_2 = set()
    n_3 = set()

    t1 = query.Term('sentence', entity1)
    t3 = query.Term('sentence', entity2)

    with idx.searcher() as searcher:
        # Entities proximity query without relational words
        q1 = spans.SpanNear2([t1, t3], slop=distance, ordered=True, mindist=1)
        print q1
        hits_1 = searcher.search(q1, limit=q_limit)
        print "entities hits", len(hits_1)
        print "\n"
        for d in hits_1:
            n_1.add(d.get("sentence"))

        tokens_rel = re.findall(tokenize, rel)
        # Entities proximity considering relational words
        token_terms = list()
        for t in tokens_rel:
            if re.search(entity, t) is None:
                token_terms.append(query.Term('sentence', t))

        l1 = [t for t in token_terms]
        l1.insert(0, t1)
        l2 = [t for t in token_terms]
        l2.append(t3)

        q2 = spans.SpanNear2(l1, slop=distance-1, ordered=True, mindist=1)
        print q2
        hits_2 = searcher.search(q2, limit=q_limit)

        q3 = spans.SpanNear2(l2, slop=distance-1, ordered=True, mindist=1)
        print q3
        hits_3 = searcher.search(q3, limit=q_limit)

        for d in hits_1:
            n_1.add(d.get("sentence"))

        for d in hits_2:
            n_2.add(d.get("sentence"))

        for d in hits_3:
            n_3.add(d.get("sentence"))

        print "q1", len(hits_1)
        print "q2", len(hits_2)
        print "q3", len(hits_3)
        print "q2 intersect q3", len(n_2.intersection(n_3))

        entities_occurr = len(hits_1)
        entities_occurr_with_r = len(n_1.intersection(n_2).intersection(n_3))

        return float(entities_occurr_with_r) / float(entities_occurr)

"""
<ORG>Juventus</ORG> failed to fully capitalise after veteran forward <PER>Cristiano Doni</PER> produced a masterful display to inspire <ORG>Atalanta</ORG> to a stunning 3-1 victory over <PER>Jose Mourinho</PER> 's <ORG>Inter Milan</ORG> on Sunday .
Juventus
Cristiano Doni

[Term('sentence', '<ORG>Juventus</ORG>'), Term('sentence', u'failed'), Term('sentence', u'fully'), Term('sentence', u'capitalise'), Term('sentence', u'after'), Term('sentence', u'veteran'), Term('sentence', u'forward'), Term('sentence', '<PER>Cristiano Doni</PER>')] 2
[Term('sentence', '<ORG>Juventus</ORG>'), Term('sentence', '<PER>Cristiano Doni</PER>')] 1

SpanNear2([Term('sentence', '<ORG>Juventus</ORG>'), Term('sentence', u'failed'), Term('sentence', u'fully'), Term('sentence', u'capitalise'), Term('sentence', u'after'), Term('sentence', u'veteran'), Term('sentence', u'forward'), Term('sentence', '<PER>Cristiano Doni</PER>')], slop=2, ordered=True, mindist=1)
SpanNear2([Term('sentence', '<ORG>Juventus</ORG>'), Term('sentence', '<PER>Cristiano Doni</PER>')], slop=8, ordered=True, mindist=1)
"""


def main():
    index = "index_2005_2010"
    distance = 9
    q_limit = 500

    entity1 = "<ORG>Elizabeth II</ORG>"
    entity2 = "<PER>Obama</PER>"
    rel = "also sent a personal message of support for"
    pmi = proximity_pmi(entity1, entity2, rel, index, distance, q_limit)
    print entity1, rel, entity2, str(pmi)

    entity1 = "<ORG>Microsoft</ORG>"
    entity2 = "<PER>Bill Gates</PER>"
    rel = "founder"
    pmi = proximity_pmi(entity1, entity2, rel, index, distance, q_limit)
    print entity1, rel, entity2, str(pmi)

if __name__ == "__main__":
    main()