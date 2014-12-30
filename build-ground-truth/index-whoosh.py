#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from whoosh.analysis import RegexTokenizer
from whoosh.fields import Schema, TEXT
from whoosh.index import create_in
from whoosh.query import *
from Sentence import Sentence


def create_index():
    regex_tokenize = re.compile('\w+|<[A-Z]+>[^<]+</[A-Z]+>', re.U)
    tokenizer = RegexTokenizer(regex_tokenize)

    schema = Schema(entity1=TEXT(stored=True),
                    entity2=TEXT(stored=True),
                    sentence=TEXT(stored=True, analyzer=tokenizer))

    if not os.path.exists("index"):
        os.mkdir("index")
    ix = create_in("index", schema)

    return ix


def index(idx, document):

    """
    writer = ix.writer()
    writer.add_document(entity1=u"Bush", entity2=u"Reagen", sentence=u"Bush agains Reagen <LOC>tem isto</LOC>")
    writer.add_document(entity1=u"Microsoft", entity2=u"Redmond", sentence=u"<ORG>Microsoft</ORG> in the past had the headquarters in <LOC>Redmond</LOC>")
    writer.add_document(entity1=u"Joe", entity2=u"Lisboa", sentence=u"Joe está em Lisbon")
    writer.commit()
    #searcher = ix.searcher()
    """


def query(idx,query):
    """
    with ix.searcher() as searcher:
        #myquery = Or([Term("sentence", u"Microsoft"), Term("sentence", u"Redmond")])
        #myquery = Or([Term("sentence", u"<LOC>tem isto</LOC>")])
        parser = QueryParser("sentence", ix.schema)
        #myquery = parser.parse(""whoosh library"~5")
        myquery = parser.parse('"<ORG>Microsoft</ORG> ~headquarters"~6')
        print myquery
        results = searcher.search(myquery)
        print(len(results))
        for hit in results:
            print hit['entity1']
            print hit['entity2']
            print hit['sentence']
    """


def main():
    #idx = create_index()
    for l in sys.stdin:
        s = Sentence(l.strip())
        for r in s.relationships:
            print r.ent1, '\t', r.ent2
            print r.sentence
            print r.before
            print r.between
            print r.after
            print "======================"

if __name__ == "__main__":
    main()