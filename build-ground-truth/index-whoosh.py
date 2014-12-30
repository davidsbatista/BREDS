#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from whoosh.analysis import RegexTokenizer
from whoosh.fields import Schema, TEXT
from whoosh.index import create_in
from whoosh.query import *
from whoosh.qparser import QueryParser


def create_index():
    regex_tokenize = re.compile('\w+|<[A-Z]+>[^<]+</[A-Z]+>', re.U)
    tokenizer = RegexTokenizer(regex_tokenize)

    schema = Schema(entity1=TEXT(stored=True),
                    entity2=TEXT(stored=True),
                    sentence=TEXT(stored=True, analyzer=tokenizer))


    if not os.path.exists("index"):
        os.mkdir("index")
    ix = create_in("index", schema)
    #ix = open_dir("index")

    """
    An analyzer is a function or callable class (a class with a __call__ method) that takes a unicode string
    and returns a generator of tokens.
    """

    # NOTE: Indexed text fields must be passed a unicode value!
    writer = ix.writer()
    writer.add_document(entity1=u"Bush", entity2=u"Reagen", sentence=u"Bush agains Reagen <LOC>tem isto</LOC>")
    writer.add_document(entity1=u"Microsoft", entity2=u"Redmond", sentence=u"<ORG>Microsoft</ORG> in the past had the headquarters in <LOC>Redmond</LOC>")
    writer.add_document(entity1=u"Joe", entity2=u"Lisboa", sentence=u"Joe está em Lisbon")
    writer.commit()

    #searcher = ix.searcher()

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



def main():
    create_index()
    """
    #regex = re.compile('<[A-Z]+>([^<]+)</[A-Z]+>', re.U)
    #self.TOKENIZER= r'\,|\(|\)|\w+(?:-\w+)+|\d+(?:[:|/]\d+)+|\d+(?:[.]?[oaºª°])+|\w+\'\w+|\d+(?:[,|.]\d+)*\%?|[\w+\.-]+@[\w\.-]+|https?://[^\s]+|\w+'

    regex = re.compile('\w+|<[A-Z]+>[^<]+</[A-Z]+>', re.U)

    for l in sys.stdin:
        matches = []
        # before indexing documents, change spaces withing entities to "_", e.g.:
        # In <LOC>Bonn</LOC> , the head of the <ORG>German Social Democratic Party</ORG>
        # becomes:
        # In <LOC>Bonn</LOC> , the head of the <ORG>German_Social_Democratic_Party</ORG>

        print re.findall(regex, l)

        for m in re.finditer(regex, l):
            matches.append(m)
            for x in range(0, len(matches)):
                new = re.sub(r'\s', "_", matches[x].group())
                l = l.replace(matches[x].group(), new)

        print l
    """

if __name__ == "__main__":
    main()