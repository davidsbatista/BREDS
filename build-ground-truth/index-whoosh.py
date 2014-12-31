#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import os
import sys
import time

from whoosh.analysis import RegexTokenizer
from whoosh.fields import Schema, TEXT
from whoosh.index import create_in
from whoosh.query import *
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


@timecall
def create_index():
    regex_tokenize = re.compile('\w+|<[A-Z]+>[^<]+</[A-Z]+>', re.U)
    tokenizer = RegexTokenizer(regex_tokenize)
    schema = Schema(entity1=TEXT(stored=True),
                    entity2=TEXT(stored=True),
                    sentence=TEXT(stored=True, analyzer=tokenizer))
    if not os.path.exists("index_full"):
        os.mkdir("index_full")
    idx = create_in("index_full", schema)
    return idx


@timecall
def index_sentences(writer):
    #total: 26.729.482
    count = 0
    for l in sys.stdin:
        s = Sentence(l.strip())
        for r in s.relationships:
            writer.add_document(entity1=unicode(r.ent1), entity2=unicode(r.ent2), sentence=unicode(r.sentence))
        count += 1
        if count % 50000 == 0:
            print count, "lines processed"
    writer.commit()


def main():
    idx = create_index()
    writer = idx.writer(limitmb=2048, procs=8, multisegment=True)
    index_sentences(writer)
    idx.close()


if __name__ == "__main__":
    main()