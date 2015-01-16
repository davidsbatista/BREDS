#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import os
import sys
import time
import codecs

from whoosh.analysis import RegexTokenizer
from whoosh.fields import Schema, TEXT
from whoosh.index import create_in
from whoosh.query import *


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
    regex_tokenize = re.compile('\w+(?:-\w+)+|<[A-Z]+>[^<]+</[A-Z]+>|\w+', re.U)
    tokenizer = RegexTokenizer(regex_tokenize)
    schema = Schema(sentence=TEXT(stored=True, analyzer=tokenizer))
    if not os.path.exists("index_full"):
        os.mkdir("index_full")
    idx = create_in("index_full", schema)
    return idx


@timecall
def index_sentences(writer):
    #total: 26.729.482
    count = 0
    f = codecs.open(sys.argv[1], "r", "utf-8")
    for l in f:
        try:
            writer.add_document(sentence=l.strip())
        except UnicodeDecodeError, e:
            print e
            print l
            sys.exit(0)

        count += 1
        if count % 50000 == 0:
            print count, "lines processed"
    f.close()
    writer.commit()


def main():
    idx = create_index()
    writer = idx.writer(limitmb=2048, procs=5, multisegment=True)
    index_sentences(writer)
    idx.close()


if __name__ == "__main__":
    main()