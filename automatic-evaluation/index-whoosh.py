#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import functools
import os
import sys
import time
import codecs

from whoosh.analysis import RegexTokenizer
from whoosh.fields import Schema, TEXT
from whoosh.index import create_in, open_dir
from whoosh.query import *

from nltk.corpus import stopwords
from Sentence import Sentence

bad_tokens = [",", "(", ")", ";", "''",  "``", "'s", "-", "vs.", "v", "'", ":", ".", "--"]
stopwords_list = stopwords.words('english')
not_valid = bad_tokens + stopwords_list


def timecall(f):
    @functools.wraps(f)
    def wrapper(*args, **kw):
        start = time.time()
        result = f(*args, **kw)
        end = time.time()
        print "%s %.2f seconds" % (f.__name__, end - start)
        return result
    return wrapper


def create_index():
    regex_tokenize = re.compile('\w+(?:-\w+)+|<[A-Z]+>[^<]+</[A-Z]+>|\w+', re.U)
    tokenizer = RegexTokenizer(regex_tokenize)
    schema = Schema(sentence=TEXT(stored=True, analyzer=tokenizer))
    if not os.path.exists("index_full"):
        os.mkdir("index_full")
        idx = create_in("index_full", schema)
    else:
        idx = open_dir("index_full")
    return idx

@timecall
def index_sentences(writer):
    count = 0
    f = codecs.open(sys.argv[1], "r", "utf-8")
    max_tokens = 6
    min_tokens = 1
    context_window = 2
    for l in f:
        valid = 0
        invalid = 0
        s = Sentence(l, max_tokens, min_tokens, context_window)
        if len(s.relationships) == 0:
            continue
        else:
            for r in s.relationships:
                if "." in r.between:
                    invalid += 1
                elif all(x in not_valid for x in r.between):
                    invalid += 1
                else:
                    valid += 1

            if valid > invalid:
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