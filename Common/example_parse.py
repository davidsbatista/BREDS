#!/usr/bin/env python
# -*- coding: utf-8 -

import os
import StanfordDependencies
from nltk.parse.stanford import StanfordParser


def main():
    # JAVA_HOME needs to be set, calling 'java -version' should show: java version "1.8.0_45"
    os.environ['STANFORD_PARSER'] = '/home/dsbatista/stanford-parser-full-2015-04-20/'
    os.environ['STANFORD_MODELS'] = '/home/dsbatista/stanford-parser-full-2015-04-20/'
    parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
    sd = StanfordDependencies.get_instance(backend='subprocess', jar_filename='/home/dsbatista/stanford-parser-full-2015-04-20/stanford-parser.jar')
    print "PARSER", os.environ['STANFORD_PARSER']
    print "MODELS", os.environ['STANFORD_MODELS']

    """
    examples = ["the quick brown fox jumps over the lazy dog",
                "the quick grey wolf jumps over the lazy fox",
                "Hello, My name is Melroy.", "What is your name?",
                "Amazon.com chief executive Jeff Bezos"]
    """
    examples = ["Amazon.com founder and chief executive Jeff Bezos."]
    examples = ["Google based in Mountain View, California."]

    for s in examples:
        print s
        t = parser.raw_parse(s)
        print t
        sent = sd.convert_tree(str(t[0]))
        for token in sent:
            print token
        t[0].draw()
        print "\n"


if __name__ == "__main__":
    main()