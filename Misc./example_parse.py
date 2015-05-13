#!/usr/bin/env python
# -*- coding: utf-8 -

import os
import StanfordDependencies
from nltk.parse.stanford import StanfordParser


def main():
    # JAVA_HOME needs to be set, calling 'java -version' should show: java version "1.8.0_45" or higher
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
    examples = ["NBC is currently a poor third in the ratings of the four US networks , having slipped from first place when General Electric acquired it from RCA in 1985 ."]
    examples = ["The new channel will focus on `` people , contemporary and historical , in and out of the headlines '' according to Westinghouse Electric , the company that acquired CBS last year for 5.4 billion dollars ."]
    #sentence: In favour of the deal were <PER>Ted Turner</PER> , founder and boss of <ORG>TBS</ORG> , a cable-based business , and <ORG>Time Warner</ORG> chairman <PER>Gerald Levin</PER> , whose empire already held an 18 percent stake in <ORG>TBS</ORG> .
    #sentence: Including the roughly 1,500 workers picked up in the <ORG>DoubleClick</ORG> acquisition , <ORG>Google</ORG> now has more than 18,000 employees worldwide .
    examples = ["The story was first seen at Techcrunch , the picked up by the Wall Street Journal and has since been the subject of much talk , posts and thoughts over the past few days and finally it has been confirmed that <p1> Google </p1> have purchased <p2> Youtube </p2> for $ 1.65 billion in an official statement ."]
    examples = ["For KFC , whose Louisville headquarters resemble a white-columned mansion , the recipe is more than a treasured link to its roots"]
    for s in examples:
        print s
        t = parser.raw_parse(s)
        print t
        # note: http://www.nltk.org/_modules/nltk/parse/stanford.html
        # the wrapper for StanfordParser does not give syntatic dependencies
        sent = sd.convert_tree(str(t[0]))
        for token in sent:
            print token
        t[0].draw()
        print "\n"


if __name__ == "__main__":
    main()