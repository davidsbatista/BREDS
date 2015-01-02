#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'dsbatista'

from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic


def similarity(word1, word2):
    #brown_ic = wordnet_ic.ic('ic-brown.dat')

    print word1
    for synset in wn.synsets(word1, pos=wn.VERB):
        print synset
        print synset.definition()
    for synset in wn.synsets(word1, pos=wn.NOUN):
        print synset
        print synset.definition()

    print "\n"
    print word2
    for synset in wn.synsets(word2, pos=wn.VERB):
        print synset
        print synset.definition()
    for synset in wn.synsets(word2, pos=wn.NOUN):
        print synset.definition()


    """
    w1 = wn.synset('dog.n.01')
    w2 = wn.synset('cat.n.01')
    print w1.jcn_similarity(w2, brown_ic)
    """

def main():
    similarity('founder', 'founded')

if __name__ == "__main__":
    main()