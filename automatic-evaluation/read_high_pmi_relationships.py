#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import cPickle
import sys


class ExtractedFact(object):
    def __init__(self, _e1, _e2, _score, _bef, _bet, _aft, _sentence, _passive_voice):
        self.ent1 = _e1
        self.ent2 = _e2
        self.score = _score.strip()
        self.bef_words = _bef
        self.bet_words = _bet
        self.aft_words = _aft
        self.sentence = _sentence
        self.passive_voice = _passive_voice

    def __hash__(self):
        sig = hash(self.ent1) ^ hash(self.ent2) ^ hash(self.bef_words) ^ hash(self.bet_words) ^ hash(self.aft_words) ^ \
            hash(self.score) ^ hash(self.sentence)
        return sig

    def __eq__(self, other):
        if self.ent1 == other.ent1 and self.ent2 == other.ent2 and self.score == other.score and self.bef_words == \
                other.bef_words and self.bet_words == other.bet_words and self.aft_words == other.aft_words \
                and self.sentence == other.sentence:
            return True
        else:
            return False


def main():
    rel_file = sys.argv[1]
    f = open(rel_file)
    print "\nLoading high PMI facts not in the database", rel_file
    relationships = cPickle.load(f)
    for r in relationships:
        print "sentence    :", r.sentence
        print "relationship:", r.ent1, r.ent2
        print "\n"
    print len(relationships)
    f.close()


if __name__ == "__main__":
    main()