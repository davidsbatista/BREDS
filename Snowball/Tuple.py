# -*- coding: utf-8 -*-
from nltk import PunktWordTokenizer

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"


class Tuple(object):

        def __init__(self, _e1, _e2, _sentence, _before, _between, _after, config):
            self.e1 = _e1
            self.e2 = _e2
            self.sentence = _sentence
            self.confidence = 0
            self.confidence_old = 0
            self.bef_words = _before
            self.bet_words = _between
            self.aft_words = _after
            self.config = config

            self.bef_vector = self.create_vector(self.bef_words)
            self.bet_vector = self.create_vector(self.bet_words)
            self.aft_vector = self.create_vector(self.aft_words)

            print self.bef_words, self.bef_vector
            print self.bet_words, self.bet_vector
            print self.aft_words, self.aft_vector

        def create_vector(self, text):
            return self.config.vsm.dictionary.doc2bow(self.tokenize(text))

        def tokenize(self, text):
            return [word for word in PunktWordTokenizer().tokenize(text.lower()) if word not in self.config.stopwords]

        def __str__(self):
            return "Not defined"

        def __cmp__(self, other):
            if other.confidence > self.confidence:
                return -1
            elif other.confidence < self.confidence:
                return 1
            else:
                return 0