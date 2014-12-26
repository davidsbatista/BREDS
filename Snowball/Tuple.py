# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import sys
from nltk import PunktWordTokenizer


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
            self.bef_vector = None
            self.bet_vector = None
            self.aft_vector = None
            #self.bef_vector = self.create_vector(self.bef_words)
            #self.bet_vector = self.create_vector(self.bet_words)
            #self.aft_vector = self.create_vector(self.aft_words)

        def get_vector(self, context):
            if context == "bef":
                return self.bef_vector
            elif context == "bet":
                return self.bet_vector
            elif context == "aft":
                return self.aft_vector
            else:
                print "Error, vector must be 'bef', 'bet' or 'aft'"
                sys.exit(0)

        def create_vector(self, text):
            vect_ids = self.config.vsm.dictionary.doc2bow(self.tokenize(text))
            return self.config.vsm.tf_idf_model[vect_ids]

        def tokenize(self, text):
            return [word for word in PunktWordTokenizer().tokenize(text.lower()) if word not in self.config.stopwords]

        def __str__(self):
            return str(self.bef_words.encode("utf8")+' '+self.bet_words.encode("utf8")+' '+self.aft_words.encode("utf8"))

        def __eq__(self, other):
            print "chamai o __eq__ do Tuple"
            return (self.e1 == other.e1 and self.e2 == other.e2 and self.bef_words == other.bef_words and
                    self.bet_words == other.bet_words and self.aft_words == other.aft_words)
