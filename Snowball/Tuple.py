# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

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
            self.bef_vector = self.create_vector(self.bef_words)
            self.bet_vector = self.create_vector(self.bet_words)
            self.aft_vector = self.create_vector(self.aft_words)

        def create_vector(self, text):
            vect_ids = self.config.vsm.dictionary.doc2bow(self.tokenize(text))
            return self.config.vsm.tf_idf_model[vect_ids]

        def tokenize(self, text):
            return [word for word in PunktWordTokenizer().tokenize(text.lower()) if word not in self.config.stopwords]

        def __str__(self):
            return str(self.bef_words.encode("utf8")+','+self.bet_words.encode("utf8")+','+self.aft_words.encode("utf8"))

        def __cmp__(self, other):
            if other.confidence > self.confidence:
                return -1
            elif other.confidence < self.confidence:
                return 1
            else:
                return 0