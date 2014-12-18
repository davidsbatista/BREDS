# -*- coding: utf-8 -*-

__author__ = 'dsbatista'

from Word2VecWrapper import Word2VecWrapper
from reverb.ReVerb import Reverb


class Tuple(object):

        def __init__(self, _e1, _e2, _sentence, _before, _between, _after, config):
            self.e1 = _e1
            self.e2 = _e2
            self.sentence = _sentence
            self.confidence = 0
            self.confidence_old = 0
            self.bef = _before
            self.bet = _between
            self.aft = _after
            self.pattern_vectors = list()
            self.patterns_words = list()
            self.extract_patterns(self, config)

        @staticmethod
        def extract_patterns(self, config):
            """
            Extract ReVerb patterns and construct Word2Vec representations
            If no ReVerb patterns are found extract word from context
            """
            patterns_bet, patterns_bet_tags = Reverb.extract_reverb_patterns(self.bet)
            if len(patterns_bet) > 0:
                self.patterns_words = patterns_bet
                # TODO: só estou a usar o primeiro ReVerb pattern
                pattern_vector_bet = Word2VecWrapper.pattern2vector(patterns_bet_tags[0], config)
                self.pattern_vectors.append(pattern_vector_bet)