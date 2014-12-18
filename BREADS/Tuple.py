from BREADS import Word2VecWrapper

__author__ = 'dsbatista'

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
            #patterns_bef = Reverb.extract_reverb_patterns(self.bef)
            patterns_bet = Reverb.extract_reverb_patterns(self.bet)
            #patterns_aft = Reverb.extract_reverb_patterns(self.aft)

            """
            if len(patterns_bef[0]) > 0:
                pattern_vector_bef = Word2VecWrapper.pattern2vector(patterns_bef[0], config)
                print pattern_vector_bef
            """
            for p in patterns_bet:
                self.patterns_words.append(p)
                pattern_vector_bet = Word2VecWrapper.pattern2vector(p, config)
                self.pattern_vectors.append(pattern_vector_bet)
            """
            if len(patterns_aft[0]) > 0:
                pattern_vector_aft = Word2VecWrapper.pattern2vector(patterns_aft[0], config)
                print pattern_vector_aft
            """