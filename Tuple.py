__author__ = 'dsbatista'

from reverb.ReVerb import Reverb


class Tuple(object):

        def __init__(self, _e1, _e2, _sentence, _before, _between, _after):
            self.e1 = _e1
            self.e2 = _e2
            self.sentence = _sentence
            self.confidence = 0
            self.confidence_old = 0
            self.bef = _before
            self.bet = _between
            self.aft = _after
            self.relational_words_vector = list()
            self.ReVerbpatterns = list()
            self.extract_patterns(self)

        @staticmethod
        def extract_patterns(self):
            """
            Extract ReVerb patterns and construct Word2Vec representations
            If no ReVerb patterns are found extract word from context
            """
            patterns_bef = Reverb.extract_reverb_patterns(self.bef)
            patterns_bet = Reverb.extract_reverb_patterns(self.bet)
            patterns_aft = Reverb.extract_reverb_patterns(self.aft)

            print patterns_bef[0]
            print patterns_bet[0]
            print patterns_aft[0]
