# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

from nltk import PunktWordTokenizer, pos_tag
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
            self.patterns_vectors = list()
            self.patterns_words = list()
            self.extract_patterns(self, config)

        def __str__(self):
            return " | ".join([p for p in self.patterns_words]).encode("utf8")

        @staticmethod
        def extract_patterns(self, config):
            """ Extract ReVerb patterns and construct Word2Vec representations"""
            patterns_bet, patterns_bet_tags = Reverb.extract_reverb_patterns(self.bet)
            if len(patterns_bet) > 0:
                self.patterns_words = patterns_bet
                # TODO: só estou a usar o primeiro ReVerb pattern
                pattern_vector_bet = Word2VecWrapper.pattern2vector(patterns_bet_tags[0], config)
                self.patterns_vectors.append(pattern_vector_bet)
            else:
                """ If no ReVerb patterns are found extract word from context """

                # split text into tokens
                text_tokens = PunktWordTokenizer().tokenize(self.bet)

                # tag the sentence, using the default NTLK English tagger
                # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
                tags_ptb = pos_tag(text_tokens)

                # http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html
                # select everything except stopwords and ADJ, ADV
                filter_pos = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB']
                pattern = [t[0] for t in tags_ptb if t[0].lower() not in config.stopwords and t[1] not in filter_pos]
                patterns_vector_bet = Word2VecWrapper.pattern2vector(pattern, config)
                self.patterns_vectors.append(patterns_vector_bet)
                self.patterns_words = pattern