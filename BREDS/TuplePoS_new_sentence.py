#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

from numpy import zeros


class Tuple(object):

        def __init__(self, _e1, _e2, _sentence, _before, _between, _after, config):
            self.e1 = _e1
            self.e2 = _e2
            self.sentence = _sentence
            self.confidence = 0
            self.confidence_old = 0
            self.before = _before
            self.between = _between
            self.after = _after
            self.bef_vector = None
            self.bet_vector = None
            self.aft_vector = None
            self.bef_words = None
            self.bet_words = None
            self.aft_words = None
            self.passive_voice = False
            self.debug = False
            self.construct_vectors(config)

        def __str__(self):
            return str(self.e1+'\t'+self.e2+'\t'+self.bef_words+'\t'+self.bet_words+'\t'+self.aft_words).encode("utf8")

        def __cmp__(self, other):
            if other.confidence > self.confidence:
                return -1
            elif other.confidence < self.confidence:
                return 1
            else:
                return 0

        def __eq__(self, other):
            return (self.e1 == other.e1 and self.e2 == other.e2 and self.bef_words == other.bef_words and
                    self.bet_words == other.bet_words and self.aft_words == other.aft_words)

        @staticmethod
        def pattern2vector_sum(tokens, config):
            pattern_vector = zeros(config.vec_dim)
            if len(tokens) > 1:
                for t in tokens:
                    try:
                        vector = config.word2vec[t[0].strip()]
                        pattern_vector += vector
                    except KeyError:
                        continue

            elif len(tokens) == 1:
                try:
                    pattern_vector = config.word2vec[tokens[0][0].strip()]
                except KeyError:
                    pass

            return pattern_vector

        def construct_words_vectors(self, tagged_words, context, config):
            # remove stopwords and adjective
            words = [t for t in tagged_words if t[0].lower() not in config.stopwords and t[1] not in config.filter_pos]
            if len(words) >= 1:
                vector = self.pattern2vector_sum(words, config)
                if context == 'before':
                    self.bef_vector = vector
                    self.bef_words = words
                elif context == 'between':
                    self.bet_vector = vector
                    self.bet_words = words
                elif context == 'after':
                    self.aft_vector = vector
                    self.aft_words = words

        def construct_pattern_vector(self, reverb_pattern, config):
            # remove stopwords and adjectives
            pattern = [t[0] for t in reverb_pattern if t[0].lower() not in config.stopwords and t[1] not in config.filter_pos]
            if len(pattern) >= 1:
                vector = self.pattern2vector_sum(pattern, config)
                return vector
            else:
                "ERROR"
                return zeros(config.vec_dim)

        def construct_vectors(self, config):
            reverb_pattern = config.reverb.extract_reverb_patterns_tagged_ptb(self.between)

            if len(reverb_pattern) > 0:
                self.passive_voice = config.reverb.detect_passive_voice(reverb_pattern)
                self.bet_vector = self.construct_pattern_vector(reverb_pattern, config)
                self.bet_words = reverb_pattern
            else:
                self.passive_voice = False
                self.construct_words_vectors(reverb_pattern, "between", config)

            # extract words before the first entity, and words after the second entity
            if len(self.before) > 0:
                self.construct_words_vectors(self.before, "before", config)
            if len(self.after) > 0:

                self.construct_words_vectors(self.after, "after", config)


            """
            print self.sentence
            print "BEF", self.before
            print "BET", self.between
            print "AFT", self.after
            print "passive", self.passive_voice
            print "ReVerb:", self.reverb_pattern
            print "\n"
            """
