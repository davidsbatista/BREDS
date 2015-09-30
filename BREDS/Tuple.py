#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

from numpy import zeros


class Tuple(object):
        # http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html
        filter_pos = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB']

        def __init__(self, _e1, _e2, _sentence, _before, _between, _after, config):
            self.e1 = _e1
            self.e2 = _e2
            self.sentence = _sentence
            self.confidence = 0
            self.bef_tags = _before
            self.bet_tags = _between
            self.bet_filtered = None
            self.aft_tags = _after
            self.bef_words = " ".join([x[0] for x in self.bef_tags])
            self.bet_words = " ".join([x[0] for x in self.bet_tags])
            self.aft_words = " ".join([x[0] for x in self.aft_tags])
            self.bef_vector = None
            self.bet_vector = None
            self.aft_vector = None
            self.passive_voice = False
            self.construct_vectors(config)

        def __str__(self):
            return str(self.e1+'\t'+self.e2+'\t'+self.bef_words+'\t'+self.bet_words+'\t'+self.aft_words).encode("utf8")

        def __hash__(self):
            return hash(self.e1) ^ hash(self.e2) ^ hash(self.bef_words) ^ hash(self.bet_words) ^ hash(self.aft_words)

        def __eq__(self, other):
            return (self.e1 == other.e1 and self.e2 == other.e2 and self.bef_words == other.bef_words and
                    self.bet_words == other.bet_words and self.aft_words == other.aft_words)

        def __cmp__(self, other):
            if other.confidence > self.confidence:
                return -1
            elif other.confidence < self.confidence:
                return 1
            else:
                return 0

        def construct_vectors(self, config):
            # Check if BET context contains a ReVerb pattern
            reverb_pattern = config.reverb.extract_reverb_patterns_tagged_ptb(self.bet_tags)
            if len(reverb_pattern) > 0:
                # test for passive voice presence
                self.passive_voice = config.reverb.detect_passive_voice(reverb_pattern)
                bet_words = reverb_pattern
            else:
                self.passive_voice = False
                bet_words = self.bet_tags

            self.bet_filtered = [t[0] for t in bet_words if t[0].lower() not in config.stopwords and t[1]
                                 not in self.filter_pos]

            # compute the vector over the filtered BET context
            self.bet_vector = self.pattern2vector_sum(self.bet_filtered, config)

            # compute the vector for words before the first entity, and words after the second entity
            bef_no_tags = [t[0] for t in self.bef_tags]
            aft_no_tags = [t[0] for t in self.aft_tags]
            self.bef_vector = self.pattern2vector_sum(bef_no_tags, config)
            self.aft_vector = self.pattern2vector_sum(aft_no_tags, config)

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
                    pattern_vector = config.word2vec[tokens[0].strip()]
                except KeyError:
                    pass

            return pattern_vector