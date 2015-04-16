# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

from nltk import PunktWordTokenizer, pos_tag
from reverb.ReVerb import Reverb


class Tuple(object):
        # http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html
        # select everything except stopwords, ADJ and ADV
        filter_pos = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB']

        def __init__(self, _e1, _e2, _sentence, _before, _between, _after, config):
            self.e1 = _e1
            self.e2 = _e2
            self.sentence = _sentence
            self.confidence = 0
            self.confidence_old = 0
            self.bef_words = _before
            self.bet_words = _between
            self.aft_words = _after
            self.bef_vector = None
            self.bet_vector = None
            self.aft_vector = None
            self.passive_voice = False
            self.patterns_vectors = list()
            self.patterns_words = list()
            self.debug = False
            self.extract_patterns(config)

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
        def detect_passive_voice(config, pattern):
            aux_verbs = ['be']
            for i in range(0, len(pattern)):
                if pattern[i][1].startswith('V'):
                    verb = config.lmtzr.lemmatize(pattern[i][0], 'v')
                    if verb in aux_verbs and i + 2 <= len(pattern) - 1:
                        if (pattern[i+1][1] == 'VBN' or pattern[i+1][1] == 'VBD') and pattern[-1][0] == 'by':
                            return True
                        else:
                            return False

        def construct_pattern_vector(self, pattern_tags, config):
            # remove stopwords and adjectives
            """
            if self.debug is True:
                print "pattern"
                print "original", pattern_tags
            """

            pattern = [t[0] for t in pattern_tags if t[0].lower() not in config.stopwords and t[1] not in self.filter_pos]

            """
            if self.debug is True:
                print "after filtering", pattern
            """

            if len(pattern) >= 1:
                if config.embeddings == 'average':
                    pattern_vector = config.word2vecwrapper.pattern2vector_average(pattern, config)
                elif config.embeddings == 'sum':
                    pattern_vector = config.word2vecwrapper.pattern2vector_sum(pattern, config)

                """
                if (self.e1 == 'Nokia' and self.e2 == 'Espoo') or (self.e1 == 'Pfizer' and self.e2 == 'New York'):
                    print "vector", pattern_vector
                """

                return pattern_vector

        def construct_words_vectors(self, words, config):
            # split text into tokens and tag them using NLTK's default English tagger
            # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
            text_tokens = PunktWordTokenizer().tokenize(words)
            tags_ptb = pos_tag(text_tokens)

            """
            if (self.e1 == 'Nokia' and self.e2 == 'Espoo') or (self.e1 == 'Pfizer' and self.e2 == 'New York'):
                print "words"
                print "original", words
            """

            pattern = [t[0] for t in tags_ptb if t[0].lower() not in config.stopwords and t[1] not in self.filter_pos]
            if len(pattern) >= 1:

                """
                if (self.e1 == 'Nokia' and self.e2 == 'Espoo') or (self.e1 == 'Pfizer' and self.e2 == 'New York'):
                    print "after filtering", pattern
                """

                if config.embeddings == 'average':
                    words_vector = config.word2vecwrapper.pattern2vector_average(pattern, config)
                elif config.embeddings == 'sum':
                    words_vector = config.word2vecwrapper.pattern2vector_sum(pattern, config)

                """
                if (self.e1 == 'Nokia' and self.e2 == 'Espoo') or (self.e1 == 'Pfizer' and self.e2 == 'New York'):
                    print words_vector
                """

                return words_vector

        def extract_patterns(self, config):
            # extract ReVerb pattern from BET context
            patterns_bet_tags = Reverb.extract_reverb_patterns_ptb(self.bet_words)

            # detect passive voice in BET ReVerb pattern
            if len(patterns_bet_tags) > 0:
                self.passive_voice = self.detect_passive_voice(config, patterns_bet_tags)

            # Construct word2vec representations of the patterns/words
            # Three context vectors
            #  BEF: 2 words
            #  BET: ReVerb pattern
            #  AFT: 2 words

            # BEF context
            if len(self.bef_words) > 0:
                self.bef_vector = self.construct_words_vectors(self.bef_words, config)

            # BET context
            if len(patterns_bet_tags) > 0:
                self.bet_vector = self.construct_pattern_vector(patterns_bet_tags, config)
            elif len(self.bet_words) > 0:
                self.bet_vector = self.construct_words_vectors(self.bet_words, config)

            # AFT context
            if len(self.aft_words) > 0:
                self.aft_vector = self.construct_words_vectors(self.aft_words, config)