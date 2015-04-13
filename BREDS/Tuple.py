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
            self.bef = _before
            self.bet = _between
            self.aft = _after
            self.bef_vector = None
            self.bet_vector = None
            self.aft_vector = None
            self.vector = None
            self.passive_voice = False
            self.patterns_vectors = list()
            self.patterns_words = list()
            self.debug = False
            self.extract_patterns(config)

        def __str__(self):
            return str(self.patterns_words).encode("utf8")

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
                # TODO: contar com adjectivos pelo meio
                if pattern[i][1].startswith('V'):
                    verb = config.lmtzr.lemmatize(pattern[i][0], 'v')
                    if verb in aux_verbs and i + 2 <= len(pattern) - 1:
                        if (pattern[i + 1][1] == 'VBN' or pattern[i + 1][1] == 'VBD') and pattern[i + 2][0] == 'by':
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
            # extract ReVerb pattern
            patterns_bet_tags = Reverb.extract_reverb_patterns_ptb(self.bet)

            # detect passive voice in BET ReVerb pattern
            if len(patterns_bet_tags) > 0:
                self.passive_voice = self.detect_passive_voice(config, patterns_bet_tags)

            # Construct word2vec representations of the patterns/words
            if config.vector == 'version_1':
                ###################################################################
                # Version 1: just a single vector with the BET ReVerb pattern/Words
                ###################################################################
                # BET context
                if len(patterns_bet_tags) > 0:
                    self.bet_vector = self.construct_pattern_vector(patterns_bet_tags, config)
                else:
                    self.bet_vector = self.construct_words_vectors(self.bet_words, config)
                self.vector = self.bet_vector

            elif config.vector == 'version_2':
                ##################################################################################
                # Version 2: three context vectors
                #            BEF: 2 words
                #            BET: ReVerb pattern
                #            AFT: 2 words
                ##################################################################################

                """
                if (self.e1 == 'Nokia' and self.e2 == 'Espoo') or (self.e1 == 'Pfizer' and self.e2 == 'New York'):
                    print self.sentence
                    print "BEF", self.bef
                    print "BET", self.bet
                    print "AFT", self.aft
                    self.debug = True
                """

                # BEF context
                if len(self.bef) > 0:
                    self.bef_vector = self.construct_words_vectors(self.bef, config)

                # BET context
                if len(patterns_bet_tags) > 0:
                    self.bet_vector = self.construct_pattern_vector(patterns_bet_tags, config)
                elif len(self.bet) > 0:
                    self.bet_vector = self.construct_words_vectors(self.bet, config)

                # AFT context
                if len(self.aft) > 0:
                    self.aft_vector = self.construct_words_vectors(self.aft, config)

                """
                if self.debug is True:
                    print self.bef_vector
                    print self.bet_vector
                    print self.aft_vector
                """

            """
            print self.e1
            print self.e2
            print self.sentence
            print "BEF", self.bef
            print "BET", self.bet
            print "AFT", self.aft
            #print "BEF_vector", self.bef_vector
            #print "BET_vector", self.bet_vector
            #print "AFT_vector", self.aft_vector
            #print "all_words:", all_words
            #print "pattern:", pattern
            print "embeddings:", config.embeddings
            print "ReVerb Patterns BEF:", len(patterns_bef_tags)
            print "ReVerb Patterns BET:", len(patterns_bet_tags)
            print "ReVerb Patterns AFT:", len(patterns_aft_tags)
            if len(patterns_bef_tags) > 0:
                print patterns_bef_tags[0]
            if len(patterns_bet_tags) > 0:
                print patterns_bet_tags[0]
            if len(patterns_aft_tags) > 0:
                print patterns_aft_tags[0]
            #print "vector:", self.vector
            print "\n"
            """
