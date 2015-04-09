# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

from nltk import PunktWordTokenizer, pos_tag
from Word2VecWrapper import Word2VecWrapper
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
            self.vector = None
            self.passive_voice = False
            self.patterns_vectors = list()
            self.patterns_words = list()
            self.extract_patterns(self, config)

            # TODO: Em lugar de somares os vectores, penso que pode ser mais interessante fazeres a média.
            # Lembra-te que se somares, os teus vectores passam a estar influenciados também pelo tamanho dos padrões,
            # em termos do número de frases (e.g., vais ter scores mais altos em todas as dimensões no caso de padrões
            # com mais palavras). A similaridade do cosseno não se altera com isto, mas o produto interno sim.

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
            pattern = [t[0] for t in pattern_tags[0] if t[0].lower() not in config.stopwords and t[1] not in self.filter_pos]
            if len(pattern) >= 1:
                if self.config.embeddings == 'average':
                    pattern_vector = Word2VecWrapper.pattern2vector_average(pattern, config)
                elif self.config.embeddings == 'sum':
                    pattern_vector = Word2VecWrapper.pattern2vector_sum(pattern, config)
                return pattern_vector

        def construct_words_vectors(self, words, config):
            # split text into tokens and tag them using NLTK's default English tagger
            # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
            text_tokens = PunktWordTokenizer().tokenize(words)
            tags_ptb = pos_tag(text_tokens)
            pattern = [t[0] for t in tags_ptb if t[0].lower() not in config.stopwords and t[1] not in self.filter_pos]
            if len(pattern) >= 1:
                if self.config.embeddings == 'average':
                    words_vector = Word2VecWrapper.pattern2vector_average(pattern, config)
                elif self.config.embeddings == 'sum':
                    words_vector = Word2VecWrapper.pattern2vector_sum(pattern, config)
                return words_vector

        @staticmethod
        def extract_patterns(self, config):
            """ Extract ReVerb patterns and construct Word2Vec representations """
            patterns_bef, patterns_bef_tags = Reverb.extract_reverb_patterns_ptb(self.bef)
            patterns_bet, patterns_bet_tags = Reverb.extract_reverb_patterns_ptb(self.bet)
            patterns_aft, patterns_aft_tags = Reverb.extract_reverb_patterns_ptb(self.aft)

            # detect passive voice in BET ReVerb pattern
            if len(patterns_bet_tags) > 0:
                self.passive_voice = self.detect_passive_voice(config, patterns_bet_tags[0])

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
                ##########################################################
                # Version 2: ReVerb Pattern/Words in BEF, BET, and AFT
                ##########################################################

                all_words = list()

                # BEF context
                # first check if at least one pattern was found
                if len(patterns_bef_tags) > 0:
                    for e in patterns_bef_tags[0]:
                        all_words.append(e)
                else:
                    text_tokens = PunktWordTokenizer().tokenize(self.bef)
                    if len(text_tokens) >= 1:
                        tags_ptb = pos_tag(text_tokens)
                        for e in tags_ptb:
                            all_words.append(e)

                # BET context
                if len(patterns_bet_tags) > 0:
                    for e in patterns_bet_tags[0]:
                        all_words.append(e)
                else:
                    text_tokens = PunktWordTokenizer().tokenize(self.bet)
                    if len(text_tokens) >= 1:
                        tags_ptb = pos_tag(text_tokens)
                        for e in tags_ptb:
                            all_words.append(e)

                # AFT context
                if len(patterns_aft_tags) > 0:
                    for e in patterns_aft_tags[0]:
                        all_words.append(e)
                else:
                    text_tokens = PunktWordTokenizer().tokenize(self.aft)
                    if len(text_tokens) >= 1:
                        tags_ptb = pos_tag(text_tokens)
                        for e in tags_ptb:
                            all_words.append(e)

                # normalize: discard adjectives and aux verbs and construct a single vector representation
                pattern = [t[0] for t in all_words if t[0].lower() not in config.stopwords and t[1] not in self.filter_pos]
                if len(pattern) >= 1:
                    words_vector = None
                    if config.embeddings == 'average':
                        words_vector = Word2VecWrapper.pattern2vector_average(pattern, config)
                    elif config.embeddings == 'sum':
                        words_vector = Word2VecWrapper.pattern2vector_sum(pattern, config)
                    self.vector = words_vector

            """
            print self.e1
            print self.e2
            print self.sentence
            print "BEF", self.bef
            print "BET", self.bet
            print "AFT", self.aft
            print "all_words:", all_words
            print "pattern:", pattern
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
            print "vector:", self.vector
            print "\n"
            """