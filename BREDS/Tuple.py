# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

from nltk.stem.wordnet import WordNetLemmatizer
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
            self.passive_voice = False
            self.patterns_vectors = list()
            self.patterns_words = list()
            self.extract_patterns(self, config)

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
        def extract_patterns(self, config):

            # http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html
            # select everything except stopwords and ADJ, ADV
            filter_pos = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB']
            aux_verbs = ['be']

            """ Extract ReVerb patterns and construct Word2Vec representations """
            patterns_bet, patterns_bet_tags = Reverb.extract_reverb_patterns_ptb(self.bet)

            # detect passive voice in ReVerb patterns
            if len(patterns_bet_tags) > 0:
                for pattern in patterns_bet_tags:
                    for i in range(0, len(pattern)):
                        #TODO: contar com adjectivos pelo meio
                        if pattern[i][1].startswith('V'):
                            verb = config.lmtzr.lemmatize(pattern[i][0], 'v')
                            if verb in aux_verbs and i+2 <= len(pattern)-1:
                                if (pattern[i+1][1] == 'VBN' or pattern[i+1][1] == 'VBD') and pattern[i+2][0] == 'by':
                                    self.passive_voice = True

            # construct a word2vec representation
            if len(patterns_bet) > 0:
                self.patterns_words = patterns_bet
                pattern = [t[0] for t in patterns_bet_tags[0] if t[0].lower() not in config.stopwords and t[1] not in filter_pos]
                if len(pattern) >= 1:
                    pattern_vector_bet = Word2VecWrapper.pattern2vector(pattern, config)
                    self.patterns_vectors.append(pattern_vector_bet)

            else:
                """ If no ReVerb patterns are found extract words from context """
                # split text into tokens
                text_tokens = PunktWordTokenizer().tokenize(self.bet)
                # tag the sentence, using the default NTLK English tagger
                # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
                tags_ptb = pos_tag(text_tokens)
                pattern = [t[0] for t in tags_ptb if t[0].lower() not in config.stopwords and t[1] not in filter_pos]
                if len(pattern) >= 1:
                    pattern_vector_bet = Word2VecWrapper.pattern2vector(pattern, config)
                    self.patterns_vectors.append(pattern_vector_bet)
                    self.patterns_words = pattern



