# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

from nltk.stem.wordnet import WordNetLemmatizer
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
            self.bef_vector
            self.bet_vector
            self.aft_vector
            self.passive_voice = False
            self.patterns_vectors = list()
            self.patterns_words = list()
            self.extract_patterns(self, config)

            # TODO: e outro teste em que usas a média dos embeddings das palavras que estão no padrão de ReVerb.
            # TODO: combinar as duas coisas, usando as palavras antes e depois em conjunto com o padrão de Reverb.

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
                pattern_vector = Word2VecWrapper.pattern2vector(pattern, config)
                return pattern_vector

        def construct_words_vectors(self, words, config):
            # split text into tokens and tag them using NLTK's default English tagger
            # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
            text_tokens = PunktWordTokenizer().tokenize(words)
            tags_ptb = pos_tag(text_tokens)
            pattern = [t[0] for t in tags_ptb if t[0].lower() not in config.stopwords and t[1] not in self.filter_pos]
            if len(pattern) >= 1:
                words_vector = Word2VecWrapper.pattern2vector(pattern, config)
                return words_vector

        @staticmethod
        def extract_patterns(self, config):
            """ Extract ReVerb patterns and construct Word2Vec representations """
            patterns_bef, patterns_bef_tags = Reverb.extract_reverb_patterns_ptb(self.bef_words)
            patterns_bet, patterns_bet_tags = Reverb.extract_reverb_patterns_ptb(self.bet_words)
            patterns_aft, patterns_aft_tags = Reverb.extract_reverb_patterns_ptb(self.aft_words)

            # detect passive voice in BET ReVerb pattern
            if len(patterns_bet_tags) > 0:
                self.passive_voice = self.detect_passive_voice(config, patterns_bet_tags[0])

            # Construct word2vec representation of the patterns/words

            ##########################################################
            # Version 1: just a single vector with the sum of the BET ReVerb pattern/Words
            ##########################################################

            # BET context
            if len(patterns_bet_tags) > 0:
                self.bet_vector = self.construct_pattern_vector(patterns_bet_tags, config)
            else:
                self.bet_vector = self.construct_words_vectors(self.bet_words, config)

            ########################################################################################
            # TODO: Version 2: just a single vector with the average of the BET ReVerb pattern/Words
            ########################################################################################



            ##########################################################
            # TODO: Version 3: average of the words in BEF, BET, and AFT
            ##########################################################

            # BEF context
            if len(patterns_bef_tags) > 0:
                self.bef_vector = self.construct_pattern_vector(patterns_bef_tags, config)
            else:
                self.bef_vector = self.construct_words_vectors(self.bef_words, config)

            # BET context
            if len(patterns_bet_tags) > 0:
                self.bet_vector = self.construct_pattern_vector(patterns_bet_tags, config)
            else:
                self.bet_vector = self.construct_words_vectors(self.bet_words, config)

            # AFT context
            if len(patterns_aft_tags) > 0:
                self.aft_vector = self.construct_pattern_vector(patterns_aft_tags, config)
            else:
                self.aft_vector = self.construct_words_vectors(self.aft_words, config)

            ##########################################################
            # TODO: Version 4: combine Version 2 and 3
            ##########################################################


            """
            print self.e1
            print self.e2
            print self.sentence
            print "ReVerb Patterns BEF:", len(patterns_bef_tags)
            print "ReVerb Patterns BET:", len(patterns_bet_tags)
            print "ReVerb Patterns AFT:", len(patterns_aft_tags)
            if len(patterns_bef_tags) > 0:
                print patterns_bef_tags[0]
            if len(patterns_bet_tags) > 0:
                print patterns_bet_tags[0]
            if len(patterns_aft_tags) > 0:
                print patterns_aft_tags[0]
            print "BEF:", self.bef_vector
            print "BET:", self.bet_vector
            print "AFT:", self.aft_vector
            """