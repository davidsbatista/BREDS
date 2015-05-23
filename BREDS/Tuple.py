#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import re
import numpy as np

from nltk import pos_tag
from nltk import word_tokenize
from Common.ReVerb import Reverb


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
            self.dependencies = None
            self.head_e1 = None
            self.head_e2 = None
            self.deps_path = None
            self.matrix = None
            self.featues = None

            if config.embeddings == 'fcm':
                self.generate_fcm_embedding(config)

            elif config.embeddings == 'sum':
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

        def construct_pattern_vector(self, pattern_tags, config):
            # remove stopwords and adjectives
            pattern = [t[0] for t in pattern_tags if t[0].lower() not in config.stopwords and t[1] not in self.filter_pos]

            if len(pattern) >= 1:
                if config.embeddings == 'average':
                    pattern_vector = config.word2vecwrapper.pattern2vector_average(pattern, config)
                elif config.embeddings == 'sum':
                    pattern_vector = config.word2vecwrapper.pattern2vector_sum(pattern, config)

                return pattern_vector

        def construct_words_vectors(self, words, config):
            # split text into tokens and tag them using NLTK's default English tagger
            # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
            text_tokens = word_tokenize(words)
            tags_ptb = pos_tag(text_tokens)

            pattern = [t[0] for t in tags_ptb if t[0].lower() not in config.stopwords and t[1] not in self.filter_pos]
            if len(pattern) >= 1:
                if config.embeddings == 'average':
                    words_vector = config.word2vecwrapper.pattern2vector_average(pattern, config)
                elif config.embeddings == 'sum':
                    words_vector = config.word2vecwrapper.pattern2vector_sum(pattern, config)

                return words_vector

        def extract_patterns(self, config):

            # extract ReVerb pattern and detect the presence of the passive voice
            patterns_bet_tags = Reverb.extract_reverb_patterns_ptb(self.bet_words)
            if len(patterns_bet_tags) > 0:
                self.passive_voice = self.config.reverb.detect_passive_voice(patterns_bet_tags)
                # forced hack since _'s_ is always tagged as VBZ, (u"'s", 'VBZ') and causes ReVerb to identify
                # a pattern which is wrong, if this happens, ignore that a pattern was extracted
                if patterns_bet_tags[0][0] == "'s":
                    self.bet_vector = self.construct_words_vectors(self.bet_words, config)
                else:
                    self.bet_vector = self.construct_pattern_vector(patterns_bet_tags, config)
            else:
                self.bet_vector = self.construct_words_vectors(self.bet_words, config)

            # extract two words before the first entity, and two words after the second entity
            if len(self.bef_words) > 0:
                self.bef_vector = self.construct_words_vectors(self.bef_words, config)

            if len(self.aft_words) > 0:
                self.aft_vector = self.construct_words_vectors(self.aft_words, config)

        def find_index_named_entity(self, entity, dependencies):
            # split the entity into tokens
            e1_tokens = word_tokenize(entity)

            # if entities are one token only get entities index directly
            if len(e1_tokens) == 1:
                for token in dependencies:
                    if token.form == entity:
                        idx = token.index

            # if the entities are constituied by more than one token, find first match
            # compare sequentally all matches, if reaches the end of the entity, assume entity was
            # found in the dependencies
            elif len(e1_tokens) > 1:
                for token in dependencies:
                    if token.form == e1_tokens[0]:
                        j = dependencies.index(token)
                        i = 0
                        while (i + 1 < len(e1_tokens)) and e1_tokens[i + 1] == dependencies[j + 1].form:
                            i += 1
                            j += 1

                        # if all the sequente tokens are equal to the tokens in the named-entity
                        # then set the last one has the index
                        if i + 1 == len(e1_tokens):
                            idx = j+1
            return idx

        def get_heads(self, dependencies, token, heads):
            if dependencies[token.index-1].head == 0:
                heads.append(dependencies[token.index-1])
                return heads
            else:
                head_idx = dependencies[token.index-1].head
                heads.append(dependencies[head_idx-1])
                head_index = dependencies[token.index-1].head-1
                self.get_heads(dependencies, dependencies[head_index], heads)

        def extract_shortest_dependency_path(self):

            # get position of entity and entity in tree
            idx1 = self.find_index_named_entity(self.e1, self.dependencies)
            idx2 = self.find_index_named_entity(self.e2, self.dependencies)
            self.head_e1 = idx1
            self.head_e2 = idx2

            shortest_path = list()

            heads_e1 = list()
            self.get_heads(self.dependencies, self.dependencies[idx1-1], heads_e1)

            heads_e2 = list()
            self.get_heads(self.dependencies, self.dependencies[idx2-1], heads_e2)

            e1 = self.dependencies[idx1-1]
            e2 = self.dependencies[idx2-1]

            # check if e2 is parent of e1
            if e2 in heads_e1:
                #print "E2 is parent of E1"
                #print "E2 parents", heads_e1
                #print rel.ent1+"<-",
                for t in heads_e1:
                    if t == e2:
                        #print "<-"+rel.ent2
                        break
                    else:
                        #print t.form+"<-",
                        shortest_path.append(t)

            # check if e1 is parent of e2
            elif e1 in heads_e2:
                #print "E1 is parent of E2"
                #print "E2 parents", heads_e2
                #print rel.ent2+"<-",
                for t in heads_e2:
                    if t == e1:
                        #print rel.ent1
                        break
                    else:
                        #print t.form+"<-",
                        shortest_path.append(t)

            else:
                # find a common parent for both
                #print "E1 and E2 have a common parent"
                found = False
                for t1 in heads_e1:
                    if found is True:
                        break
                    for t2 in heads_e2:
                        if t1 == t2:
                            index_t1 = heads_e1.index(t1)
                            index_t2 = heads_e2.index(t2)
                            found = True
                            break

                #print "\nshortest path: "
               # print rel.ent1+"->",
                for t in heads_e1:
                    if t != heads_e1[index_t1] and t != self.dependencies[idx2-1]:
                        #print t.form+"->",
                        shortest_path.append(t)
                    else:
                        #print t.form
                        shortest_path.append(t)
                        break

                #print rel.ent2+"->",
                for t in heads_e2:
                    if t == self.dependencies[idx1-1]:
                        break
                    elif t != heads_e2[index_t2]:
                        #print t.form+"->",
                        shortest_path.append(t)
                    else:
                        #print t.form
                        #shortest_path.append(t)
                        break

            return shortest_path

        def build_matrix(self, config):

            word_matrixes = list()
            sentence = re.sub(config.tags_regex, "", self.sentence)
            tokens = word_tokenize(sentence)
            assert len(tokens) == len(self.dependencies)

            # find start and end indexes for named-entities
            # TODO: this can be done much quickly by looking at the Tree structure
            e1_tokens = word_tokenize(self.e1)
            e2_tokens = word_tokenize(self.e2)

            if len(e1_tokens) == 1:
                pos_ent1_bgn = tokens.index(self.e1)
                pos_ent1_end = tokens.index(self.e1)

            else:
                pos_ent1_bgn = tokens.index(e1_tokens[0])
                z = pos_ent1_bgn+1
                i = 1
                while z < len(tokens) and i < len(e1_tokens):
                    if tokens[z] != e1_tokens[i]:
                        break
                    else:
                        z += 1
                        i += 1

                if z - pos_ent1_bgn == i:
                    pos_ent1_end = z-1
                else:
                    print "E1", self.e1, "not found"
                    sys.exit(0)

            if len(e2_tokens) == 1:
                pos_ent2_bgn = tokens.index(self.e2)
                pos_ent2_end = tokens.index(self.e2)

            else:
                pos_ent2_bgn = tokens.index(e2_tokens[0])
                z = pos_ent2_bgn+1
                i = 1
                while z < len(tokens) and i < len(e2_tokens):
                    if tokens[z] != e2_tokens[i]:
                        break
                    else:
                        z += 1
                        i += 1

                if z - pos_ent2_bgn == i:
                    pos_ent2_end = z-1
                else:
                    print "E2", self.e1, "not found"
                    sys.exit(0)

            # start feature extraction
            for w in range(len(tokens)):
                features = dict()
                features["left_context_e1"] = 0
                features["right_context_e1"] = 0
                features["left_context_e2"] = 0
                features["right_context_e2"] = 0

                f_types = ['head_emb', 'on-path', 'in-between']

                for f_t in f_types:
                    features[f_t] = 0
                    for e_type1 in config.e_types:
                        f1 = f_t+"_h1:"+e_type1
                        f2 = f_t+"_h2:"+e_type1
                        features[f1] = 0
                        features[f2] = 0
                        for e_type2 in config.e_types:
                            f3 = f_t+"_h1_h2:"+e_type1+"_"+e_type2
                            features[f3] = 0

                #################################################
                # extract features that depend on the parse tree
                #################################################

                # wether the word is the head entity
                if self.dependencies[w] == self.dependencies[self.head_e1-1]:
                    features["head_emb"] = 1
                    features["head_emb_h1:"+config.e1_type] = 1

                if self.dependencies[w] == self.dependencies[self.head_e2-1]:
                    features["head_emb"] = 1
                    features["head_emb_h2:"+config.e2_type] = 1

                # whether the word is on the path between the two entities
                if self.dependencies[w] in self.deps_path:
                    features["on-path"] = 1
                    features["on-path_h1:"+config.e1_type] = 1
                    features["on-path_h2:"+config.e2_type] = 1
                    features["on-path_h1_h2:"+config.e1_type+"_"+config.e2_type] = 1

                ##########################
                # extract local features
                ##########################
                # in-between
                if pos_ent1_end < w < pos_ent2_bgn:
                    features["in-between"] = 1
                    features["in-between_h1:"+config.e1_type] = 1
                    features["in-between_h2:"+config.e2_type] = 1
                    features["in-between_h1_h2:"+config.e1_type+"_"+config.e2_type] = 1

                # context
                if w == pos_ent1_bgn or w == pos_ent1_end:
                    if w-1 > 0:


                        features["left_context_e1"] = config.dictionary.token2id([tokens[pos_ent1_bgn-1]])
                    if pos_ent1_end+1 < len(tokens):
                        features["right_context_e1"] = config.dictionary.token2id([tokens[pos_ent1_end+1]])

                if w == pos_ent2_bgn or w == pos_ent2_end:
                    if w-1 > 0:
                        features["left_context_e2"] = config.dictionary.token2id([tokens[pos_ent2_bgn-1]])
                    if pos_ent2_end+1 < len(tokens):
                        features["right_context_e2"] = config.dictionary.token2id([tokens[pos_ent2_end+1]])

                # add filled features vectors to Relationship instance
                self.features = features

                # transform features into feature vector
                features_keys = sorted(features.keys())
                vector = list()
                for feature in features_keys:
                    vector += [features[feature]]
                feature_vector = np.array(vector)

                # the try/catch below is to avoid the crashing
                # when a word is not found in the word2vec model
                try:
                    # outer vector
                    outer = np.outer(feature_vector, config.word2vec[tokens[w].lower()])
                    word_matrixes.append(outer)
                    # print outer.shape

                except KeyError:
                    pass
                    #print "Not found", tokens[w].lower()

            # add every matrix and return the sum
            final_matrix = np.zeros_like(word_matrixes[0])
            for m in word_matrixes:
                np.add(final_matrix, m, final_matrix)

            # normalization
            final_matrix = np.divide(final_matrix, final_matrix.max())

            return final_matrix

        def generate_fcm_embedding(self, config):
            sentence = re.sub(config.tags_regex, "", self.sentence)
            t = config.parser.raw_parse(sentence)
            # http://www.nltk.org/_modules/nltk/parse/stanford.html
            # note: the wrapper for StanfordParser does not give syntatic dependencies
            # use the StanfordDependencies module
            tree_deps = config.sd.convert_tree(str(t[0]))
            self.dependencies = tree_deps
            self.deps_path = self.extract_shortest_dependency_path()
            sentence_matrix = self.build_matrix(config)
            self.matrix = sentence_matrix
