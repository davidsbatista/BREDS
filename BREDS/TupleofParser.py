#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import sys
import numpy as np

from nltk import word_tokenize


class TupleOfParser(object):
        # http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html
        # select everything except stopwords, ADJ and ADV
        filter_pos = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB']

        def __init__(self, _dependency_tree, _e1, _e2, _sentence, config):
            self.e1 = _e1
            self.e2 = _e2
            self.sentence = _sentence
            self.confidence = 0
            self.confidence_old = 0
            self.passive_voice = False
            self.dependencies = _dependency_tree
            self.head_e1 = None
            self.head_e2 = None
            self.deps_path = None
            self.matrix = None
            self.features = None
            self.generate_fcm_embedding(config)

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

        def find_index_named_entity(self, entity, dependencies):

            # split the entity into tokens
            e1_tokens = word_tokenize(entity)

            if e1_tokens[-1] == ".":
                e1_tokens = [entity]

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
            try:
                return idx
            except UnboundLocalError:
                print entity
                print e1_tokens
                for t in dependencies:
                    print t
                print self.sentence
                sys.exit(0)

        def get_heads(self, dependencies, token, heads):
            if dependencies[token.index-1].head == 0:
                heads.append(dependencies[token.index-1])
                return heads
            else:
                head_idx = dependencies[token.index-1].head
                heads.append(dependencies[head_idx-1])
                head_index = dependencies[token.index-1].head-1
                self.get_heads(dependencies, dependencies[head_index], heads)

        def extract_shortest_dependency_path(self, config):
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
            tokens = list()
            for t in self.dependencies:
                tokens.append(t.form)
            try:
                assert len(tokens) == len(self.dependencies)
            except AssertionError:
                print self.sentence
                print tokens
                for t in self.dependencies:
                    print t
                sys.exit(0)

            # find start and end indexes for named-entities
            # TODO: this can be done much quickly by looking at the Tree structure
            e1_tokens = word_tokenize(self.e1)
            e2_tokens = word_tokenize(self.e2)

            if e1_tokens[-1] == ".":
                e1_tokens = [self.e1]

            if e1_tokens[-1] == ".":
                e1_tokens = [self.e2]

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
                        try:
                            features["left_context_e1"] = config.dictionary.token2id[tokens[pos_ent1_bgn-1]]
                        except KeyError:
                            #print tokens[pos_ent1_bgn-1]
                            pass
                    if pos_ent1_end+1 < len(tokens):
                        try:
                            features["right_context_e1"] = config.dictionary.token2id[tokens[pos_ent1_end+1]]
                        except KeyError:
                            #print tokens[pos_ent1_end+1]
                            pass

                if w == pos_ent2_bgn or w == pos_ent2_end:
                    if w-1 > 0:
                        try:
                            features["left_context_e2"] = config.dictionary.token2id[tokens[pos_ent2_bgn-1]]
                        except KeyError:
                            #print tokens[pos_ent2_bgn-1]
                            pass
                    if pos_ent2_end+1 < len(tokens):
                        try:
                            features["right_context_e2"] = config.dictionary.token2id[tokens[pos_ent2_end+1]]
                        except KeyError:
                            #print tokens[pos_ent2_end+1]
                            pass

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
            self.deps_path = self.extract_shortest_dependency_path(config)
            sentence_matrix = self.build_matrix(config)
            self.matrix = sentence_matrix
