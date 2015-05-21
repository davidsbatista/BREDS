#!/usr/bin/env python
# -*- coding: utf-8 -
import random

__author__ = 'dsbatista'
__email__ = "dsbatista@inesc-id.pt"

import fileinput
import os
import re
import sys
import graphviz
import numpy as np
import StanfordDependencies

from gensim.models import Word2Vec
from nltk import word_tokenize
from nltk.parse.stanford import StanfordParser

entities_regex = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)
tags_regex = re.compile('</?[A-Z]+>', re.U)
e_types = {'<ORG>': 3, '<LOC>': 4, '<PER>': 5}


class Relationship(object):

    def __init__(self, _sentence, _ent1, _ent2, _e1_type, _e2_type):
        self.sentence = _sentence
        self.ent1 = _ent1
        self.ent2 = _ent2
        self.e1_type = _e1_type
        self.e2_type = _e2_type
        self.head_e1 = None
        self.head_e2 = None
        self.pos_e1 = None
        self.pos_e2 = None
        self.dependencies = None
        self.dep_path = None
        self.matrix = None


def extract_features_word(rel, vocabulary):
    """
    :param rel: a relationship
    :return: a matrix representing the relationship sentence
    """

    word_matrixes = list()

    if rel.e1_type == rel.e2_type:
        same = 1

    sentence = re.sub(tags_regex, "", rel.sentence)
    tokens = word_tokenize(sentence)

    assert len(tokens) == len(rel.dependencies)

    """
    pos_ent1_bgn = 0
    pos_ent1_end = 0
    pos_ent2_bgn = 0
    pos_ent2_end = 0
    """

    # find start and end indexes for named-entities
    e1_tokens = word_tokenize(rel.ent1)
    e2_tokens = word_tokenize(rel.ent2)

    if len(e1_tokens) == 1:
        pos_ent1_bgn = tokens.index(rel.ent1)
        pos_ent1_end = tokens.index(rel.ent1)

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
            print "E1", rel.ent1, "not found"
            sys.exit(0)

    if len(e2_tokens) == 1:
        pos_ent2_bgn = tokens.index(rel.ent2)
        pos_ent2_end = tokens.index(rel.ent2)

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
            print "E2", rel.ent1, "not found"
            sys.exit(0)

    # start feature extraction
    for w in range(len(tokens)):
        features = dict()
        features["head_emb"] = 0
        features["head_emb_h1:"+rel.e1_type] = 0
        features["head_emb_h2:"+rel.e2_type] = 0
        features["head_emb_h1_h2:"+rel.e1_type+"_"+rel.e1_type] = 0

        features["on-path"] = 0
        features["on-path:"+rel.e1_type] = 0
        features["on-path:"+rel.e2_type] = 0
        features["on-path_h1_h2:"+rel.e1_type+"_"+rel.e1_type] = 0

        features["in-between"] = 0
        features["in-between:"+rel.e1_type] = 0
        features["in-between:"+rel.e2_type] = 0
        features["in-between_h1_h2:"+rel.e1_type+"_"+rel.e1_type] = 0

        features["left_context_e1"] = 0
        features["right_context_e1"] = 0
        features["left_context_e2"] = 0
        features["right_context_e2"] = 0

        #################################################
        # extract features that depend on the parse tree
        #################################################

        # wether the word is the head entity
        if rel.dependencies[w] == rel.dependencies[rel.head_e1-1]:
            features["head_emb"] = 1
            features["head_emb_h1:"+rel.e1_type] = 1

        if rel.dependencies[w] == rel.dependencies[rel.head_e2-1]:
            features["head_emb"] = 1
            features["head_emb_h1:"+rel.e2_type] = 1

        # whether the word is on the path between the two entities
        if rel.dependencies[w] in rel.dep_path:
            features["on-path"] = 0
            features["on-path:"+rel.e1_type] = 0
            features["on-path:"+rel.e2_type] = 0
            features["on-path_h1_h2:"+rel.e1_type+"_"+rel.e1_type] = 0

        ##########################
        # extract local features
        ##########################
        # in-between
        if pos_ent1_end < w < pos_ent2_bgn:
            features["in-between"] = 1
            features["in-between:"+rel.e1_type] = 1
            features["in-between:"+rel.e2_type] = 1
            features["in-between_h1_h2:"+rel.e1_type+"_"+rel.e1_type] = 1

        # context
        if w == pos_ent1_bgn or w == pos_ent1_end:
            if w-1 > 0:
                features["left_context_e1"] = vocabulary[tokens[pos_ent1_bgn-1]]
            if pos_ent1_end+1 < len(tokens):
                features["right_context_e1"] = vocabulary[tokens[pos_ent1_end+1]]

        if w == pos_ent2_bgn or w == pos_ent2_end:
            if w-1 > 0:
                features["left_context_e2"] = vocabulary[tokens[pos_ent2_bgn-1]]
            if pos_ent2_end+1 < len(tokens):
                features["right_context_e2"] = vocabulary[tokens[pos_ent2_end+1]]

        """
        print tokens[w]
        for feature in features:
            print feature, features[feature]
        """

        lexical_context_vector = [features["left_context_e1"], features["right_context_e1"], features["left_context_e2"], features["right_context_e2"]]
        in_between_vector = [features["in-between"], features["in-between:"+rel.e1_type], features["in-between:"+rel.e2_type], features["in-between_h1_h2:"+rel.e1_type+"_"+rel.e1_type]]
        on_path_vector = [features["on-path"], features["on-path:"+rel.e1_type], features["on-path:"+rel.e2_type], features["on-path_h1_h2:"+rel.e1_type+"_"+rel.e1_type]]
        head_emb_vector = [features["head_emb"], features["head_emb_h1:"+rel.e1_type], features["head_emb"], features["head_emb_h1:"+rel.e2_type]]
        feature_vector = np.array(on_path_vector + head_emb_vector + in_between_vector + lexical_context_vector)

        try:
            # outer vector
            outer = np.outer(feature_vector, word2vec[tokens[w].lower()])
            word_matrixes.append(outer)

        except KeyError, e:
            pass
            #print e
            #print tokens[w].lower()

    # add every matrix and return the sum
    matrix_acc = np.zeros_like(word_matrixes[0])
    for m in word_matrixes:
        np.add(matrix_acc, m, matrix_acc)

    # simple normalization, divide each element by the maximum
    final_matrix = np.divide(matrix_acc, matrix_acc.max())

    return final_matrix


def find_index_named_entity(entity, dependencies):
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
                """
                print "len(e1_tokens)", len(e1_tokens)
                print "ent1", i, e1_tokens[i]
                print "token.form", j, rel.dependencies[j].form
                """
                while (i + 1 < len(e1_tokens)) and e1_tokens[i + 1] == dependencies[j + 1].form:
                    """
                    print "entrei"
                    print "ent1", i, e1_tokens[i]
                    print "token.form", j, rel.dependencies[j].form
                    """
                    i += 1
                    j += 1
                    """
                    print "i", i
                    print "j", j
                    """

                # if all the sequente tokens are equal to the tokens in the named-entity
                # then set the last one has the index
                if i + 1 == len(e1_tokens):
                    idx = j+1
    return idx


def get_heads(dependencies, token, heads):
    if dependencies[token.index-1].head == 0:
        heads.append(dependencies[token.index-1])
        return heads
    else:
        head_idx = dependencies[token.index-1].head
        heads.append(dependencies[head_idx-1])
        head_index = dependencies[token.index-1].head-1
        get_heads(dependencies, dependencies[head_index], heads)


def extract_shortest_dependency_path(rel):
    # get position of entity and entity in tree
    idx1 = find_index_named_entity(rel.ent1, rel.dependencies)
    idx2 = find_index_named_entity(rel.ent2, rel.dependencies)

    """
    print "e1", idx1
    print "e2", idx2
    print "ent1: ", rel.ent1
    print "ent2: ", rel.ent2
    """

    shortest_path = list()

    heads_e1 = list()
    get_heads(rel.dependencies, rel.dependencies[idx1-1], heads_e1)

    heads_e2 = list()
    get_heads(rel.dependencies, rel.dependencies[idx2-1], heads_e2)

    e1 = rel.dependencies[idx1-1]
    e2 = rel.dependencies[idx2-1]

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
            if t != heads_e1[index_t1] and t != rel.dependencies[idx2-1]:
                #print t.form+"->",
                shortest_path.append(t)
            else:
                #print t.form
                shortest_path.append(t)
                break

        #print rel.ent2+"->",
        for t in heads_e2:
            if t == rel.dependencies[idx1-1]:
                break
            elif t != heads_e2[index_t2]:
                #print t.form+"->",
                shortest_path.append(t)
            else:
                #print t.form
                #shortest_path.append(t)
                break

    return shortest_path


def generate_relationships(parser, sd, examples, words):
    count = 1
    for rel in examples:
        sentence = re.sub(tags_regex, "", rel.sentence)
        t = parser.raw_parse(sentence)
        # draws the consituients tree
        # t[0].draw()

        # note: http://www.nltk.org/_modules/nltk/parse/stanford.html
        # the wrapper for StanfordParser does not give syntatic dependencies
        tree_deps = sd.convert_tree(str(t[0]))
        rel.dependencies = tree_deps

        idx1 = find_index_named_entity(rel.ent1, rel.dependencies)
        idx2 = find_index_named_entity(rel.ent2, rel.dependencies)
        rel.head_e1 = idx1
        rel.head_e2 = idx2

        deps = extract_shortest_dependency_path(rel)
        rel.dep_path = deps

        sentence_matrix = extract_features_word(rel, words)
        rel.matrix = sentence_matrix

        # renders a PDF by default
        """
        dotgraph = tree_deps.as_dotgraph()
        dotgraph.format = 'svg'
        dotgraph.render('file_'+str(count))
        """
        count += 1


def generate_vocabulary(count, examples, words):
    for rel in examples:
        sentence = re.sub(tags_regex, "", rel.sentence)
        tokens = word_tokenize(sentence)
        for w in tokens:
            if w not in words:
                words[w] = count
                count += 1


def parse_files(data_file):
    examples = list()
    for line in fileinput.input(data_file):
        if line.startswith("sentence: "):
            sentence = line.split("sentence: ")[1].strip()
            matches = []
            for m in re.finditer(entities_regex, sentence):
                matches.append(m.group())
            entity_1 = re.sub("</?[A-Z]+>", "", matches[0])
            entity_2 = re.sub("</?[A-Z]+>", "", matches[1])
            arg1 = re.search("</?[A-Z]+>", matches[0])
            arg2 = re.search("</?[A-Z]+>", matches[1])
            examples.append(Relationship(sentence, entity_1, entity_2, arg1.group(), arg2.group()))
    fileinput.close()
    return examples


def sample(examples, n):
    samples = list()
    for i in range(0, n):
        e = random.choice(examples)
        samples.append(e)
        del examples[examples.index(e)]
    return samples


################################
# distances between matrix norms
################################

# http://math.stackexchange.com/questions/507742/distance-similarity-between-two-matrices?rq=1
# I'm sure there are many others. If you look up "matrix norms", you'll find lots of material.
# And if ∥∥ is any matrix norm, then ∥A−B∥ gives you a measure of the "distance" between two matrices A and B.

def sim_matrix(matrix1, matrix2):
    assert matrix1.shape == matrix2.shape
    diff_matrix = matrix1-matrix2
    diff_matrix_abs = np.absolute(diff_matrix)
    sum_abs_diff = np.sum(diff_matrix_abs)
    return sum_abs_diff


def sim_matrix_l2(matrix1, matrix2):
    diff_matrix = matrix1-matrix2
    diff_matrix = np.power(diff_matrix, 2)
    sum_diff = np.sum(diff_matrix)
    return np.sqr(sum_diff)


def distance_general(a, b, norm_type):
    # ‘fro’ 	Frobenius norm
    # 2 	2-norm (largest sing. value)
    # -2 	smallest singular value
    distance_general(a, b, 'fro')
    distance_general(a, b, '2')
    distance_general(a, b, '-22')
    norm_a = np.linalg.norm(a, ord=norm_type)
    norm_b = np.linalg.norm(b, ord=norm_type)
    return norm_a - norm_b


def main():
    model = "/home/dsbatista/gigaword/word2vec/afp_apw_xing200.bin"
    print "Loading word2vec model"
    global word2vec
    word2vec = Word2Vec.load_word2vec_format(model, binary=True)
    global VECTOR_DIM
    VECTOR_DIM = word2vec.layer1_size

    # JAVA_HOME needs to be set, calling 'java -version' should show: java version "1.8.0_45" or higher
    # PARSER and STANFORD_MODELS enviroment variables need to be set
    os.environ['STANFORD_PARSER'] = '/home/dsbatista/stanford-parser-full-2015-04-20/'
    os.environ['STANFORD_MODELS'] = '/home/dsbatista/stanford-parser-full-2015-04-20/'
    parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
    sd = StanfordDependencies.get_instance(backend='subprocess', jar_filename='/home/dsbatista/stanford-parser-full-2015-04-20/stanford-parser.jar')

    positive_examples = parse_files(sys.argv[1])
    negative_examples = parse_files(sys.argv[2])

    # first pass to generate indices for each word in the vocabulary
    vocabulary_words = dict()
    count = 1
    generate_vocabulary(count, positive_examples, vocabulary_words)
    generate_vocabulary(count, negative_examples, vocabulary_words)
    print len(vocabulary_words), "words"

    # generate relationship matrix representations
    print "Processing positive sentences"
    generate_relationships(parser, sd, positive_examples, vocabulary_words)
    print len(positive_examples), "positive sentences processed\n"
    print "Processing negative sentences"
    generate_relationships(parser, sd, negative_examples, vocabulary_words)
    print len(negative_examples), "negative sentences processed\n"

    correct_s = sample(positive_examples, 26)
    incorrect_s = sample(negative_examples, 26)
    positive = 0
    negative = 0
    for rel in correct_s:
        min_distance = sys.maxint
        closest_sentence = None
        closest_sentence_type = 0
        print "sentence:", rel.sentence

        for rel1 in incorrect_s:
            if rel == rel1:
                continue
            assert rel.matrix.shape == rel1.matrix.shape
            dist_with_neg = sim_matrix(rel.matrix, rel1.matrix)
            if dist_with_neg < min_distance:
                print dist_with_neg
                min_distance = dist_with_neg
                closest_sentence = rel1
                closest_sentence_type = -1

        for rel2 in correct_s:
            if rel == rel2:
                continue
            assert rel.matrix.shape == rel2.matrix.shape
            dist_with_pos = sim_matrix(rel.matrix, rel2.matrix)
            if dist_with_pos < min_distance:
                print dist_with_pos
                min_distance = dist_with_pos
                closest_sentence = rel2
                closest_sentence_type = 1

        if closest_sentence_type == -1:
            negative += 1
        else:
            positive += 1

        print "closest sentence:"
        print closest_sentence.sentence
        print closest_sentence_type
        print "====================================\n"

    print "closest to negative", negative
    print "closest to positive", positive

if __name__ == "__main__":
    main()














    #examples.append(Relationship("In favour of the deal were Ted Turner, founder and boss of TBS, a cable-based business, and Time Warner chairman Gerald Levin, whose empire already held an 18 percent stake in TBS.", "Ted Turner", "TBS"))
    """
    # negative - no relationship
    examples.append(Relationship("Anthony Shadid is an Associated Press newsman based in Cairo .", "Associated Press", "Cairo"))
    examples.append(Relationship("The timing of Merrill 's investment enabled Enron to book sales income of 12 million dollars in its 1999 financial statements for its African division .", "Merrill", "Enron"))
    examples.append(Relationship("Dan was not born in Lisbon", "Dan", "Lisbon"))

    # positive - affilation
    examples.append(Relationship("Amazon.com chief executive Jeff Bezos", "Amazon.com", "Jeff Bezos"))
    examples.append(Relationship("Bob is a history professor at Stanford", "Bob", "Stanford"))
    examples.append(Relationship("One document _ a handwritten note at the bottom of a Dec. 12 , 1999 , fax by Merrill Lynch 's senior finance chief James Brown _ questioned whether there would be a `` reputational risk '' if the firm helped `` aid/abet Enron income stmt manipulation . ''", "James Brown", "Merrill Lynch"))
    examples.append(Relationship("Mary Ann Glendon, a professor at Harvard University, will be the first woman to lead a Holy See delegation to an international conference , the Vatican announced Friday.", "Mary Ann Glendon", "Harvard University"))

    # positive - headquarters
    examples.append(Relationship("For KFC, whose Louisville headquarters resemble a white-columned mansion, the recipe is more than a treasured link to its roots", "KFC", "Louisville"))
    examples.append(Relationship("Google based in Mountain View, California.", "Google", "California"))

    # positive - study-at
    examples.append(Relationship("Bob studied history at Stanford", "Bob", "Stanford"))
    examples.append(Relationship("Bob studied journalism at Stanford", "Bob", "Stanford"))
    examples.append(Relationship("Bob studied journalism at Stanford, and is currently working for Microsoft", "Bob", "Stanford"))

    # positive - acquired
    examples.append(Relationship("Turner sold CNN and his other broadcasting operations to Time Warner in 1996 , but his operational role has been minimized following Time Warner 's takeover by AOL .", "AOL", "Time Warner"))
    examples.append(Relationship("NBC is currently a poor third in the ratings of the four US networks , having slipped from first place when General Electric acquired it from RCA in 1985 .", "General Electric", "NBC"))
    examples.append(Relationship("The new channel will focus on `` people , contemporary and historical , in and out of the headlines '' according to Westinghouse Electric , the company that acquired CBS last year for 5.4 billion dollars .", "Westinghouse Electric", "CBS"))
    examples.append(Relationship("Including the roughly 1,500 workers picked up in the DoubleClick acquisition , Google now has more than 18,000 employees worldwide .", "Google", "DoubleClick"))
    examples.append(Relationship("The story was first seen at Techcrunch , the picked up by the Wall Street Journal and has since been the subject of much talk , posts and thoughts over the past few days and finally it has been confirmed that Google have purchased Youtube for $ 1.65 billion in an official statement .","Google", "Youtube"))

    examples.append(Relationship("Bob is a history professor at Stanford", "Bob", "Stanford"))
    examples.append(Relationship("Bob studied journalism at Stanford, and is currently working for Microsoft", "Bob", "Stanford"))
    """
    """
    examples.append(Relationship("Amazon.com founder and chief executive Jeff Bezos, said that he is happy to announce new gains.", "Amazon.com", "Jeff Bezos"))
    examples.append(Relationship("But Richard Klein, a paleoanthropology professor at Stanford University , said the evidence is `` pretty sparse . ''", "Richard Klein", "Stanford University"))
    examples.append(Relationship("Protesters seized several pumping stations, holding 127 Shell workers hostage", "Protesters", "stations"))
    examples.append(Relationship("Protesters seized several pumping stations, holding 127 Shell workers hostage", "workers", "stations"))
    examples.append(Relationship("Troops recently have raided churches, warning ministers to stop preaching", "Troops", "churches"))
    examples.append(Relationship("Troops recently have raided churches, warning ministers to stop preaching", "ministers", "churches"))
    """

    """
    for line in fileinput.input("golden_standard/acquired_negative.txt"):
        if line.startswith("sentence: "):
            sentence = line.split("sentence: ")[1].strip()
            matches = []
            for m in re.finditer(entities_regex, sentence):
                matches.append(m.group())

            entity_1 = re.sub("</?[A-Z]+>", "", matches[0])
            entity_2 = re.sub("</?[A-Z]+>", "", matches[1])
            arg1 = re.search("</?[A-Z]+>", matches[0])
            arg2 = re.search("</?[A-Z]+>", matches[1])

            examples.append(Relationship(sentence, entity_1, entity_2, arg1.group(), arg2.group()))
    """