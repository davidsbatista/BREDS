#!/usr/bin/env python
# -*- coding: utf-8 -

__author__ = 'dsbatista'
__email__ = "dsbatista@inesc-id.pt"

import fileinput
import os
import re
import sys
import xdot
import graphviz
import StanfordDependencies

from nltk.parse.stanford import StanfordParser
from nltk import PunktWordTokenizer
from itertools import product

entities_regex = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)
tags_regex = re.compile('</?[A-Z]+>', re.U)

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


e_types = {'<ORG>': 3, '<LOC>': 4, '<PER>': 5}


def extract_features_word(rel):
    """
    :param rel: a relationship
    :return: a matrix representing the relationship sentence
    """

    if rel.e1_type == rel.e2_type:
        same = 1

    template = [0, e_types[rel.e1_type], e_types[rel.e2_type], same]

    # extract features that depende on the parse tree
    for t in rel.dependencies:
        is_head_1 = 0
        is_head_2 = 0
        on_path = 0

        # wether the word is the head entity
        if t == rel.dependencies[rel.head_e1-1]:
            is_head_1 = 1
        if t == rel.dependencies[rel.head_e2-1]:
            is_head_2 = 1

        #features_head_emb = product([is_head_1, is_head_2], template, repeat=1)
        # whether the word is on the path between the two entities
        if t in rel.dep_path:
            on_path = 1

        # features_on-path = (1) x template
        print t.form, '\t', "on_path:", on_path, '\t', "is_head:", [is_head_1, is_head_2]

    # extract features that depende on context
    sentence = re.sub(tags_regex, "", rel.sentence)
    tokens = PunktWordTokenizer().tokenize(sentence)
    pos_ent1 = 0
    pos_ent2 = 0

    e1_tokens = PunktWordTokenizer().tokenize(rel.ent1)
    e2_tokens = PunktWordTokenizer().tokenize(rel.ent2)

    print e1_tokens
    print e2_tokens
    print tokens

    if len(e1_tokens) == 1:
        pos_ent1 = tokens.index(rel.ent1)

    else:
        print "TODO"

    if len(e2_tokens) == 1:
        pos_ent2 = tokens.index(rel.ent2)

    else:
        print "TODO"

    print pos_ent1
    print pos_ent2

    for w in range(len(tokens)):
        in_between = 0
        context_left_h1 = None
        context_right_h1 = None
        context_left_h2 = None
        context_right_h2 = None

        # in-between
        if pos_ent1 < w < pos_ent2:
            in_between = 1

        # context
        if w == pos_ent1:
            if w-1 > 0:
                context_left_h1 = tokens[w-1]
            if w+1 < len(tokens):
                context_right_h1 = tokens[w+1]

        if w == pos_ent2:
            if w-1 > 0:
                context_left_h2 = tokens[w-1]
            if w+1 < len(tokens):
                context_right_h2 = tokens[w+1]

        print tokens[w], '\t', "in_between:", in_between, '\t', "context:", context_left_h1, context_left_h2, context_right_h1, context_right_h2


def find_index_named_entity(entity, dependencies):
    # split the entity into tokens
    e1_tokens = PunktWordTokenizer().tokenize(entity)

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
        print "E2 is parent of E1"
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
        print "E1 is parent of E2"
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
        print "E1 and E2 have a common parent"
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


def main():
    """
    model = "/home/dsbatista/gigaword/word2vec/afp_apw_xing200.bin"
    print "Loading word2vec model"
    global word2vec
    word2vec = Word2Vec.load_word2vec_format(model, binary=True)
    global VECTOR_DIM
    #VECTOR_DIM = word2vec.layer1_size
    VECTOR_DIM = 200

    reverb = Reverb()
    print "Processing sentences..."
    sentences = read_sentences_bunescu(sys.argv[1])
    for rel_type in sentences.keys():
        print rel_type, len(sentences[rel_type])

    print "Constructing vector representations"
    for rel in sentences.keys():
        print rel in sentences[rel_type]
        construct_vector(rel, reverb)
    """
    # JAVA_HOME needs to be set, calling 'java -version' should show: java version "1.8.0_45" or higher
    # PARSER and STANFORD_MODELS enviroment variables need to be set
    os.environ['STANFORD_PARSER'] = '/home/dsbatista/stanford-parser-full-2015-04-20/'
    os.environ['STANFORD_MODELS'] = '/home/dsbatista/stanford-parser-full-2015-04-20/'
    parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
    sd = StanfordDependencies.get_instance(backend='subprocess', jar_filename='/home/dsbatista/stanford-parser-full-2015-04-20/stanford-parser.jar')


    #examples.append(Relationship("In favour of the deal were Ted Turner, founder and boss of TBS, a cable-based business, and Time Warner chairman Gerald Levin, whose empire already held an 18 percent stake in TBS.", "Ted Turner", "TBS"))

    examples = list()
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

    for line in fileinput.input("golden_standard/acquired_positive.txt"):
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

    count = 1
    for rel in examples:
        sentence = re.sub(tags_regex, "", rel.sentence)
        print count, sentence
        t = parser.raw_parse(sentence)
        # draws the consituients tree
        #t[0].draw()

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

        print rel.ent1
        print rel.ent2
        for t in deps:
            print t

        print "\n\n"

        extract_features_word(rel)

        sys.exit(0)

        # renders a PDF by default
        """
        dotgraph = tree_deps.as_dotgraph()
        dotgraph.format = 'svg'
        dotgraph.render('file_'+str(count))
        """
        count += 1

if __name__ == "__main__":
    main()