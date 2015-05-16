#!/usr/bin/env python
# -*- coding: utf-8 -
import fileinput

import os
import re
import sys
from nltk import PunktWordTokenizer
import xdot
import graphviz
import StanfordDependencies
from nltk.parse.stanford import StanfordParser


class Relationship(object):

    def __init__(self, _sentence, _ent1, _ent2):
        self.sentence = _sentence
        self.ent1 = _ent1
        self.ent2 = _ent2
        self.dependencies = None


def compute_vectors():
    #TODO: simple sum or average
    pass


def extract_reverb_patterns():
    #TODO:
    pass


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
    head_idx = dependencies[token.index-1].head
    if head_idx == 0:
        heads.append(dependencies[head_idx-1])
        return heads
    else:
        heads.append(dependencies[head_idx-1])
        head_index = dependencies[token.index-1].head-1
        get_heads(dependencies, dependencies[head_index], heads)


def extract_shortest_dependency_path(rel):

    """
    for token in rel.dependencies:
        print token
    """

    # get position of entity and entity in tree
    idx1 = find_index_named_entity(rel.ent1, rel.dependencies)
    idx2 = find_index_named_entity(rel.ent2, rel.dependencies)

    #print "e1", idx1
    #print "e2", idx2

    print "ent1: ", rel.ent1
    print "ent2: ", rel.ent2

    heads_e1 = list()
    get_heads(rel.dependencies, rel.dependencies[idx1-1], heads_e1)

    heads_e2 = list()
    get_heads(rel.dependencies, rel.dependencies[idx2-1], heads_e2)

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

    print "\nshortest path: "
    print rel.ent1+"->",
    for t in heads_e1:
        if t != heads_e1[index_t1] and t != rel.dependencies[idx2-1]:
            print t.form+"->",
        else:
            print t.form
            break

    print rel.ent2+"->",
    for t in heads_e2:
        if t == rel.dependencies[idx1-1]:
            break
        elif t != heads_e2[index_t2]:
            print t.form+"->",
        else:
            print t.form
            break

    print "\n\n"

    """
    # get direct ascendent of ent1 and ent2
    head_e1 = rel.dependencies[idx1-1].head
    head_e2 = rel.dependencies[idx2-1].head

    if rel.dependencies[head_e1-1] == rel.dependencies[head_e2-1]:
        print rel.ent1, "-->", rel.dependencies[head_e1-1].form, "<--", rel.ent2
    """


def main():
    # JAVA_HOME needs to be set, calling 'java -version' should show: java version "1.8.0_45" or higher
    # PARSER and STANFORD_MODELS enviroment variables need to be set
    os.environ['STANFORD_PARSER'] = '/home/dsbatista/stanford-parser-full-2015-04-20/'
    os.environ['STANFORD_MODELS'] = '/home/dsbatista/stanford-parser-full-2015-04-20/'
    parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
    sd = StanfordDependencies.get_instance(backend='subprocess', jar_filename='/home/dsbatista/stanford-parser-full-2015-04-20/stanford-parser.jar')


    #examples.append(Relationship("In favour of the deal were Ted Turner, founder and boss of TBS, a cable-based business, and Time Warner chairman Gerald Levin, whose empire already held an 18 percent stake in TBS.", "Ted Turner", "TBS"))
    #examples.append(Relationship("Amazon.com founder and chief executive Jeff Bezos, said that he is happy to announce new gains.", "Amazon", "Jeff Bezos"))
    #examples.append(Relationship("But Richard Klein, a paleoanthropology professor at Stanford University , said the evidence is `` pretty sparse . ''", "", ""))
    #examples.append(Relationship("Protesters seized several pumping stations, holding 127 Shell workers hostage", "Protesters", "stations"))
    #examples.append(Relationship("Protesters seized several pumping stations, holding 127 Shell workers hostage", "workers", "stations"))
    #examples.append(Relationship("Troops recently have raided churches, warning ministers to stop preaching", "Troops", "churches"))
    #examples.append(Relationship("Troops recently have raided churches, warning ministers to stop preaching", "ministers", "churches"))

    examples = list()
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

    """
    for line in fileinput.input("golden_standard/acquired_negative_sentences.txt"):
        if line.startswith("sentence: "):
            sentence = line.split("sentence: ")[1].strip()
            examples.append(Relationship(sentence, "", ""))

    for line in fileinput.input("golden_standard/acquired_positive_sentences.txt"):
        if line.startswith("sentence: "):
            sentence = line.split("sentence: ")[1].strip()
            examples.append(Relationship(sentence, "", ""))
    """
    tags_regex = re.compile('</?[A-Z]+>', re.U)
    count = 0
    for rel in examples:
        sentence = re.sub(tags_regex, "", rel.sentence)
        print count, sentence
        t = parser.raw_parse(sentence)
        # draws the consituients tree
        #t[0].draw()

        # note: http://www.nltk.org/_modules/nltk/parse/stanford.html
        # the wrapper for StanfordParser does not give syntatic dependencies
        deps = sd.convert_tree(str(t[0]))
        rel.dependencies = deps

        extract_shortest_dependency_path(rel)

        # renders a PDF by default
        dotgraph = deps.as_dotgraph()
        dotgraph.format = 'svg'
        dotgraph.render('file_'+str(count))
        count += 1

if __name__ == "__main__":
    main()