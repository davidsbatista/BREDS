#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import copy
import fileinput
import math
import numpy as np
import os.path
import operator
import pickle
import random
import re
import string
import StringIO
import sys

# collections
from collections import defaultdict

# nltk
import nltk
from nltk import pos_tag
from nltk import word_tokenize
from nltk.tag.mapping import map_tag


class Relationship:

    def __init__(self, _sentence, _type, _arg1, _arg2, _e1_type = None,_e2_type = None, _offset_e1 = None, _offset_e2 = None, _id = None):
        self.sentence = _sentence
        if (_id is not None):
            self.id = None
        self.rel_type = _type
        self.arg1 = _arg1
        self.arg2 = _arg2

        if (_e1_type is not None):
            self.e1_type = _e1_type
        if (_e2_type is not None):
            self.e2_type = _e2_type
        if (_offset_e1 is not None):
            self.offset_e1 = _offset_e1
        if (_offset_e2 is not None):
            self.offset_e2 = _offset_e2

        self.before = ''
        self.between = ''
        self.after = ''

        self.tf_idf_BEF = None
        self.tf_idf_BET = None
        self.tf_idf_AFT = None

        # contexts with assocaited PoS-tags
        self.before_tags  = []
        self.between_tags = []
        self.after_tags   = []

        self.good_patterns = []

        # ReVerb patterns
        self.patterns_bef = []
        self.patterns_bet = []
        self.patterns_aft = []

        # ReVerb patterns normalized
        self.patterns_bef_norm = []
        self.patterns_bet_norm = []
        self.patterns_aft_norm = []
        self.patterns_bet_norm_tags = []

        # ReVerb patterns with associated PoS-tags
        self.patterns_bef_tags = []
        self.patterns_bet_tags = []
        self.patterns_aft_tags = []

        # word2vec vectors
        self.patterns_bef_vectors = []
        self.patterns_bet_vectors = []
        self.patterns_aft_vectors = []

    def __repr__(self):
        return "sentence         : "+self.sentence+'\n'+\
               "rel_type         : "+self.rel_type+'\n'+\
               "before           : "+self.before+'\n'+\
               "between          : "+self.between+'\n'+\
               "after            : "+self.after+'\n'+\
               "before_tags      : "+str(self.before_tags)+'\n'+\
               "between_tags     : "+str(self.between_tags)+'\n'+\
               "after_tags       : "+str(self.after_tags)+'\n'+\
               "pattern bef      : "+str(self.patterns_bef)+'\n'\
               "pattern bet      : "+str(self.patterns_bet)+'\n'\
               "pattern aft      : "+str(self.patterns_aft)+'\n'\
               "pattern bef_norm : "+str(self.patterns_bef_norm)+'\n'\
               "pattern bet_norm : "+str(self.patterns_bet_norm)+'\n'\
               "pattern aft_norm : "+str(self.patterns_aft_norm)+'\n'\


def process_semeval(data,direction_symmetric):
    print "\nProcessing SemEval dataset"
    relationships = defaultdict(list)
    rel_type = ''
    sentence = ''
    for line in fileinput.input(data):
        if re.match('^[0-9]+\t',line):
            id,sentence = line.strip().split('\t')
        elif not re.match('^Comment',line):
            rel_type = line.strip()
        if rel_type!='' and sentence!='':
            #rel = Relationship(id,sentence,rel_type,None,None)
            rel = Relationship(sentence,rel_type,None,None,None,None,None,None,id)
            sentence = ''
            relationships[rel.rel_type].append(rel)
    fileinput.close()
    
    if direction_symmetric==True:
        relationships_symmetric = defaultdict(list)
        # aggreate all sentences from the same relationship ignoring directions
        rel1 = ("Member-Collection(e1,e2)","Member-Collection(e2,e1)")
        rel2 = ("Entity-Destination(e1,e2)","Entity-Destination(e2,e1)")
        rel3 = ("Component-Whole(e1,e2)","Component-Whole(e2,e1)")
        rel4 = ("Content-Container(e1,e2)","Content-Container(e2,e1)")
        rel5 = ("Cause-Effect(e1,e2)","Cause-Effect(e2,e1)")
        rel6 = ("Entity-Origin(e1,e2)","Entity-Origin(e2,e1)")
        rel7 = ("Message-Topic(e1,e2)","Message-Topic(e2,e1)")
        rel8 = ("Instrument-Agency(e1,e2)","Instrument-Agency(e2,e1)")
        rel9 = ("Product-Producer(e1,e2)","Product-Producer(e2,e1)")
        rel10 = "Other"
        relations = [rel1,rel2,rel3,rel4,rel5,rel6,rel7,rel8,rel9,rel10]
        for rel in relations:
            if rel!="Other":
                rel_left = relationships[rel[0]]
                rel_right = relationships[rel[1]]
                rel_type = rel[0].split("(")[0]
                relationships_symmetric[rel_type] = rel_left + rel_right
            else:
                relationships_symmetric[rel] = relationships[rel]
        
        relationships = relationships_symmetric

    return relationships


def process_aimed(sentences,label):
    #TODO
    relationships = defaultdict(list)
    rel_type = ''
    sentence = ''
    for line in fileinput.input(data):
        if re.match('^[0-9]+\t',line):
            id,sentence = line.strip().split('\t')
        elif not re.match('^Comment',line):
            rel_type = line.strip()
        if rel_type!='' and sentence!='':
            rel = Relationship(id,sentence,rel_type,None,None)
            sentence = ''
            relationships[rel.rel_type].append(rel)
    fileinput.close()
    return relationships



def processACE(data):
    relationships = defaultdict(list)
    rel_type = ''
    sentence = ''
    offsets = ''
    entities = ''
    
    for line in fileinput.input(data):
        if len(line)==1:
            continue
            
        elif re.match('^sentence:',line):
            sentence = line.strip()
        
        elif re.match('^e1:',line):
            e1 = line.strip()
            
        elif re.match('^e2:',line):
            e2 = line.strip()

        elif re.match('^offset_e1:',line):
            offset_e1 = line.strip()

        elif re.match('^offset_e2:',line):
            offset_e2 = line.strip()

        elif re.match('^reltype:',line):
            rel_type = line.strip()
            sentence = sentence.split("sentence: ")[1]
            e1 = e1.split("e1: ")[1]
            e2 = e2.split("e2: ")[1]
            e1_type,e1 = e1.split("(")
            e2_type,e2 = e2.split("(")
            e1 = e1.split(")")[0]
            e2 = e2.split(")")[0]
            offset_e1 = offset_e1.split("offset_e1: ")[1]
            offset_e2 = offset_e2.split("offset_e2: ")[1]
            rel_type  = rel_type.split("reltype: ")[1]
            if rel_type == "[]": rel_type = "Other"
            else:
                rel_type = rel_type.strip("[").strip("]")

            rel = Relationship(sentence,rel_type,e1,e2,e1_type,e2_type,offset_e1,offset_e2)
            relationships[rel_type].append(rel)

    return relationships

"""
Dataset created by Bunescu et al. 2007 (http://www.cs.utexas.edu/~ml/papers/bunescu-acl07.pdf)
Available at: http://knowitall.cs.washington.edu/hlt-naacl08-data.txt
"""
def processBunescu(data):
    # Dataset created by Bunescu et al. 2007
    # http://www.cs.utexas.edu/~ml/papers/bunescu-acl07.pdf
    # http://knowitall.cs.washington.edu/hlt-naacl08-data.txt
    acquired = [('google','youtube'),('adobe','macromedia'),\
                ('viacom','dreamworks'),('novartis','eon labs'),
                ('adobe systems','macromedia'),('viacom','dreamworks skg'),\
                ('novartis','eon'),('google','youtube!!'),\
                ('google','inc. youtube inc.'),('google','youtube inc.'),\
                ('google','youtube..'),('google','youtube!!!'),\
                ('google','youtube!'),('google','inc. youtube'),\
                ('adobe systems,inc.','macromedia inc.'),\
                ('adobe systems incorporated','macromedia , inc.'),\
                ('adobe systems incorporated','macromedia , inc'),\
                ('adobe','macromedia!'),('viacom','inc dreamworks'),\
                ('viacom','dreamworks pictures'),\
                ('google inc.', 'youtube inc.'),('novartis', 'eon labs , inc.'),
                ('novartis ag', 'eon'),('adobe systems', 'macromedia inc.'),
                ('adobe systems inc.', 'macromedia inc.'),\
                ('viacom inc', 'dreamworks'),('novartis ag', 'eon labs'),\
                ('google , inc.', 'youtube')]

    born_in  = [('franz kafka','prague'),('andre agassi','las vegas'),\
                ('agassi','las vegas'),('george gershwin','new york'),\
                ('charlie chaplin','london'),('chaplin','london'),\
                ('kafka','prague'),('gershwin','new york'),\
                ('andre - agassi','las vegas')]
    
    awarded = [('albert einstein','nobel prize'),('francis crick','nobel prize'),\
               ('crick','nobel prize'),('john steinbeck','pulitzer prize'),\
               ('joseph pulitzer', 'pulitzer prize'),('einstein','nobel prize'),\
               ('steinbeck','pulitzer prize'),('john mccain','purple heart')]

    created = [('tim berners - lee','the world wide web'),\
                ('ruth handler','the barbie doll'),('kamen','the segway'),\
                ('berners - lee','the world wide web'),\
                ('dean kamen','the segway human transporter'),\
                ('dean kamen','the segway'),('john pemberton','coca cola'),\
                ('john pemberton','coca - cola'),\
                ('president truman', 'presidential medal of freedom'),\
                ('pemberton','coca - cola'),\
                ('harry truman', 'presidential medal of freedom')]

    relationships = defaultdict(list)
    regex = re.compile(r'<p[1-2]>[^<]+</p[1-2]>',re.U)

    for line in fileinput.input(data):
        if not line.startswith('#'):
            matches = []
            for m in re.finditer(regex,line):
                matches.append(m)

            for x in range(0,len(matches)-1):
                ent1 = matches[x].group()
                ent2 = matches[x+1].group()

                arg1 = re.sub("</?p[1-2]>","",ent1)
                arg2 = re.sub("</?p[1-2]>","",ent2)
                arg1 = arg1.lower().strip()
                arg2 = arg2.lower().strip()
                
                r = (arg1,arg2)
                sentence = line

                if r in acquired:
                    rel = Relationship(id,sentence,'acquired',None,None)
                    relationships['acquired'].append(rel)
                
                elif r in born_in:
                    rel = Relationship(id,sentence,'born_in',None,None)
                    relationships['born_in'].append(rel)
                
                elif r in created:
                    rel = Relationship(id,sentence,'created',None,None)
                    relationships['created'].append(rel)
                    
                elif r in awarded:
                    rel = Relationship(id,sentence,'awarded',None,None)
                    relationships['awarded'].append(rel)
                    
                else:
                    rel = Relationship(id,sentence,'false',None,None)
                    relationships['false'].append(rel)

    return relationships

"""
- PoS-taggs a sentence
- Extract ReVerB patterns
- Splits the sentence into 3 contexts: BEFORE,BETWEEN,AFTER
- Fills in the attributes in the Relationship class with this information
"""
def processSentencesSemEval(rel):
    # remove the tags and extract tokens
    text_tokens = word_tokenize(re.sub(r"</?e[1-2]>|\"", "", rel.sentence))
    regex = re.compile(r'<e[1-2]>[^<]+</e[1-2]>',re.U)

    # tag the sentence, using the default NTLK English tagger
    # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
    tagged = pos_tag(text_tokens)

    # convert the tags to reduced tagset (Petrov et al. 2012)
    # http://arxiv.org/pdf/1104.2086.pdf
    tags = []
    for t in tagged:
       tag = map_tag('en-ptb', 'universal', t[1])
       tags.append((t[0],tag))

    # extract contexts along with PoS-Tags
    matches = []

    for m in re.finditer(regex,rel.sentence):
        matches.append(m)

    for x in range(0,len(matches)-1):
        if x == 0:
            start = 0
        if x>0:
            start = matches[x-1].end()
        try:
            end = matches[x+2].init_bootstrapp()
        except:
            end = len(rel.sentence)-1

        before = rel.sentence[ start :matches[x].init_bootstrapp()]
        between = rel.sentence[matches[x].end():matches[x+1].init_bootstrapp()]
        after = rel.sentence[matches[x+1].end(): end ]
        ent1 = matches[x].group()
        ent2 = matches[x+1].group()

        arg1 = re.sub("</?e[1-2]>","",ent1)
        arg2 = re.sub("</?e[1-2]>","",ent2)

        rel.arg1 = arg1
        rel.arg2 = arg2

        quote = False
        bgn_e2 = rel.sentence.index("<e2>")
        end_e2 = rel.sentence.index("</e2>")
        if (rel.sentence[bgn_e2-1])=="'": 
            quote = True
        if (rel.sentence[end_e2+len("</e2>")])=="'":
            quote = True

        arg1_parts = arg1.split()
        arg2_parts = arg2.split()

        if quote:
            new = []
            for e in arg2_parts:
                if e.startswith("'") or e.endswith("'"):
                    e = '"'+e+'"'
                    new.append(e)
            arg2_parts = new

        rel.before = before
        rel.between = between
        rel.after = after

        before_tags = []
        between_tags = []
        after_tags = []

        # to split the tagged sentence into contexts, preserving the PoS-tags
        # has to take into consideration multi-word entities
        # NOTE: this works, but probably can be done in a much cleaner way

        before_i = 0
        for i in range(0,len(tags)):
            j = i
            z = 0
            while ( (z<=len(arg1_parts)-1) and tags[j][0]==arg1_parts[z]):
                j += 1
                z += 1
            if (z==len(arg1_parts)):
                before_i = i
                break;

        for i in range(before_i,len(tags)):
            j = i
            z = 0
            while ( (z<=len(arg2_parts)-1) and tags[j][0]==arg2_parts[z]):
                j += 1
                z += 1
            if (z==len(arg2_parts)):
                after_i = i
                break;

        before_tags = tags[:before_i]

        if len(arg1_parts)>1:
            between_tags = tags[before_i+2:after_i]
            after_tags = tags[after_i+1:]

        elif len(arg2_parts)>1:
            between_tags = tags[before_i+1:after_i]
            after_tags = tags[after_i+2:]

        else:
            between_tags = tags[before_i+1:after_i]
            after_tags = tags[after_i+1:]

        # fill attributes with contextual information in Relationship class
        rel.before_tags  = before_tags
        rel.between_tags = between_tags
        rel.after_tags   = after_tags


def process_ace(rel):
    # remove the tags and extract tokens
    text_tokens = word_tokenize(rel.sentence)

    # tag the sentence, using the default NTLK English tagger
    # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
    tagged = pos_tag(text_tokens)
    
    print tagged
    print rel

    # convert the tags to reduced tagset (Petrov et al. 2012)
    # http://arxiv.org/pdf/1104.2086.pdf
    tags = []
    for t in tagged:
       tag = map_tag('en-ptb', 'universal', t[1])
       tags.append((t[0],tag))

    # extract contexts along with PoS-Tags
    matches = []

    for m in re.finditer(regex,rel.sentence):
        matches.append(m)

    for x in range(0,len(matches)-1):
        if x == 0:
            start = 0
        if x>0:
            start = matches[x-1].end()
        try:
            end = matches[x+2].init_bootstrapp()
        except:
            end = len(rel.sentence)-1

        before = rel.sentence[ start :matches[x].init_bootstrapp()]
        between = rel.sentence[matches[x].end():matches[x+1].init_bootstrapp()]
        after = rel.sentence[matches[x+1].end(): end ]
        ent1 = matches[x].group()
        ent2 = matches[x+1].group()

        arg1 = re.sub("</?e[1-2]>","",ent1)
        arg2 = re.sub("</?e[1-2]>","",ent2)

        rel.arg1 = arg1
        rel.arg2 = arg2

        quote = False
        bgn_e2 = rel.sentence.index("<e2>")
        end_e2 = rel.sentence.index("</e2>")
        if (rel.sentence[bgn_e2-1])=="'": 
            quote = True
        if (rel.sentence[end_e2+len("</e2>")])=="'":
            quote = True

        arg1_parts = arg1.split()
        arg2_parts = arg2.split()

        if quote:
            new = []
            for e in arg2_parts:
                if e.startswith("'") or e.endswith("'"):
                    e = '"'+e+'"'
                    new.append(e)
            arg2_parts = new

        rel.before = before
        rel.between = between
        rel.after = after

        before_tags = []
        between_tags = []
        after_tags = []

        # to split the tagged sentence into contexts, preserving the PoS-tags
        # has to take into consideration multi-word entities
        # NOTE: this works, but probably can be done in a much cleaner way

        before_i = 0
        for i in range(0,len(tags)):
            j = i
            z = 0
            while ( (z<=len(arg1_parts)-1) and tags[j][0]==arg1_parts[z]):
                j += 1
                z += 1
            if (z==len(arg1_parts)):
                before_i = i
                break;

        for i in range(before_i,len(tags)):
            j = i
            z = 0
            while ( (z<=len(arg2_parts)-1) and tags[j][0]==arg2_parts[z]):
                j += 1
                z += 1
            if (z==len(arg2_parts)):
                after_i = i
                break;

        before_tags = tags[:before_i]

        if len(arg1_parts)>1:
            between_tags = tags[before_i+2:after_i]
            after_tags = tags[after_i+1:]

        elif len(arg2_parts)>1:
            between_tags = tags[before_i+1:after_i]
            after_tags = tags[after_i+2:]

        else:
            between_tags = tags[before_i+1:after_i]
            after_tags = tags[after_i+1:]

        # fill attributes with contextual information in Relationship class
        rel.before_tags = before_tags
        rel.between_tags = between_tags
        rel.after_tags = after_tags


def process_bunescu(rel):
    # remove the tags and extract tokens
    text_tokens = word_tokenize(re.sub(r"</?p[1-2]>|\"", "", rel.sentence))
    regex = re.compile(r'<p[1-2]>[^<]+</p[1-2]>',re.U)

    # tag the sentence, using the default NTLK English tagger
    # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
    try:
        tagged = pos_tag(text_tokens)
    except Exception, e:
        print e
        print rel.sentence
        sys.exit(0)

    # convert the tags to reduced tagset (Petrov et al. 2012)
    # http://arxiv.org/pdf/1104.2086.pdf
    tags = []
    for t in tagged:
       tag = map_tag('en-ptb', 'universal', t[1])
       tags.append((t[0],tag))

    # extract contexts along with PoS-Tags
    matches = []

    for m in re.finditer(regex,rel.sentence):
        matches.append(m)

    for x in range(0,len(matches)-1):
        if x == 0:
            start = 0
        if x>0:
            start = matches[x-1].end()
        try:
            end = matches[x+2].init_bootstrapp()
        except:
            end = len(rel.sentence)-1

        before = rel.sentence[ start :matches[x].init_bootstrapp()]
        between = rel.sentence[matches[x].end():matches[x+1].init_bootstrapp()]
        after = rel.sentence[matches[x+1].end(): end ]
        ent1 = matches[x].group()
        ent2 = matches[x+1].group()

        arg1 = re.sub("</?p[1-2]>","",ent1)
        arg2 = re.sub("</?p[1-2]>","",ent2)

        rel.arg1 = arg1
        rel.arg2 = arg2

        quote = False
        bgn_e2 = rel.sentence.index("<p2>")
        end_e2 = rel.sentence.index("</p2>")
        if (rel.sentence[bgn_e2-1])=="'": 
            quote = True
        if (rel.sentence[end_e2+len("</p2>")])=="'":
            quote = True

        arg1_parts = arg1.split()
        arg2_parts = arg2.split()

        if quote:
            new = []
            for e in arg2_parts:
                if e.startswith("'") or e.endswith("'"):
                    e = '"'+e+'"'
                    new.append(e)
            arg2_parts = new

        rel.before = before
        rel.between = between
        rel.after = after

        before_tags = []
        between_tags = []
        after_tags = []

        # to split the tagged sentence into contexts, preserving the PoS-tags
        # has to take into consideration multi-word entities
        # NOTE: this works, but probably can be done in a much cleaner way

        if rel.sentence=='<p1> Adobe </p1> acquired <p2> Macromedia! </p2> - sephiroth.it - flash &amp;amp; php':
            debug = True

        before_i = 0
        for i in range(0,len(tags)): 
            j = i
            z = 0
            while ( (z<=len(arg1_parts)-1) and tags[j][0]==arg1_parts[z]):
                j += 1
                z += 1
            if (z==len(arg1_parts)):
                before_i = i
                break;

        for i in range(before_i,len(tags)):
            j = i
            z = 0
            while ( (z<=len(arg2_parts)-1) and tags[j][0]==arg2_parts[z]):
                j += 1
                z += 1
            if (z==len(arg2_parts)):
                after_i = i
                break;

        before_tags = tags[:before_i]

        if len(arg1_parts)>1:
            between_tags = tags[before_i+2:after_i]
            after_tags = tags[after_i+1:]

        elif len(arg2_parts)>1:
            between_tags = tags[before_i+1:after_i]
            after_tags = tags[after_i+2:]

        else:
            try:
                between_tags = tags[before_i+1:after_i]
                after_tags = tags[after_i+1:]
            except Exception,e:
                print e
                print rel
                print tags
                sys.exit(0)


def reverb(rel):
    text = re.sub(r"</?e[1-2]>|\"", "", rel.between)
    
    #print "Pos-tagging "
    # Pos-Tagging and chunking
    #print "sentence :",text    
    l = nltk.pos_tag(nltk.word_tokenize(text))
    #print "tagged   :",l
    np_chunks_tree = chunker.parse(l)
    
    normalize = True
    lex_syn_constraints = True
    allow_unary = True
    short_rel = True
    
    # ReVerb relationship-extraction
    reverb = Reverb(text, np_chunks_tree, normalize, lex_syn_constraints, allow_unary, short_rel)
    triples = reverb.extract_triples()
    #rel.good_patterns = triples
    return triples
    #print "\n"
