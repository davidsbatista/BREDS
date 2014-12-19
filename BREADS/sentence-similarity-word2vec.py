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

# scipy
from scipy.sparse import csr_matrix
from scipy.linalg import norm

# collections
from collections import defaultdict

# nltk
import nltk
from nltk import pos_tag
from nltk import word_tokenize
from nltk import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag.mapping import map_tag

# word2vec
from gensim.models import Word2Vec

# scikit
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# custom stuff
from relations import Reverb

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

"""
Parse SemEval relationships file
returns: dict(rel_type,relationship)
"""
def processSemEval(data,direction_symmetric):
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


"""
Parse AImed relationships file
returns: dict(rel_type,relationship)
"""
def processAImed(sentences,label):
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
Extract ReVerb relational patterns
http://homes.cs.washington.edu/~afader/bib_pdf/emnlp11.pdf
"""
def extractReVerbPatterns(tagged_text):
    """
    VERB - verbs (all tenses and modes) 
    NOUN - nouns (common and proper) 
    PRON - pronouns 
    ADJ - adjectives 
    ADV - adverbs 
    ADP - adpositions (prepositions and postpositions) 
    CONJ - conjunctions 
    DET - determiners 
    NUM - cardinal numbers 
    PRT - particles or other function words 
    X - other: foreign words, typos, abbreviations 
    . - punctuation
    """
    # extract ReVerb patterns:
    # V | V P | V W*P
    # V = verb particle? adv?
    # W = (noun | adj | adv | pron | det)
    # P = (prep | particle | inf. marker)

    tags = tagged_text

    patterns = []
    patterns_tags = []
    i = 0
    limit = len(tags)-1

    while i <= limit:
        tmp = StringIO.StringIO()
        tmp_tags = []
        # a ReVerb pattern always starts with a verb
        if (tags[i][1] == 'VERB'):
            tmp.write(tags[i][0]+' ')
            t = (tags[i][0],tags[i][1])
            tmp_tags.append(t)
            i += 1
            # V = verb particle? adv? (also capture auxiliary verbs)
            while (i <= limit and tags[i][1] in ['VERB','PRT','ADV']):
                tmp.write(tags[i][0]+' ')
                t = (tags[i][0],tags[i][1])
                tmp_tags.append(t)
                i += 1
            # W = (noun | adj | adv | pron | det)
            while (i <= limit and tags[i][1] in ['NOUN','ADJ','ADV','PRON','DET']):
                tmp.write(tags[i][0]+' ')
                t = (tags[i][0],tags[i][1])
                tmp_tags.append(t)
                i += 1
            # P = (prep | particle | inf. marker)
            while (i <= limit and tags[i][1] in ['ADP','PRT']):
                tmp.write(tags[i][0]+' ')
                t = (tags[i][0],tags[i][1])
                tmp_tags.append(t)
                i += 1
            # add the build pattern to the list collected patterns
            patterns.append(tmp.getvalue())
            patterns_tags.append(tmp_tags)
        i += 1

    return patterns,patterns_tags


def testReVerbPatternsExtraction(sentences):
    for line in fileinput.input(sentences):
        #s = line.split('sentence:')[1].strip()
        text_tokens = word_tokenize(re.sub(r"</?e[1-2]>|\"", "", line))
        tagged = pos_tag(text_tokens)

        # convert the tags to reduced tagset
        tags = []
        for t in tagged:
           tag = map_tag('en-ptb', 'universal', t[1])
           tags.append((t[0],tag))

        #r = Relationship(None, s, None, None, None)
        #extractRelationalWords(r)
        print tags


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
            end = matches[x+2].start()
        except:
            end = len(rel.sentence)-1

        before = rel.sentence[ start :matches[x].start()]
        between = rel.sentence[matches[x].end():matches[x+1].start()]
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

        # extract ReVerb patterns from each context
        rel.patterns_bef, rel.patterns_bef_tags = extractReVerbPatterns(before_tags)
        rel.patterns_bet, rel.patterns_bet_tags = extractReVerbPatterns(between_tags)
        rel.patterns_aft, rel.patterns_aft_tags = extractReVerbPatterns(after_tags)



def processSentencesACE(rel):
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
            end = matches[x+2].start()
        except:
            end = len(rel.sentence)-1

        before = rel.sentence[ start :matches[x].start()]
        between = rel.sentence[matches[x].end():matches[x+1].start()]
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

        # extract ReVerb patterns from each context
        rel.patterns_bef, rel.patterns_bef_tags = extractReVerbPatterns(before_tags)
        rel.patterns_bet, rel.patterns_bet_tags = extractReVerbPatterns(between_tags)
        rel.patterns_aft, rel.patterns_aft_tags = extractReVerbPatterns(after_tags)

def processSentencesBunescu(rel):
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
            end = matches[x+2].start()
        except:
            end = len(rel.sentence)-1

        before = rel.sentence[ start :matches[x].start()]
        between = rel.sentence[matches[x].end():matches[x+1].start()]
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

        # fill attributes with contextual information in Relationship class
        rel.before_tags  = before_tags
        rel.between_tags = between_tags
        rel.after_tags   = after_tags

        # extract ReVerb patterns from each context
        rel.patterns_bef, rel.patterns_bef_tags = extractReVerbPatterns(before_tags)
        rel.patterns_bet, rel.patterns_bet_tags = extractReVerbPatterns(between_tags)
        rel.patterns_aft, rel.patterns_aft_tags = extractReVerbPatterns(after_tags)


def status(seeds,target,others):
    print "\n"
    print "number in seed set   :",len(seeds)
    print "number in target set :",len(target)
    tmp = []
    for k in others.keys():
        for r in others[k]: 
            tmp.append(r)
    print "number in others set :",len(tmp)

"""
 takes the dict of relationships
 and returns a list
"""
def getOthersList(others):
    tmp = []
    for k in others.keys():
        for r in others[k]: 
            tmp.append(r)
    return tmp


"""
Counts the TF-IDF for each token in the datasets of relationships. 
Computes a vector representations for each context: BEF,BET,AFT
"""
def buildTFIDF(relationships):
    token_dict = {}
    
    id = 0
    for key in relationships.keys():
        for r in relationships[key]:
            text = re.sub(r"</?e[1-2]>|\"", "", r.sentence)
            lowers = text.lower()
            #remove the punctuation using the character deletion step of translate
            no_punctuation = lowers.translate(None, string.punctuation)
            token_dict[id] = no_punctuation
            id +=1

    # this can take some time
    #tfidf = TfidfVectorizer(tokenizer=word_tokenize, stop_words='english')
    tfidf = TfidfVectorizer(tokenizer=word_tokenize)
    tfs = tfidf.fit_transform(token_dict.values())

    for key in relationships.keys():
        print key
        for rel in relationships[key]:
            bef = tfidf.transform([rel.before])
            bet = tfidf.transform([rel.between])
            aft = tfidf.transform([rel.after])
            rel.tf_idf_BEF = bef
            rel.tf_idf_BET = bet
            rel.tf_idf_AFT = aft

def main():
    cluster = False;
    bootstrap = True;
    global symmetric
    symmetric = False;

    heuristic = sys.argv[2]
    if heuristic not in ['DBSCAN','Centroids','TFIDF','Aggregated']:
        print "Use one of the following:"
        print 'DBSCAN'
        print 'Centroids'
        print 'TFIDF'
        print 'Aggregated'
        sys.exit(0)

    print "Loading word2vec model ...\n"
    global model
    model = Word2Vec.load_word2vec_format("/home/dsbatista/word2vec-read-only/vectors.bin",binary=True)
    #model = Word2Vec.load_word2vec_format("/home/dsbatista/GoogleNews-vectors-negative300.bin",binary=True)
    global vectors_dim
    vectors_dim = 200
    global words_not_found
    words_not_found = []
    
    """
    print "Initialize Chunker"
    train_sents = nltk.corpus.conll2000.chunked_sents('train.txt', chunk_types=['NP'])
    global chunker
    chunker = BigramChunker(train_sents)
    """

    # if dump of pre-processed dataset does not exists pre-process
    if not os.path.isfile("sentences_processed.pkl"):
        dataset = 'ACE'
        if dataset=='SemEval':
            relationships = processSemEval(sys.argv[1],symmetric)
        elif dataset=='Bunescu':
            relationships = processBunescu(sys.argv[1])
        elif dataset=='ACE':
            relationships = processACE(sys.argv[1])
            l = relationships['Other']
            relationships['Other'] = l[:500]
        
        # extract ReVerb patterns from the three contexts
        print "\nExtracting relational patterns from all sentences"
        for rel_type in relationships:
            print rel_type
            for r in relationships[rel_type]:
                if dataset=='SemEval':
                    processSentencesSemEval(r)
                elif dataset=='Bunescu':
                    processSentencesBunescu(r)
                elif dataset=='ACE':
                    processSentencesACE(r)


        # build a the three vector (TF-IDF) representation: BEF,BET,AFT
        print "\nBulding TF-IDF for BEF,BET,AFT contexts"
        buildTFIDF(relationships)
        
        # save processed sentences do disk
        with open('sentences_processed.pkl', 'wb') as f:
            pickle.dump(relationships, f)
        f.close()
        
    else:
        # load processed sentences from disk
        print "\nLoading pre-processed sentences..."
        with open('sentences_processed.pkl', 'r') as f:
            relationships = pickle.load(f)
        f.close()
        
    """
    Constructs a vector representation of ReVerb patterns
    by summing the words that are part of the ReVerb pattern
    
     TODO (cases to analyze)
     - sentences with no tokens in between
     - pattern/words not in model ? this happens?
     - sentences with > 1 patterns
    """
    
    """
    for rel_type in relationships:
        print rel_type
        for r in relationships[rel_type]:
            reverb(r)
    """
    
    print "\nGenerating word2vec vectors from ReVerb patterns"
    for rel_type in relationships:
        print rel_type
        for r in relationships[rel_type]:
            if len(r.patterns_bet_tags)>=1:
                for pattern in r.patterns_bet_tags:
                    norm_tags = normalizeReVerbPattern(pattern)
                    if norm_tags!=None:
                        norm = ' '.join([w[0] for w in norm_tags])

                        """
                        print "NORM"
                        print norm
                        print norm_tags
                        print ""
                        """

                        r.patterns_bet_norm.append(norm)
                        r.patterns_bet_norm_tags = norm_tags
                    """
                    norm = reverb(r)
                    print r
                    print "ReVerb:  ", norm
                    print "\n"
                    if norm!=None and len(norm)>0:
                        r.patterns_bet_norm.append(norm[0])
                    """

            elif len(r.between_tags)>=1:
                pass
                # if no ReVerb patterns were found in the 'between' context
                # but if between context contains tokens use them as a pattern
                # TODO: select what goes and what goes not
                """
                tmp = []
                #print r.between_tags
                #print r.sentence
                for token in r.between_tags:
                    if token[0]!="'s" and token[0]!=" ":
                        tmp.append(token[0])
                if len(tmp)>0:
                    pattern = ' '.join(tmp)
                    r.patterns_bet_norm.append(pattern)
                """
            else:
                #TODO
                # if between context is empty
                #  - use the entity
                #  - verbs close to the entities in BEF and AFT
                pass

            # generate a word2vec vector base on one of the following
            #  - normalized ReVerb patterns
            #  - context words
            patterns2Vectors(r)

            # assert that every sentence has at least on pattern vector
            #assert len(r.patterns_bet_vectors)>0
            
            # assert that every pattern vectors do not contain NaN
            # assert that there are no 0s vectors
            for p in r.patterns_bet_vectors:
                assert not np.isnan(p).any()
                #assert np.count_nonzero(p)>=1

    fdist_words = FreqDist(w for w in words_not_found)
    print "\n",len(fdist_words),"words not found in word2vec model"

    total = 0;
    for rel_type in relationships:
        #f = open(rel_type+".patterns",'w')
        for r in relationships[rel_type]:
            total += 1
            #f.write(str(r)+'\n')
        #f.close()

    # test semantic drift for all relationship type
    easy = ['Cause-Effect(e1,e2)','Cause-Effect(e2,e1)',\
            'Entity-Destination(e1,e2)','Entity-Destination(e2,e1)',\
            'Member-Collection(e1,e2),Member-Collection(e2,e1)']

    for rel in easy:
    #for rel in relationships:
        relationships_tmp = copy.deepcopy(relationships)
        print "\nProcessing", rel
        results = init_bootstrap(relationships_tmp,rel,heuristic,total)
        print "Relationship      : ",rel
        print "New Relationships : ",results[0]
        print "Positive          : ",results[1]
        print "Negative          : ",results[2]
        print "Accuracy          : ",results[3]
        print "Coverage          : ",results[4]
        print "F1                : ",results[5]
        print "=========================================\n"

"""
Generate word2vec vectors based on words that mediate the relationship
- If between context is empty
"""
def patterns2Vectors(rel):
    # sum each word of a pattern in the 'between' context
    if len(rel.patterns_bet_norm)>0:
        pattern = rel.patterns_bet_norm[0]
        pattern_vector = np.zeros(vectors_dim)
        for word in word_tokenize(pattern):
            try:
                vector = model[word.strip()]
                pattern_vector += vector
            except Exception, e:
                words_not_found.append(word.strip())
        rel.patterns_bet_vectors.append(pattern_vector)


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


"""
Normalize relation phrase by removing: auxiliary verbs, adjectives, and adverbs
TODO: if we are dealing with direction, do not remove aux verbs, can be used to 
      detect passive voice
"""
def normalizeReVerbPattern(pattern):
    lmtzr = WordNetLemmatizer()
    aux_verbs = ['be', 'do', 'have', 'will', 'would', 'can', 'could', 'may', \
    'might', 'must', 'shall', 'should', 'will', 'would']
    norm_pattern = []
    
    #print rel_type
    #print "pattern  :",pattern
    
    # remove ADJ, ADV, and auxialiary VERB
    for i in range(0,len(pattern)):
        if pattern[i][1] == 'ADJ':
            continue
        elif pattern[i][1] == 'ADV':
            continue
        else:
            norm_pattern.append(pattern[i])
        """
        elif pattern[i][1]== 'VERB':
            verb = lmtzr.lemmatize(pattern[i][0],'v')
            if (verb in aux_verbs):
                if (len(pattern)>=i+2):
                    if (pattern[i+1][1]=='VERB'):
                        continue
                    else:
                        norm_pattern.append(pattern[i])
                else:
                    norm_pattern.append(pattern[i])
            else:
                norm_pattern.append(pattern[i])
        """


    # if last word of pattern is a NOUN remove
    #while norm_pattern[-1][1]=='NOUN':
    #    del norm_pattern[-1]

    #print "norm     :",norm_pattern

    # Lemmatize VERB using WordNet
    """
    lemma_pattern = []
    for i in range(0,len(norm_pattern)):
        if norm_pattern[i][1]== 'VERB':
            norm = WordNetLemmatizer().lemmatize(norm_pattern[i][0],'v')
            lemma_pattern.append((norm,'VERB'))
        elif norm_pattern[i][0] == "'s":
            continue
        else:
            lemma_pattern.append(norm_pattern[i])    
    #print "lemma    :",lemma_pattern
    """
    
    # Short Version: just VERBs and ending ADP, PRT, DET
    short = []
    #short.append(lemma_pattern[0])   
    #if (len(lemma_pattern)>1) and lemma_pattern[1][1]=='VERB':
    #    short.append(lemma_pattern[1])
    
    short.append(norm_pattern[0])
    if (len(norm_pattern)>1) and norm_pattern[1][1]=='VERB':
        short.append(norm_pattern[1])
    tmp = []
    for w in reversed(norm_pattern):
        if (w[1] == 'ADP' or w[1] == 'NOUN' or w[1] == 'PRT' or w[1] == 'DET'):
            tmp.append(w)
            
    for w in reversed(tmp):
        short.append(w)

    if len(short)==1 and short[0][1]== 'VERB':
        verb = lmtzr.lemmatize(short[0][0],'v')
        if (verb in aux_verbs):
            return None
    
    else:
        return short

"""
Start the bootstrap process based on sentences, by simulating iterations:
  - Extract sentences holding the target relationship
  - Extract randomly noisy sentences
  - Apply different heuristics do discard noisy sentences and keep target
"""
def init_bootstrap(relationships,TARGET_RELATIONSHIP,heuristic,total):

    """
    seeds:    initial seeds, with the target relationship
    target:   target class relationships sentences
    others:   other classes relationship sentences
    TARGET_RELATIONSHIP: relationship to be extracted
    """    
    # SemEval results: http://www.cl.cam.ac.uk/~do242/Papers/semeval2multiway.pdf
    
    total_target = len(relationships[TARGET_RELATIONSHIP])
    num_classes = len(relationships)
    T = int(0.20 * total_target)
    S = 30
    threshold = 0.33
    seeds = []
    target = relationships[TARGET_RELATIONSHIP]
    others = defaultdict(list)
    for rel_type in relationships.keys():
        if rel_type!=TARGET_RELATIONSHIP:
            for r in relationships[rel_type]:
                others[rel_type].append(r)

    """
    status(seeds,target,others)
    print "Total :", len(seeds)+len(target)+len(getOthersList(others))
    """
    assert len(seeds)+len(target)+len(getOthersList(others)) == total
    
    # select 'T' randomly seed sentences from target set
    count = 0
    while ( count < T ):
         # select a random sentence  add it to the seed list,
         # remove it from target list
         random.seed() 
         index = random.randint(0,len(target)-1)
         r = target[index]
         seeds.append(r)
         del target[index]
         count += 1

    discarded = 0
    iteration = 0
    # get randomly sentences from target and from other classes
    # γ · (|target| − |seeds|) + (1 − γ) · |others| = S
    while (len(target)+len(getOthersList(others)))>0:
        # determine how many sentences from each group
        num_target = threshold * S
        num_others = (1-threshold) * S
        iteration_sentences = []
        
        """
        print "Selecting..."
        print num_target,"target sentences"
        print num_others,"others sentences"
        """
        
        # select randomly sentences from target class
        count = 0
        while (count < num_target):
            # select a random sentence 
            # add it to sentences considered for this iteration
            # remove it from target list
            random.seed()
            if len(target)>0:
                index = random.randint(0,len(target)-1)
                r = target[index]
                iteration_sentences.append(r)
                del target[index]
                count += 1
            else:
                break
                
        # print "added",len(iteration_sentences),"seed sentences"
        
        # select randomly sentences from other classes
        # number of sentences for each class is equally distributed
        num_per_class = num_others / (num_classes-1)        
        for key in others.keys():
            count = 0
            while (count < num_per_class):
                # select a random sentence 
                # add it to sentences considered for this iteration
                # remove it from other list
                random.seed() 
                if len(others[key])>0:
                    index = random.randint(0,len(others[key])-1)
                    r = others[key][index]
                    iteration_sentences.append(r)
                    del others[key][index]
                    count += 1
                else:
                    break

        """
        status(seeds,target,others)
        print "New sentences to analyze    : ",len(iteration_sentences)
        print "Sentences already discarded : ",discarded
        print "Total", (len(seeds) + len(target) + len(getOthersList(others)) + len(iteration_sentences) + discarded)
        """
        
        assert len(seeds) + len(target) + len(getOthersList(others))\
         + len(iteration_sentences) + discarded == total

        # assure lists are disjoint
        assert len(set(seeds).intersection(set(target)).intersection(set(getOthersList(others))))==0

        # explore different heuristics do detect semantic drift
        if heuristic=='DBSCAN':
            selected = simDBSCANCentroids(seeds,list(iteration_sentences))
        elif heuristic=='Centroids':
            selected = simPatternCentroids(seeds,list(iteration_sentences))
        elif heuristic=='TFIDF':
            selected = simTFIDF(seeds,list(iteration_sentences))
        elif heuristic=='Aggregated':
            selected = simAggregatedPatterns(seeds,list(iteration_sentences),iteration)
        else:
            print "erro!"
            print sys.exit(0)

        selected = analyzeSelected(selected)

        """
        print "analyzed             : ", len(iteration_sentences)
        print "selected             : ", len(selected)
        print "discarded            : ", len(iteration_sentences)-len(selected)
        """
        discarded += len(iteration_sentences) - len(selected)

        if (discarded<0):
            print "discarded cannot be negative"
            sys.exit(0)
        
        #print "total", (len(seeds) + len(target) + len(getOthersList(others)) + len(selected) + discarded)
        assert len(seeds) + len(target) + len(getOthersList(others))\
         + discarded + len(selected) == total
        
        
        #TODO: from the selected analyze if they are related
        
        # from the selected sentences select which are new, that is,
        # were not already part of the seed set
        already_seen = 0
        added = 0
        for rel in selected:
            if rel not in seeds:
                seeds.append(rel)
                added += 1
            else: 
                already_seen += 1
        discarded += already_seen

        """
        print "\nseeds        :", len(seeds)
        print "target       :", len(target)
        print "others       :", len(getOthersList(others))
        print "already seen :", already_seen
        print "discarded    :", discarded
        print "total", (len(seeds) + len(target) + len(getOthersList(others)) + discarded )
        """
        
        assert len(seeds) + len(target) + len(getOthersList(others))\
         + discarded == total
        
        # assure lists are disjoint
        if len(set(seeds).intersection(set(target)).intersection(set(getOthersList(others))))>0:
            print "related and unrelated and seed sentences not disjoint"
            sys.exit(0)

        positive = 0
        negative = 0
        for r in seeds:
            if symmetric == True:
                # symmetric relationships
                if r.rel_type.split("(")[0] == TARGET_RELATIONSHIP:
                    positive += 1
                elif r.rel_type.split("(")[0] != TARGET_RELATIONSHIP:
                    negative += 1
            elif symmetric == False:
                if r.rel_type == TARGET_RELATIONSHIP:
                    positive += 1
                elif r.rel_type != TARGET_RELATIONSHIP:
                    negative += 1

        # accuracy over the whole class
        """
        accuracy = float(positive)/float(positive+negative)
        coverage = float(positive)/float(total_target)
        """
        
        # accuracy just over the unknown part, to be discovered/extracted
        if (positive+negative - T)>0:
            accuracy = float(positive - T)/float(positive+negative - T)
        else: accuracy = 0
        
        if (total_target - T)>0:
            coverage = float(positive - T)/float(total_target - T)
        else: coverage = 0
        
        if (accuracy+coverage)>0: 
            f1 = 2*(accuracy*coverage)/(accuracy+coverage)
        else: 
            f1 = 0
        
        print "Relationship      : ",TARGET_RELATIONSHIP
        print "New Relationships : ",len(seeds)-T
        print "Positive          : ",positive-T
        print "Negative          : ",negative
        print "Accuracy          : ",accuracy
        print "Coverage          : ",coverage
        print "F1                : ",f1
        #status(seeds,target,others)
        print "=========================================\n"
        if len(target)>0:
            iteration += 1
        else:
            break
        
    return [len(seeds)-T,positive-T,negative,accuracy,coverage,f1]


"""
A TF-IDF centroid is calculated for each context, from the seed sentences
Every new sentence is compared with the centroid, if its above a threshold
it is added to the seed set
"""
def simTFIDF(seeds,iteration_sentences):
    alpha = 0.0
    beta = 1.0
    gamma = 0.0
    threshold = 0.6
    selected = []

    BEF_centroid = csr_matrix(seeds[0].tf_idf_BEF.shape)
    BET_centroid = csr_matrix(seeds[0].tf_idf_BET.shape)
    AFT_centroid = csr_matrix(seeds[0].tf_idf_AFT.shape)

    for i in range(0,len(seeds)):
        BEF_tmp = csr_matrix(BEF_centroid.shape)
        BET_tmp = csr_matrix(BET_centroid.shape)
        AFT_tmp = csr_matrix(AFT_centroid.shape)
        
        BEF_tmp = BEF_centroid + seeds[i].tf_idf_BEF
        BET_tmp = BET_centroid + seeds[i].tf_idf_BET
        AFT_tmp = AFT_centroid + seeds[i].tf_idf_AFT

        BEF_centroid = BEF_tmp
        BET_centroid = BET_tmp
        AFT_centroid = AFT_tmp

    BEF_centroid = BEF_centroid/len(seeds)
    BET_centroid = BET_centroid/len(seeds)
    AFT_centroid = AFT_centroid/len(seeds)
    
    # compare centroid with other TF-IDFS
    for other in iteration_sentences:
        sim_BEF = cosine_similarity(BEF_centroid,other.tf_idf_BEF)
        sim_BET = cosine_similarity(BET_centroid,other.tf_idf_BET)
        sim_AFT = cosine_similarity(AFT_centroid,other.tf_idf_AFT)
        tfidf_sim = alpha*sim_BEF+beta*sim_BET+gamma*sim_AFT
        if (tfidf_sim[0][0]>=threshold):
            selected.append(other)

    return selected


def simAggregatedPatterns(seeds,iteration_sentences,iteration):
    selected = []
    threshold = 0.6
    ratio_threshold = 0.6
    
    #NOTE: if its the first iteration, all are valid patterns
    patterns = []
    patterns_vectors = dict()
    no_patterns = 0
    for r in seeds:
        if len(r.patterns_bet_norm)==0:
            no_patterns += 1
        else:
            #TODO: so estou a usar o primero pattern
            """
            print r.sentence
            print r.patterns_bet_tags
            print r.patterns_bet_norm
            print "\n"
            """
            patterns.append(r.patterns_bet_norm[0])
            patterns_vectors[r.patterns_bet_norm[0]] = r.patterns_bet_vectors[0]
    
    assert no_patterns + len(patterns) == len(seeds)
    fdist_patterns = FreqDist(p for p in patterns)

    """
    print "seeds  sentences :", len(seeds)
    print "no patterns      :", no_patterns
    print "seeds  patterns  :", len(patterns)
    print "unique patterns  :", fdist_patterns.B()
    print "other sentences  :", len(iteration_sentences)
    """

    items = fdist_patterns.items()
    #select only patterns with frequency > 1
    #selected_patterns = [t for t in items if t[1] >= 1]
    selected_patterns = []
    for i in items:
        if not len(word_tokenize(i[0]))>3:
            selected_patterns.append(i)

    """
    for i in selected_patterns:
        print i
    """
    
    #print "Pairwise comparision"
    patterns_idx = []
    all_vectors = []
    for pattern in selected_patterns:
        p = pattern[0]
        vector = patterns_vectors[p]
        # make sure there are no NaN and no 0s-only vectors
        assert not np.isnan(vector).any()
        if np.count_nonzero(vector)>=1:
            all_vectors.append(vector)
            patterns_idx.append(pattern)
        #else:
        #    print pattern
        #    print vector
        #    sys.exit(0)

    #[‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]
    #print "\nComputing pairwise distances...",len(all_vectors)
    # NOTE: this applies the cosine distance:
    #       1 - cos(x,y)
    try:
        matrix = pairwise.pairwise_distances(np.array(all_vectors),metric='cosine',n_jobs=-1)
    except Exception, e:
        print e
        print all_vectors
        sys.exit(0)
    
    """
    idx1 = 0
    for row in matrix:
        print patterns_idx[idx1]
        print len(row)
        idx2 = 0
        for score in row:
            print patterns_idx[idx1],",",patterns_idx[idx2],'\t',1-score
            idx2 += 1
        idx1 += 1
    
    sys.exit(0)
    """
    eps = 0.2
    min_samples = 2

    db = DBSCAN(eps, min_samples, metric='precomputed')
    db.fit(matrix)

    # Number of clusters in labels, ignoring noise if present
    n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

    #print n_clusters_,"clusters generated"

    # instances classified as noise go into 'noise' list
    # 'cluster' is a list where each index is the label(number) of a cluster
    # each position in the list contains a list with the objects instances
    noise = []
    clusters = []
    for i in range(0,n_clusters_):
        clusters.append([])

    assert len(db.labels_) == len(patterns_idx)

    for i in range(0,len(db.labels_)):
        cluster_n = int(db.labels_[i])
        if cluster_n == -1:
            noise.append(patterns_idx[i])
        else:
            clusters[cluster_n].append(patterns_idx[i])

    """
    for cluster in clusters:
        print cluster
    """

    # Compare each other pattern with every element in each
    # cluster, use the maximum score
    # TODO: alternative, use average
    for o in iteration_sentences:
        positive = 0
        #maximum = -float("inf")
        if len(o.patterns_bet_norm)==0:
            #TODO: frases sem patterns
            no_patterns += 1
        else:
            #TODO: só estou a usar 1 vector
            o_vector = o.patterns_bet_vectors[0]
            i = 0
            averages = []
            for cluster in clusters:
                maximum = -float("inf")
                average = 0
                for p in cluster:
                    # every element in a cluster is a tuple from FreqDist
                    # (textual_pattern,absolut_frequency)
                    # gets the word2vec representations of the pattern
                    vector = patterns_vectors[p[0]]
                    score = cosine_similarity(o_vector,vector)[0][0]
                    """
                    print "rel_type         :",o.rel_type
                    print "sentence         :",o.sentence
                    print "pattern sentence :",o.patterns_bet_norm[0]
                    print "pattern seed     :",p
                    print "score    :",score
                    print ""
                    """
                    average += score
                    if score>maximum:
                        maximum = score

                average /= len(cluster)
                averages.append(average)
                if maximum>=threshold:
                    positive += 1
            
            #print "positive     :",positive
            #print "num_clusters :",len(clusters)
            ratio = float(positive)/len(clusters)
            """
            print "\n"
            print "rel_type         :",o.rel_type
            print "sentence         :",o.sentence
            print "pattern sentence :",o.patterns_bet_norm[0]
            print "ratio        :",ratio
            print "averages     :",averages
            """
            if (ratio>=ratio_threshold):
                selected.append(o)
            #print ""


    return selected

"""
Cluster patterns seeds with DBSCAN
Calculates a centroid for each generated cluster
Keeps sentences with a sim threshold to iteration_sentences
"""
def simDBSCANCentroids(seeds,iteration_sentences):
    eps=0.2
    min_samples=2
    threshold=0.6
    threshold_ratio=0.5
    selected = []
    
    all_vectors = []
    seeds_idx = []
    for seed in seeds:
        if len(seed.patterns_bet)==1:
            seedPatternVector = seed.patterns_bet_vectors[0]
            # make sure there are no NaN and no 0s-only vectors
            assert not np.isnan(seedPatternVector).any()
            if np.count_nonzero(seedPatternVector)>=1:
                all_vectors.append(seed.patterns_bet_vectors[0])
                seeds_idx.append(seed)
            else:
                pass
                """
                print "found zero vector"
                print seed
                sys.exit(0)
                """

    #[‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]
    #print "\nComputing pairwise distances..."
    matrix = pairwise.pairwise_distances(np.array(all_vectors),metric='cosine',n_jobs=-1)
    
    c = 0
    for row in matrix:
        for e in range(0,len(row)):
            # TODO: assure there are no nan, use other relational words
            # if is 'nan', one of the vectors had 0s, then set distance to 1
            if np.isnan(row[e]):
                row[e]=1
                print "Found a Nan"
                print rel_a.rel_type,rel_b.rel_type,'\t',row[e]
                sys.exit(0)
        c += 1

    #print "DBScan clustering...",
    db = DBSCAN(eps, min_samples, metric='precomputed')
    db.fit(matrix)

    # Number of clusters in labels, ignoring noise if present
    n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

    #print n_clusters_,"clusters generated"

    # instances classified as noise go into 'noise' list
    # 'cluster' is a list where each index is the label(number) of a cluster
    # each position in the list contains a list with the objects instances
    noise = []
    clusters = []
    for i in range(0,n_clusters_):
        clusters.append([])

    assert len(db.labels_) == len(seeds_idx)

    for i in range(0,len(db.labels_)):
        cluster_n = int(db.labels_[i])
        if cluster_n == -1:
            noise.append(seeds_idx[i])
        else:
            clusters[cluster_n].append(seeds_idx[i])

    # calculate centroid of each cluster
    centroids = []
    for cluster in clusters:
        centroid = calculateCentroid(cluster)
        centroids.append(centroid)


    # calculate the distance of 'other' to each centroid
    for other in iteration_sentences:
        if len(other.patterns_bet)==1:
            positive = 0
            otherPatternVector = other.patterns_bet_vectors[0]
            # make sure there are no NaN and no 0s-only vectors
            assert not np.isnan(seedPatternVector).any()
            if np.count_nonzero(seedPatternVector)>=1:
                for centroid in centroids:
                    score = cosine_similarity(otherPatternVector,seedPatternVector)[0][0]
                    if score>=threshold:
                        positive += 1

            negative = len(centroids)-positive
            ratio = 0
            if positive>0:
                ratio = positive/float(positive+negative)
            #print "ratio",ratio
            if ratio>=threshold_ratio:
                selected.append(other)
            #c += 1
            #if (c % 50 == 0): print c,"/",len(iteration_sentences)

    return selected


"""
Calculates a centroid given a list of seeds
The pattern vectors are normalized to the norm
Returns a centroid
"""
def calculateCentroid(sentences):
    # calculate seed patterns centroid
    centroid = np.zeros(vectors_dim)
    for s in sentences:
        if len(s.patterns_bet)==1:
            patternVector = s.patterns_bet_vectors[0]
            
            # make sure there are no NaN and no 0s-only vectors
            assert not np.isnan(patternVector).any()
            if np.count_nonzero(patternVector)>=1:
                # normalize pattern vector
                n = norm(patternVector)
                patternVectorNorm = np.divide(patternVector,n)
                # accumulate
                centroid += patternVectorNorm
    
    centroid /= len(sentences)

    return centroid


def simPatternCentroids(seeds,iteration_sentences):
    selected = []
    threshold = 0.6
    
    # calculate seed patterns centroid
    pattern_vector_centroid = np.zeros(vectors_dim)
    for seed in seeds:
        if len(seed.patterns_bet)==1:
            seedPatternVector = seed.patterns_bet_vectors[0]
            # make sure there are no NaN and no 0s-only vectors
            assert not np.isnan(seedPatternVector).any()
            if np.count_nonzero(seedPatternVector)>=1:
                # normalize pattern vector
                n = norm(seedPatternVector)
                patternVectorNorm = np.divide(seedPatternVector,n)
                assert not np.isnan(patternVectorNorm).any()
                # accumulate
                pattern_vector_centroid += patternVectorNorm

    pattern_vector_centroid /= len(seeds)

    for other in iteration_sentences:
        if len(other.patterns_bet)==1:
            otherPatternVector = other.patterns_bet_vectors[0]
            # make sure there are no NaN and no 0s-only vectors
            assert not np.isnan(otherPatternVector).any()
            if np.count_nonzero(otherPatternVector)>=1:
                n = norm(otherPatternVector)
                otherPatternVectorNorm = np.divide(otherPatternVector,n)
                assert not np.isnan(otherPatternVectorNorm).any()
                score = cosine_similarity(otherPatternVectorNorm,pattern_vector_centroid)[0][0]
                if score>=threshold:
                    selected.append(other)

    return selected


"""
 Average inter-cluster distance
"""
def inter_distance(cluster):
    all_vectors = []
    
    for v in cluster:
        #TODO: using only one between pattern
        all_vectors.append(v.patterns_bet_vectors[0])
    
    matrix = pairwise.pairwise_distances(np.array(all_vectors),metric='cosine')    
    avg = 0
    for row in matrix:
        tmp = 0
        for c in row:
            tmp += c
        avg = tmp
        
    return avg / float(len(cluster))


if __name__ == "__main__":
    main()








