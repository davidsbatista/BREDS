#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import fileinput
import re

from collections import defaultdict

def process_semeval(data,direction_symmetric):
    print "\nProcessing SemEval dataset"
    relationships = defaultdict(list)
    rel_type = ''
    sentence = ''
    for line in fileinput.input(data):
        if re.match('^[0-9]+\t', line):
            id, sentence = line.strip().split('\t')
        elif not re.match('^Comment', line):
            rel_type = line.strip()
        if rel_type!='' and sentence!='':
            #rel = Relationship(id,sentence,rel_type,None,None)
            rel = Relationship(sentence,rel_type,None,None,None,None,None,None,id)
            sentence = ''
            relationships[rel.rel_type].append(rel)
    fileinput.close()
    
    if direction_symmetric is True:
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

    return relationships


def process_bunescu(data):
    # Dataset created by Bunescu et al. 2007
    # Dataset created by Bunescu et al. 2007 (http://www.cs.utexas.edu/~ml/papers/bunescu-acl07.pdf)
    # Available at: http://knowitall.cs.washington.edu/hlt-naacl08-data.txt
    acquired = [('google', 'youtube'), ('adobe', 'macromedia'),\
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