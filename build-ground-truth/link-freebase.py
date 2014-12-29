#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'dsbatista'

import fileinput
import os
import sys

from collections import defaultdict
from Sentence import Sentence


def collect_entities(data):
    #######################
    # Weak entity linking #
    #######################
    # load freebase entities to memory
    # select only a few entities, based on the relationships

    rel_to_consider = ['Governance of', 'Leader of', 'Organization founded', 'Organization acquired',
                       'Employment history', 'Venture Investment', 'Peer', 'Place founded', 'Sibling', 'Spouse']

    # wc -l facts
    # 241 897 882

    relationships = defaultdict(list)

    try:
        os.path.isfile("relationships.txt")
        print "\nLoading relationships from disk..."
        for line in fileinput.input("relationships.txt"):
            e1, r, e2 = line.split('\t')
            relationships[(e1, e2.strip())].append(r)
        fileinput.close()

    except IOError:
        count = 0
        for line in fileinput.input(data):
            if count % 50000 == 0:
                print count, "of 241897882\tRelationships: ", len(relationships)
            e1, r, e2, point = line.split('\t')
            if r in rel_to_consider:
                if not e1.startswith('/') and not e2.startswith('/'):
                    relationships[(e1, e2)].append(r)
            count += 1
        fileinput.close()

        print "Writing collected relationships to disk"

        f_entities = open("relationships.txt", "w")
        for r in relationships.keys():
            for e in relationships[r]:
                f_entities.write(r[0]+'\t'+e+'\t'+r[1]+'\n')
        f_entities.close()

    return relationships


def collect_afp_sentences(sentences, freebase_relations, entities):
    # for each sentence/relationship in afp-tagged, containing at least two entities:
    # if entities match freebase entities, select sentence

    for line in fileinput.input(sentences):
        s = Sentence(line.strip())
        if len(s.relationships) > 0:
            for r in s.relationships:
                if r.ent1 in entities and r.ent2 in entities:
                    # r.ent1 and r.ent2 in relationship
                    if len(freebase_relations[(r.ent1, r.ent2)]):
                        print r.sentence
                        print r.ent1, '\t', r.ent2
                        print freebase_relations[(r.ent1, r.ent2)]
                        print "bef", r.before
                        print "bet", r.between
                        print "aft", r.after
                        print "\n"

                        # mapear relações
                        # 1 - string igual?
                        # 2 - usar wordnet e JC
                        # get synsets for all words




def main():
    relationships = collect_entities(sys.argv[1])
    print len(relationships), "relationships loaded"
    entities = set()
    for r in relationships.keys():
        entities.add(r[0])
        entities.add(r[1])
    print len(entities), "unique entities"

    collect_afp_sentences(sys.argv[2], relationships, entities)


if __name__ == "__main__":
    main()