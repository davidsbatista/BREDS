#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cPickle
import os

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import re
import urllib2
import codecs
import sys
import json
import multiprocessing

from os import listdir
from os.path import isfile, join

# mappingbased_properties_en.nt -> relationships between entities
# instance_types_en.nt -> entities types


def load_dbpedia_entities(data):
    entities = dict()
    with codecs.open(data, 'rb', encoding='utf-8') as f:
        count = 0
        for line in f:
            e = line.split()
            if e[2].startswith('<http://dbpedia.org/ontology/'):
                # add other types of entities to extract different types of relationships
                if e[2].endswith('/Organisation>') or e[2].endswith('/Person>') or e[2].endswith('/Place>'):
                    entity_name = re.search(r'resource/(.*)>', e[0]).group(1)
                    entities[entity_name] = (e[0], e[2])
            count += 1
            if count % 500000 == 0:
                sys.stdout.write(".")

    print len(entities.keys()), "entities loaded"
    return entities


def load_sentences(directory, dbpedia_entities):
    #sentences = multiprocessing.Manager.Queue()
    #count = 0
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    ookbe_regex = re.compile(r'\[\[AIDA:--OOKBE--\|([^\]]+)\]\]')
    for f in files:
        if f.endswith('.json'):
            print join(directory, f)
            with open(join(directory, f)) as data_file:
                data = json.load(data_file)
                # first load the entities identified into a strucutre which
                # can be searched as we process each sentence
                entities_data = data['entityMetadata']

                # go through all the sentences and select only sentences that have at least two entities
                # grounded to dbpedia/wikipedia and whose type is organisation, person or location
                sentences = data['annotatedText'].split('\n')
                for s in sentences:
                    number_valid_entities = 0
                    valid_entities = set()
                    entities_in_dbpedia = 0
                    persons_s = set()
                    organisations_s = set()
                    places_s = set()
                    others = set()

                    # extract all entities in a sentence with a regex
                    wikilink_rx = re.compile(r'\[\[[^\]]+\]\]')
                    entities = re.findall(wikilink_rx, s)

                    # select only entities that are grounded to an URL
                    for e in entities:
                        entity_id = e.split('[[')[1].split('|')[0]
                        try:
                            if 'url' in entities_data[entity_id]:
                                valid_entities.add(e)
                                number_valid_entities += 1
                        except KeyError:
                            pass

                    # from the grounded entities select only those that match the types we want, e.g.: org, loc, per
                    for e in valid_entities:
                        entity_name = re.search(r'\[\[YAGO:(.+)\|', e).group(1)
                        if entity_name in dbpedia_entities:
                            entities_in_dbpedia += 1
                            e_type = dbpedia_entities[entity_name][1]
                            if e_type.endswith('/Organisation>'):
                                organisations_s.add(e)
                            elif e_type.endswith('/Person>'):
                                persons_s.add(e)
                            elif e_type.endswith('/Place>'):
                                places_s.add(e)
                        else:
                            # store other entity types
                            # build an histogram of other entity types
                            # can be usefull to analyze what we are missing
                            print "NOT in DBpedia", entity_name, e
                            others.add(e)

                    if entities_in_dbpedia >= 2:
                        for e in valid_entities:
                            entity_id = e.split('[[')[1].split('|')[0]
                            entity_wiki_url = entities_data[entity_id]['url']
                            entity_name = re.search(r'\|(.+)\]\]', e).group(1)
                            url = entity_wiki_url.replace("%20", "_")

                            # the selected types
                            if e in persons_s:
                                s = s.replace(e, "<PER url="+url+">"+entity_name.encode("utf8")+"</PER>")
                            elif e in places_s:
                                s = s.replace(e, "<LOC url="+url+">"+entity_name.encode("utf8")+"</LOC>")
                            elif e in organisations_s:
                                s = s.replace(e, "<ORG url="+url+">"+entity_name.encode("utf8")+"</ORG>")

                            # other entities, grounded to wikipedia but not part of the selected types
                            elif e in others:
                                s = s.replace(e, entity_name.encode("utf8"))

                        # clean the Out-of-Knowledge base entities, i.e.: AIDA:--OOKBE-
                        sentence_no_ookbe = re.sub(ookbe_regex, r'\g<1>', s)
                        # print sentence to stdout
                        print urllib2.unquote(sentence_no_ookbe.encode("utf8"))
                        #print "==================================="
                        print "\n"


def main():
    if os.path.isfile("dbpedia_entities.pkl"):
        f = open("dbpedia_entities.pkl")
        print "\nLoading selected DBpedia entities from disk"
        dbpedia_entities = cPickle.load(f)
        f.close()
        print len(dbpedia_entities.keys()), "entities loaded"

        # stored as unicode
        #for e in dbpedia_entities.keys():
        #    print type(e)

        load_sentences(sys.argv[2], dbpedia_entities)

    else:
        dbpedia_entities = load_dbpedia_entities(sys.argv[1])
        f = open("dbpedia_entities.pkl", "wb")
        print "Dumping selected DBpedia entities to disk"
        cPickle.dump(dbpedia_entities, f)
        f.close()
        load_sentences(sys.argv[2], dbpedia_entities)

if __name__ == "__main__":
    main()
