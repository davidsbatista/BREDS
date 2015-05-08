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
    for f in files:
        if f.endswith('.json'):
            #print join(directory, f)
            with open(join(directory, f)) as data_file:
                data = json.load(data_file)
                # first load the entities identified into a strucutre which
                # can be searched as we process each sentence
                entities_data = data['entityMetadata']

                # go through all sentences and select only sentences that have at least two entities
                # grounded to dbpedia/wikipedia and whose type is organisation, person or location
                sentences = data['annotatedText'].split('\n')
                for s in sentences:
                    number_valid_entities = 0
                    valid_entities = set()
                    entities_in_dbpedia = 0
                    persons_s = set()
                    organisations_s = set()
                    places_s = set()

                    # extracte all entities in a sentence with a regex
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
                            #print "NOT FOUND", entity_wiki_url
                            #print "\n"
                            pass

                    if entities_in_dbpedia >= 2:
                        for e in valid_entities:
                            entity_id = e.split('[[')[1].split('|')[0]
                            entity_wiki_url = entities_data[entity_id]['url']
                            entity_name = re.search(r'\|(.+)\]\]', e).group(1)

                            if e in persons_s:
                                s = s.replace(e, "<PER url="+entity_wiki_url+">"+entity_name.encode("utf8")+"</PER>")
                            elif e in places_s:
                                s = s.replace(e, "<LOC url="+entity_wiki_url+">"+entity_name.encode("utf8")+"</LOC>")
                            elif e in organisations_s:
                                s = s.replace(e, "<ORG url="+entity_wiki_url+">"+entity_name.encode("utf8")+"</ORG>")
                            else:
                                # entities that are grounded to wikipedia but are not selected
                                # i.e., they are not classified as person, location or organisation
                                print "No found in persons, places, orgs", e
                                sys.exit(0)

                        print urllib2.unquote(s.encode("utf8"))
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
