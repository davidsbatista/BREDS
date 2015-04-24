#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pprint import pprint
import re
import urllib2

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import codecs
import sys
import json
import multiprocessing

from os import listdir
from os.path import isfile, join

# mappingbased_properties_en.nt -> relationships between entities
# instance_types_en.nt -> entities types


def load_dbpedia_entities(data):
    # parse instance_types_en.nt and select only person, locations and organizations
    persons = set()
    organisations = set()
    locations = set()
    print "Loading entities from DBpedia"
    with codecs.open(data, 'rb', encoding='utf-8') as f:
        for line in f:
            if '<http://dbpedia.org/ontology/Organisation>' in line:
                organisations.add(urllib2.unquote(line.split()[0].encode("utf8")))
            elif '<http://dbpedia.org/ontology/Person>' in line:
                persons.add(urllib2.unquote(line.split()[0].encode("utf8")))
            elif '<http://dbpedia.org/ontology/Place>' in line:
                locations.add(urllib2.unquote(line.split()[0].encode("utf8")))


def load_sentences(directory):
    #sentences = multiprocessing.Manager.Queue()
    #count = 0
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    for f in files:
        if f.endswith('.json'):
            print join(directory, f)
            with open(join(directory, f)) as data_file:
                data = json.load(data_file)
                # load the entities identified
                entities_data = data['entityMetadata']

                # select only sentences that have at least two entities grounded to dbpedia/wikipedia
                # and whose type is organisation, person or location
                sentences = data['annotatedText'].split('\n')
                for s in sentences:
                    number_valid_entities = 0
                    valid_entities = set()
                    wikilink_rx = re.compile(r'\[\[[^\]]+\]\]')
                    entities = re.findall(wikilink_rx, s)
                    for e in entities:
                        entity_id = e.split('[[')[1].split('|')[0]
                        try:
                            if 'url' in entities_data[entity_id]:
                                valid_entities.add(entity_id)
                                number_valid_entities += 1
                        except KeyError:
                            pass
                    if number_valid_entities >= 2:
                        for e in valid_entities:
                            s = s.replace(e, entities_data[e]['url'])
                        print "dbpedia url"
                        print urllib2.unquote(s.encode("utf8"))
                        print "==================================="


def main():
    #load_dbpedia_entities(sys.argv[1])
    load_sentences(sys.argv[2])


if __name__ == "__main__":
    main()
