#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import re
from nltk import word_tokenize, pos_tag


def tokenize_entity(config, entity):
    if config.tag_type == "simple":
        parts = word_tokenize(entity)
    elif config.tag_type == "linked":
        parts = word_tokenize(re.findall('<[A-Z]+ url=[^>]+>([^<]+)</[A-Z]+>', entity)[0])
    if parts[-1] == '.':
        replace = parts[-2] + parts[-1]
        del parts[-1]
        del parts[-1]
        parts.append(replace)
    return parts


def find_locations(config, entity, text_tokens):
    locations = []
    e_parts = tokenize_entity(config, entity)
    for i in range(len(text_tokens)):
        if text_tokens[i:i + len(e_parts)] == e_parts:
            locations.append(i)
    return locations


class Entity:

    def __init__(self, _entity, _e_type, _locations):
        self.entity = _entity
        self.e_type = _e_type
        self.locations = _locations

    def __hash__(self):
        return hash(self.entity) ^ hash(self.e_type)

    def __eq__(self, other):
        return self.entity == other.entity and self.e_type == other.e_type


class Relationship:

    def __init__(self, _sentence, _before, _between, _after, _ent1, _ent2, e1_type, e2_type):

        self.sentence = _sentence
        self.before = _before
        self.between = _between
        self.after = _after
        self.ent1 = _ent1
        self.ent2 = _ent2
        self.arg1type = e1_type
        self.arg2type = e2_type

    def __eq__(self, other):
        if self.ent1 == other.ent1 and self.before == other.before and self.between == other.between \
                and self.after == other.after:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.ent1) ^ hash(self.ent2) ^ hash(self.before) ^ hash(self.between) ^ hash(self.after)


class Sentence:

    def __init__(self, _sentence, e1_type, e2_type, max_tokens, min_tokens, window_size, config=None):
        self.relationships = list()

        #determine which type of regex to use according to how named-entties are tagged
        entities_regex = None
        if config.tag_type == "simple":
            entities_regex = config.regex_simple
        elif config.tag_type == "linked":
            entities_regex = config.regex_linked

        # find named-entities
        entities = []
        for m in re.finditer(entities_regex, _sentence):
            entities.append(m)

        if len(entities) >= 2:

            # clean tags from text and perform part-of-speech tagging
            # split text into tokens and tag them using NLTK's default English tagger
            # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
            if config.tag_type == "simple":
                sentence_no_tags = re.sub(config.tags_regex, "", _sentence)
            elif config.tag_type == "linked":
                sentence_no_tags = re.sub(r"</[A-Z]+>|<[A-Z]+ url=[^>]+>", " ", _sentence)
            text_tokens = word_tokenize(sentence_no_tags)
            tagged_text = pos_tag(text_tokens)

            # structure to hold information collected about the entities in the sentence
            entities_info = set()
            for x in range(0, len(entities)):
                if config.tag_type == "simple":
                    ent1 = entities[x].group()
                    ent1 = re.sub("</?[A-Z]+>", "", ent1)
                    arg1match = re.match("<[A-Z]+>", ent1)
                    e_type = arg1match.group()[1:-1]
                elif config.tag_type == "linked":
                    entity = entities[x].group()
                    e_type = re.findall('<([A-Z]+)', entities[x].group())[0]
                locations = find_locations(config, entity, text_tokens)
                e = Entity(entity, e_type, locations)
                entities_info.add(e)

            # create an hashtable on which:
            # - the key is the starting index in the tokenized sentence of an entity
            # - the value the corresponding Entity instance
            locations = dict()
            for e in entities_info:
                for start in e.locations:
                    locations[start] = e

            # look for pair of entities such that:
            # the distance between the two entities is less than 'max_tokens' and greater than 'min_tokens'
            # the arguments match the seeds semantic types
            sorted_keys = list(sorted(locations))
            for i in range(len(sorted_keys)-1):
                distance = sorted_keys[i+1] - sorted_keys[i]
                e1 = locations[sorted_keys[i]]
                e2 = locations[sorted_keys[i+1]]
                if max_tokens >= distance >= min_tokens and e1.e_type == e1_type and e2.e_type == e2_type:

                    if config.tag_type == "linked":
                        entity1_string = tokenize_entity(config, e1.entity)
                        entity2_string = tokenize_entity(config, e2.entity)
                    elif config.tag_type == "simple":
                        #TODO: teste com simple
                        entity1_string = None
                        entity2_string = None

                    before = tagged_text[:sorted_keys[i]]
                    before = before[-window_size:]
                    between = tagged_text[sorted_keys[i]+len(entity1_string):sorted_keys[i+1]]
                    after = tagged_text[sorted_keys[i+1]+len(entity2_string):]
                    after = after[:window_size]

                    print "BEF", before
                    print "BET", between
                    print "AFT", after
                    print

                    if config.tag_type == "linked":
                        ent1 = re.findall('url=([^>]+)', e1.entity)
                        ent2 = re.findall('url=([^>]+)', e2.entity)

                    elif config.tag_type == "simple":
                        #TODO: teste com simple
                        entity1_string = None
                        entity2_string = None

                    r = Relationship(_sentence, before, between, after, ent1, ent2, e1_type, e2_type)
                    self.relationships.append(r)