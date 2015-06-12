#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import re
from nltk import word_tokenize, pos_tag


class Relationship:
    def __init__(self, _sentence, _before, _between, _after, _ent1, _ent2):
        self.sentence = _sentence
        self.before = _before
        self.between = _between
        self.after = _after
        self.ent1 = _ent1
        self.ent2 = _ent2

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
        """
        - finds named-entities
        - for each pair of entities checks:
            - if semantic type matches the relationship we want to find
            - if the entities are not further away more than max_tokens and exits at least min_tokens between them
        - If checks are valid it PoS-tags the sentence
        - Splits the sentence/context into 3 contexts: BEFORE, BETWEEN, AFTER for a given pair of entities
        - Builds a relationship
        """

        self.relationships = list()
        self.tagged = None
        if config is None:
            entities_regex = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)
            tags_regex = re.compile('</?[A-Z]+>', re.U)

        # find named-entities
        entities = []

        if config is None:
            for m in re.finditer(entities_regex, _sentence):
                entities.append(m)

        else:
            for m in re.finditer(config.entities_regex, _sentence):
                entities.append(m)

        if len(entities) >= 2:
            for x in range(0, len(entities) - 1):
                ent1 = entities[x].group()
                ent2 = entities[x+1].group()
                arg1 = re.sub("</?[A-Z]+>", "", ent1)
                arg2 = re.sub("</?[A-Z]+>", "", ent2)
                arg1match = re.match("<[A-Z]+>", ent1)
                arg2match = re.match("<[A-Z]+>", ent2)
                arg1type = arg1match.group()[1:-1]
                arg2type = arg2match.group()[1:-1]

                if (arg1type != e1_type or arg2type != e2_type) or (arg1 == arg2):
                    continue

                else:
                    # only consider relationships where the distance between the two entities
                    # is less than 'max_tokens' and greater than 'min_tokens'
                    between = _sentence[entities[x].end():entities[x + 1].start()]
                    number_bet_tokens = len(word_tokenize(between))
                    if number_bet_tokens > max_tokens or number_bet_tokens < min_tokens:
                        continue

                    else:
                        arg1_parts = arg1.split()
                        arg2_parts = arg2.split()

                        if self.tagged is None:
                            if config is None:
                                sentence_no_tags = re.sub(tags_regex, "", _sentence)
                            else:
                                sentence_no_tags = re.sub(config.tags_regex, "", _sentence)

                            text_tokens = word_tokenize(sentence_no_tags)
                            self.tagged = pos_tag(text_tokens)

                        # to split the tagged sentence into contexts, preserving the PoS-tags
                        # has to take into consideration multi-word entities
                        # NOTE: this works, but probably can be done in a much cleaner way
                        before_i = 0

                        for i in range(0, len(self.tagged)):
                            j = i
                            z = 0
                            while (z <= len(arg1_parts)-1) and self.tagged[j][0] == arg1_parts[z]:
                                j += 1
                                z += 1
                            if z == len(arg1_parts):
                                before_i = i
                                break

                        for i in range(before_i, len(self.tagged)):
                            j = i
                            z = 0
                            while (z <= len(arg2_parts)-1) and self.tagged[j][0] == arg2_parts[z]:
                                j += 1
                                z += 1
                            if z == len(arg2_parts):
                                after_i = i
                                break

                        before_tags = self.tagged[:before_i]
                        between_tags = self.tagged[before_i+len(arg1_parts):after_i]
                        after_tags = self.tagged[after_i+len(arg2_parts):]
                        before_tags_cut = before_tags[-window_size:]
                        after_tags_cut = after_tags[:window_size]

                        r = Relationship(_sentence, before_tags_cut, between_tags, after_tags_cut, arg1, arg2)
                        self.relationships.append(r)


class SentenceParser:

    def __init__(self, _sentence, e1_type, e2_type, config=None):
        self.relationships = set()
        self.sentence = _sentence
        self.entities = list()
        self.valid = False
        self.tree = None
        self.deps = None

        if config is None:
            entities_regex = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)

        for m in re.finditer(entities_regex, self.sentence):
            self.entities.append(m.group())

        for e1 in self.entities:
            for e2 in self.entities:
                if e1 == e2:
                    continue
                arg1match = re.match("<([A-Z]+)>", e1)
                arg2match = re.match("<([A-Z]+)>", e2)
                if arg1match.group(1) == e1_type and arg2match.group(1) == e2_type:
                    self.valid = True
                    break;