#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import sys
import re
from nltk import word_tokenize, pos_tag


class Relationship:

    def __init__(self, _sentence, _before=None, _between=None, _after=None, _ent1=None, _ent2=None, _arg1type=None,
                 _arg2type=None, config=None):

        self.sentence = _sentence
        self.before = _before
        self.between = _between
        self.after = _after
        self.ent1 = _ent1
        self.ent2 = _ent2
        self.arg1type = _arg1type
        self.arg2type = _arg2type

        if _before is None and _between is None and _after is None and _sentence is not None:
            matches = []
            #determine which type of regex to use according to how named-entties are tagged
            entities_regex = None
            if config.tag_type == "simple":
                entities_regex = config.regex_simple
            elif config.tag_type == "linked":
                entities_regex = config.regex_linked
            for m in re.finditer(entities_regex, self.sentence):
                matches.append(m)

            for x in range(0, len(matches) - 1):
                if x == 0:
                    start = 0
                if x > 0:
                    start = matches[x - 1].end()
                try:
                    end = matches[x + 2].init_bootstrap()
                except IndexError:
                    end = len(self.sentence) - 1

                self.before = self.sentence[start:matches[x].start()]
                self.between = self.sentence[matches[x].end():matches[x + 1].start()]
                self.after = self.sentence[matches[x + 1].end(): end]
                self.ent1 = matches[x].group()
                self.ent2 = matches[x + 1].group()
                arg1match = re.match("<[A-Z]+>", self.ent1)
                arg2match = re.match("<[A-Z]+>", self.ent2)
                self.arg1type = arg1match.group()[1:-1]
                self.arg2type = arg2match.group()[1:-1]

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
        - Finds named-entities.
        - For each pair of entities checks:
            - If semantic type matches the relationship type we want to find e.g. (ORG, LOC) pairs.
            - If the entities are not further away more than 'max_tokens' and exits at least 'min_tokens' between them.
        - If it passes the checks, then extract PoS-tags.
        - Splits the sentence/context into 3 contexts: BEFORE, BETWEEN, AFTER based on the entities position.
        - Builds a relationship.
        """

        self.relationships = list()
        self.tagged = None

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
            for x in range(0, len(entities) - 1):

                if config.tag_type == "simple":
                    ent1 = entities[x].group()
                    ent2 = entities[x+1].group()
                    ent1 = re.sub("</?[A-Z]+>", "", ent1)
                    ent2 = re.sub("</?[A-Z]+>", "", ent2)
                    arg1match = re.match("<[A-Z]+>", ent1)
                    arg2match = re.match("<[A-Z]+>", ent2)
                    arg1type = arg1match.group()[1:-1]
                    arg2type = arg2match.group()[1:-1]

                elif config.tag_type == "linked":
                    ent1 = entities[x].group()
                    ent2 = entities[x+1].group()
                    arg1type = re.findall('<([A-Z]+)', entities[x].group())[0]
                    arg2type = re.findall('<([A-Z]+)', entities[x+1].group())[0]

                if (arg1type != e1_type or arg2type != e2_type) or (ent1 == ent2):
                    continue

                else:
                    # only consider relationships where the distance between the two entities
                    # is less than 'max_tokens' and greater than 'min_tokens'
                    between = _sentence[entities[x].end():entities[x + 1].start()]
                    number_bet_tokens = len(word_tokenize(between))
                    if number_bet_tokens > max_tokens or number_bet_tokens < min_tokens:
                        continue

                    else:
                        # hard-coded examples, because tokenizer splits some entities with points.
                        problematic_entities = ['Ind.', 'U.S.']

                        if config.tag_type == "linked":
                            ent1_string = re.findall('<[A-Z]+ url=[^>]+>([^<]+)</[A-Z]+>', ent1)[0]
                            ent2_string = re.findall('<[A-Z]+ url=[^>]+>([^<]+)</[A-Z]+>', ent2)[0]

                        elif config.tag_type == "simple":
                            ent1_string = ent1
                            ent2_string = ent2

                        if ent1_string in problematic_entities:
                            ent1_parts = [ent1_string]
                        else:
                            if config.tag_type == "simple":
                                ent1_parts = word_tokenize(ent1)
                            elif config.tag_type == "linked":
                                ent1_parts = word_tokenize(re.findall('<[A-Z]+ url=[^>]+>([^<]+)</[A-Z]+>', ent1)[0])

                        if ent2_string in problematic_entities:
                            ent2_parts = [ent2_string]
                        else:
                            if config.tag_type == "simple":
                                ent1_parts = word_tokenize(ent2)
                            elif config.tag_type == "linked":
                                ent2_parts = word_tokenize(re.findall('<[A-Z]+ url=[^>]+>([^<]+)</[A-Z]+>', ent2)[0])

                        if self.tagged is None:
                            # clean tags from text and perform part-of-speech tagging
                            # split text into tokens and tag them using NLTK's default English tagger
                            # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'

                            if config.tag_type == "simple":
                                sentence_no_tags = re.sub(config.tags_regex, "", _sentence)

                            elif config.tag_type == "linked":
                                sentence_no_tags = re.sub(r"</[A-Z]+>|<[A-Z]+ url=[^>]+>", " ", _sentence)

                            try:
                                text_tokens = word_tokenize(sentence_no_tags)
                                self.tagged = pos_tag(text_tokens)
                            except Exception, e:
                                print e
                                print _sentence
                                print sentence_no_tags
                                print ent1_parts
                                print ent2_parts
                                print self.tagged
                                print
                                sys.exit(0)

                        # to split the tagged sentence into contexts, preserving the PoS-tags
                        # has to take into consideration multi-word entities
                        # NOTE: this works, but probably can be done in a much cleaner way
                        before_i = 0

                        for i in range(0, len(self.tagged)):
                            j = i
                            z = 0
                            while (z <= len(ent1_parts)-1) and self.tagged[j][0] == ent1_parts[z]:
                                j += 1
                                z += 1
                            if z == len(ent1_parts):
                                before_i = i
                                break

                        for i in range(before_i, len(self.tagged)):
                            j = i
                            z = 0
                            """
                            print "fora!"
                            print "self.tagged[j][0]", self.tagged[j][0]
                            print "arg2_parts[z]", ent2_parts[z]
                            print
                            """
                            while (z <= len(ent2_parts)-1) and self.tagged[j][0] == ent2_parts[z]:
                                """
                                print "entrei!"
                                print "self.tagged[j][0]", self.tagged[j][0]
                                print "arg2_parts[z]", ent2_parts[z]
                                print
                                """
                                j += 1
                                z += 1
                            if z == len(ent2_parts):
                                after_i = i
                                break

                        try:
                            before_tags = self.tagged[:before_i]
                            between_tags = self.tagged[before_i+len(ent1_parts):after_i]
                            after_tags = self.tagged[after_i+len(ent2_parts):]
                            before_tags_cut = before_tags[-window_size:]
                            after_tags_cut = after_tags[:window_size]

                        except Exception, e:
                            print
                            print e
                            print _sentence
                            print sentence_no_tags
                            print ent1_parts
                            print ent2_parts
                            print self.tagged
                            print
                            sys.exit(0)

                        if config.tag_type == "linked":
                            ent1 = re.findall('url=([^>]+)', ent1)[0]
                            ent2 = re.findall('url=([^>]+)', ent2)[0]

                        r = Relationship(_sentence, before_tags_cut, between_tags, after_tags_cut, ent1, ent2,
                                         arg1type, arg2type, config)
                        self.relationships.append(r)