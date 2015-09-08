#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import sys
import re
from nltk import word_tokenize, pos_tag

# regex for simple tags, e.g.:
# <PER>Bill Gates</PER>
regex_simple = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)

# regex for wikipedia linked tags e.g.:
# <PER url=http://en.wikipedia.org/wiki/Mark_Zuckerberg>Mark Elliot Zuckerberg</PER>
regex_linked = re.compile('<[A-Z]+ url=[^>]+>[^<]+</[A-Z]+>', re.U)


class Relationship:

    def __init__(self, _sentence, _before=None, _between=None, _after=None, _ent1=None, _ent2=None, _arg1type=None,
                 _arg2type=None):

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
            for m in re.finditer(regex_simple, self.sentence):
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

                self.before = self.sentence[start:matches[x].init_bootstrap()]
                self.between = self.sentence[matches[x].end():matches[x + 1].init_bootstrap()]
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
        #TODO: handle othe type of tagged entities
        entities_regex = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)
        tags_regex = re.compile('</?[A-Z]+>', re.U)

        # find named-entities
        entities = []
        for m in re.finditer(entities_regex, _sentence):
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
                        # hard-coded examples, because tokenizer splits some entities with points.
                        # e.g.: "U.S" becomes: [u'U.S, u'.']
                        if arg1 == "U.S.":
                            arg1_parts = [arg1]
                        else:
                            arg1_parts = word_tokenize(arg1)

                        if arg2 == "U.S.":
                            arg2_parts = [arg2]
                        else:
                            arg2_parts = word_tokenize(arg2)

                        if self.tagged is None:
                            sentence_no_tags = re.sub(tags_regex, "", _sentence)

                            # split text into tokens and tag them using NLTK's default English tagger
                            # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
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
                            #print "self.tagged[j][0]", self.tagged[j][0]
                            #print "arg2_parts[z]", arg2_parts[z]
                            while (z <= len(arg2_parts)-1) and self.tagged[j][0] == arg2_parts[z]:
                                #print "self.tagged[j][0]", self.tagged[j][0]
                                #print "arg2_parts[z]", arg2_parts[z]
                                j += 1
                                z += 1
                            if z == len(arg2_parts):
                                after_i = i
                                break

                        try:
                            before_tags = self.tagged[:before_i]
                            between_tags = self.tagged[before_i+len(arg1_parts):after_i]
                            after_tags = self.tagged[after_i+len(arg2_parts):]
                            before_tags_cut = before_tags[-window_size:]
                            after_tags_cut = after_tags[:window_size]

                        except Exception, e:
                            print e
                            print _sentence
                            sys.exit(0)

                        r = Relationship(_sentence, before_tags_cut, between_tags, after_tags_cut, arg1, arg2,
                                         arg1type, arg2type)
                        self.relationships.append(r)