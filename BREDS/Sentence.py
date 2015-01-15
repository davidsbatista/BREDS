__author__ = 'dsbatista'

import re
from nltk import PunktWordTokenizer

regex = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)


class Relationship:
    def __init__(self, _sentence, _before=None, _between=None, _after=None, _ent1=None, _ent2=None, _arg1type=None,
                 _arg2type=None, _type=None, _id=None):

        self.sentence = _sentence
        self.identifier = _id
        self.rel_type = _type
        self.before = _before
        self.between = _between
        self.after = _after
        self.ent1 = _ent1
        self.ent2 = _ent2
        self.arg1type = _arg1type
        self.arg2type = _arg2type
        self.sigs = None

        if _before is None and _between is None and _after is None and _sentence is not None:
            matches = []
            for m in re.finditer(regex, self.sentence):
                matches.append(m)

            for x in range(0, len(matches) - 1):
                if x == 0:
                    start = 0
                if x > 0:
                    start = matches[x - 1].end()
                try:
                    end = matches[x + 2].start()
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


class Sentence:
    def __init__(self, _sentence, max_tokens, min_tokens, window_size):
        self.relationships = set()
        self.sentence = _sentence
        matches = []
        for m in re.finditer(regex, self.sentence):
            matches.append(m)

        if len(matches) >= 2:
            for x in range(0, len(matches) - 1):
                if x == 0:
                    start = 0
                if x > 0:
                    start = matches[x - 1].end()
                try:
                    end = matches[x + 2].start()
                except IndexError:
                    end = len(self.sentence) - 1

                before = self.sentence[start:matches[x].start()]
                between = self.sentence[matches[x].end():matches[x + 1].start()]
                after = self.sentence[matches[x + 1].end(): end]

                # select only 'window_size' tokens from left and right context
                before = PunktWordTokenizer().tokenize(before)[-window_size:]
                after = PunktWordTokenizer().tokenize(after)[:window_size]
                before = ' '.join(before)
                after = ' '.join(after)

                # only consider relationships where the distance between the two entities
                # is less than 'max_tokens' and greter than 'min_tokens'
                number_bet_tokens = len(PunktWordTokenizer().tokenize(between))
                if not number_bet_tokens > max_tokens and not number_bet_tokens < min_tokens:
                    ent1 = matches[x].group()
                    ent2 = matches[x + 1].group()
                    arg1match = re.match("<[A-Z]+>", ent1)
                    arg2match = re.match("<[A-Z]+>", ent2)
                    ent1 = re.sub("</?[A-Z]+>", "", ent1, count=2, flags=0)
                    ent2 = re.sub("</?[A-Z]+>", "", ent2, count=2, flags=0)
                    arg1type = arg1match.group()[1:-1]
                    arg2type = arg2match.group()[1:-1]
                    rel = Relationship(_sentence, before, between, after, ent1, ent2, arg1type, arg2type, _type=None,
                                       _id=None)
                    self.relationships.add(rel)
