#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import codecs
import StringIO

from gensim import matutils
from gensim.models import Word2Vec
from numpy import dot
from numpy import zeros
from nltk import load, WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords

# tokens between entities which do not represent relationships
bad_tokens = [",", "(", ")", ";", "''",  "``", "'s", "-", "vs.", "v", "'", ":", ".", "--"]
stopwords = stopwords.words('english')
not_valid = bad_tokens + stopwords

lmtzr = WordNetLemmatizer()
aux_verbs = ['be']

# http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html
# select everything except stopwords, ADJ and ADV
filter_pos = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB']


def extract_reverb_patterns_tagged_ptb(tagged_text):
    """
    Extract ReVerb relational patterns
    http://homes.cs.washington.edu/~afader/bib_pdf/emnlp11.pdf
    """

    # The pattern limits the relation to be a verb (e.g., invented), a verb followed immediately by
    # a preposition (e.g., located in), or a verb followed by nouns, adjectives, or adverbs ending in a preposition
    # (e.g., has an atomic weight of).

    # V | V P | V W*P
    # V = verb particle? adv?
    # W = (noun | adj | adv | pron | det)
    # P = (prep | particle | inf. marker)

    patterns = []
    patterns_tags = []
    i = 0
    limit = len(tagged_text)-1
    tags = tagged_text

    verb = ['VB', 'VBD', 'VBD|VBN', 'VBG', 'VBG|NN', 'VBN', 'VBP', 'VBP|TO', 'VBZ', 'VP']
    adverb = ['RB', 'RBR', 'RBS', 'RB|RP', 'RB|VBG', 'WRB']
    particule = ['POS', 'PRT', 'TO', 'RP']
    noun = ['NN', 'NNP', 'NNPS', 'NNS', 'NN|NNS', 'NN|SYM', 'NN|VBG', 'NP']
    adjectiv = ['JJ', 'JJR', 'JJRJR', 'JJS', 'JJ|RB', 'JJ|VBG']
    pronoun = ['WP', 'WP$', 'PRP', 'PRP$', 'PRP|VBP']
    determiner = ['DT', 'EX', 'PDT', 'WDT']
    adp = ['IN', 'IN|RP']

    # TODO: detect negations
    # ('rejected', 'VBD'), ('a', 'DT'), ('takeover', 'NN')

    while i <= limit:
        tmp = StringIO.StringIO()
        tmp_tags = []

        # a ReVerb pattern always starts with a verb
        if tags[i][1] in verb:

            tmp.write(tags[i][0]+' ')
            t = (tags[i][0], tags[i][1])
            tmp_tags.append(t)
            i += 1

            # V = verb particle? adv? (also capture auxiliary verbs)
            while i <= limit and (tags[i][1] in verb or tags[i][1] in adverb or tags[i][1] in particule):
                tmp.write(tags[i][0]+' ')
                t = (tags[i][0], tags[i][1])
                tmp_tags.append(t)
                i += 1

            # W = (noun | adj | adv | pron | det)
            while i <= limit and (tags[i][1] in noun or tags[i][1] in adjectiv or tags[i][1] in adverb or
                                  tags[i][1] in pronoun or tags[i][1] in determiner):
                tmp.write(tags[i][0]+' ')
                t = (tags[i][0], tags[i][1])
                tmp_tags.append(t)
                i += 1

            # P = (prep | particle | inf. marker)
            while i <= limit and (tags[i][1] in adp or tags[i][1] in particule):
                tmp.write(tags[i][0]+' ')
                t = (tags[i][0], tags[i][1])
                tmp_tags.append(t)
                i += 1

            # add the build pattern to the list collected patterns
            patterns.append(tmp.getvalue())
            patterns_tags.append(tmp_tags)
        i += 1

    # Finally, if the pattern matches multiple adjacent sequences, we merge them into a single relation phrase
    # (e.g.,wants to extend). This refinement enables the model to readily handle relation phrases containing
    # multiple verbs.

    merged_patterns_tags = [item for sublist in patterns_tags for item in sublist]
    return merged_patterns_tags


def detect_passive_voice(pattern):
    passive_voice = False
    #TODO: há casos mais complexos, adjectivos ou adverbios pelo meio, por exemplo:
    # (to be) + (adj|adv) + past_verb + by
    # to be + past verb + by
    if len(pattern) >= 3:
        if pattern[0][1].startswith('V'):
            verb = lmtzr.lemmatize(pattern[0][0], 'v')
            if verb in aux_verbs:
                if (pattern[1][1] == 'VBN' or pattern[1][1] == 'VBD') and pattern[-1][0] == 'by':
                    passive_voice = True

                # past verb + by
                elif (pattern[-2][1] == 'VBN' or pattern[-2][1] == 'VBD') and pattern[-1][0] == 'by':
                    passive_voice = True

            # past verb + by
            elif (pattern[-2][1] == 'VBN' or pattern[-2][1] == 'VBD') and pattern[-1][0] == 'by':
                    passive_voice = True

    # past verb + by
    elif len(pattern) >= 2:
        if (pattern[-2][1] == 'VBN' or pattern[-2][1] == 'VBD') and pattern[-1][0] == 'by':
            passive_voice = True

    return passive_voice


class Tuple(object):

        def __init__(self, _e1, _e2, _sentence, _before, _between, _after):
            self.e1 = _e1
            self.e2 = _e2
            self.sentence = _sentence
            self.confidence = 0
            self.bef_tags = _before
            self.bet_tags = _between
            self.aft_tags = _after
            self.bef_words = " ".join([x[0] for x in self.bef_tags])
            self.bet_words = " ".join([x[0] for x in self.bet_tags])
            self.aft_words = " ".join([x[0] for x in self.aft_tags])
            self.bef_vector = None
            self.bet_vector = None
            self.aft_vector = None
            self.passive_voice = False
            self.construct_vectors()

        def __str__(self):
            return str(self.e1+'\t'+self.e2+'\t'+self.bef_words+'\t'+self.bet_words+'\t'+self.aft_words).encode("utf8")

        def __hash__(self):
            return hash(self.e1) ^ hash(self.e2) ^ hash(self.bef_words) ^ hash(self.bet_words) ^ hash(self.aft_words)

        def __eq__(self, other):
            return (self.e1 == other.e1 and self.e2 == other.e2 and self.bef_words == other.bef_words and
                    self.bet_words == other.bet_words and self.aft_words == other.aft_words)

        def __cmp__(self, other):
            if other.confidence > self.confidence:
                return -1
            elif other.confidence < self.confidence:
                return 1
            else:
                return 0

        def construct_pattern_vector(self, pattern_tags):
            # remove stopwords and adjectives
            pattern = [t[0] for t in pattern_tags if t[0].lower() not in stopwords and t[1] not in filter_pos]
            return self.pattern2vector_sum(pattern)

        def construct_words_vectors(self, tagged_words, context):
            # remove stopwords and adjective
            words = [t[0] for t in tagged_words if t[0].lower() not in stopwords and t[1] not in filter_pos]
            if len(words) >= 1:
                vector = self.pattern2vector_sum(words)
                if context == 'before':
                    self.bef_vector = vector
                elif context == 'between':
                    self.bet_vector = vector
                elif context == 'after':
                    self.aft_vector = vector

        def construct_vectors(self):

            print "\n"
            print self.sentence
            print self.e1, self.e2
            print self.bef_tags
            print self.bet_tags
            print self.aft_tags
            print

            reverb_pattern = extract_reverb_patterns_tagged_ptb(self.bet_tags)

            if len(reverb_pattern) > 0:
                self.passive_voice = detect_passive_voice(reverb_pattern)
                self.bet_vector = self.construct_pattern_vector(reverb_pattern)
            else:
                self.passive_voice = False
                self.construct_words_vectors(reverb_pattern, "between")

            # extract words before the first entity, and words after the second entity
            if len(self.bef_tags) > 0:
                self.construct_words_vectors(self.bef_tags, "before")
            if len(self.aft_tags) > 0:
                self.construct_words_vectors(self.aft_tags, "after")

        @staticmethod
        def pattern2vector_sum(tokens):
            pattern_vector = zeros(200)
            if len(tokens) > 1:
                for t in tokens:
                    try:
                        vector = word2vec[t[0].strip()]
                        pattern_vector += vector
                    except KeyError:
                        continue

            elif len(tokens) == 1:
                print tokens
                print tokens[0].strip()
                try:
                    pattern_vector = word2vec[tokens[0].strip()]
                except KeyError:
                    pass

            return pattern_vector


def tokenize_entity(entity):
    parts = word_tokenize(entity)
    if parts[-1] == '.':
        replace = parts[-2] + parts[-1]
        del parts[-1]
        del parts[-1]
        parts.append(replace)
    return parts


def find_locations(entity_string, text_tokens):
    locations = []
    e_parts = tokenize_entity(entity_string)
    for i in range(len(text_tokens)):
        if text_tokens[i:i + len(e_parts)] == e_parts:
            locations.append(i)
    return e_parts, locations


class EntitySimple:
    def __init__(self, _e_string, _e_parts, _e_type, _locations):
        self.string = _e_string
        self.parts = _e_parts
        self.type = _e_type
        self.locations = _locations

    def __hash__(self):
        return hash(self.string) ^ hash(self.type)

    def __eq__(self, other):
        return self.string == other.string and self.type == other.type


class Relationship:
    def __init__(self, _sentence, _before, _between, _after, _ent1, _ent2, e1_type, e2_type):
        self.sentence = _sentence
        self.before = _before
        self.between = _between
        self.after = _after
        self.e1 = _ent1
        self.e2 = _ent2
        self.e1_type = e1_type
        self.e2_type = e2_type

    def __eq__(self, other):
        if self.e1 == other.e1 and self.before == other.before and self.between == other.between \
                and self.after == other.after:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.e1) ^ hash(self.e2) ^ hash(self.before) ^ hash(self.between) ^ hash(self.after)


class Sentence:

    def __init__(self, sentence, e1_type, e2_type, max_tokens, min_tokens, window_size, pos_tagger=None):
        self.relationships = list()
        self.tagged_text = None

        #determine which type of regex to use according to how named-entties are tagged
        entities_regex = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)

        # find named-entities
        entities = []
        for m in re.finditer(entities_regex, sentence):
            entities.append(m)

        if len(entities) >= 2:
            # clean tags from text
            sentence_no_tags = re.sub(re.compile('</?[A-Z]+>', re.U), "", sentence)
            text_tokens = word_tokenize(sentence_no_tags)

            # extract information about the entity, create an Entity instance and store in a
            # structure to hold information collected about all the entities in the sentence
            entities_info = set()
            for x in range(0, len(entities)):
                entity = entities[x].group()
                e_string = re.findall('<[A-Z]+>([^<]+)</[A-Z]+>', entity)[0]
                e_type = re.findall('<([A-Z]+)', entity)[0]
                e_parts, locations = find_locations(e_string, text_tokens)
                e = EntitySimple(e_string, e_parts, e_type, locations)
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
                if max_tokens >= distance >= min_tokens and e1.type == e1_type and e2.type == e2_type:

                    # ignore relationships between the same entity
                    if e1.string == e2.string:
                        continue

                    # run PoS-tagger over the sentence only onces
                    if self.tagged_text is None:
                        # split text into tokens and tag them using NLTK's default English tagger
                        # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
                        self.tagged_text = pos_tagger.tag(text_tokens)

                    before = self.tagged_text[:sorted_keys[i]]
                    before = before[-window_size:]
                    between = self.tagged_text[sorted_keys[i]+len(e1.parts):sorted_keys[i+1]]
                    after = self.tagged_text[sorted_keys[i+1]+len(e2.parts):]
                    after = after[:window_size]

                    # ignore relationships where BET context is only stopwords or other invalid words
                    if all(x in not_valid for x in text_tokens[sorted_keys[i]+len(e1.parts):sorted_keys[i+1]]):
                        continue

                    r = Relationship(sentence, before, between, after, e1.string, e2.string, e1_type, e2.type)
                    self.relationships.append(r)


def similarity_3_contexts(p, t):
        (bef, bet, aft) = (0, 0, 0)

        if t.bef_vector is not None and p.bef_vector is not None:
            bef = dot(matutils.unitvec(t.bef_vector), matutils.unitvec(p.bef_vector))

        if t.bet_vector is not None and p.bet_vector is not None:
            bet = dot(matutils.unitvec(t.bet_vector), matutils.unitvec(p.bet_vector))

        if t.aft_vector is not None and p.aft_vector is not None:
            aft = dot(matutils.unitvec(t.aft_vector), matutils.unitvec(p.aft_vector))

        return 0*bef + 1*bet + 0*aft


def main():
    print "Loading word2vec"
    global word2vec
    word2vec = Word2Vec.load_word2vec_format(sys.argv[2], binary=True)
    tagger = load('taggers/maxent_treebank_pos_tagger/english.pickle')
    f_sentences = codecs.open(sys.argv[1], encoding='utf-8')
    invalid = list()
    valid = list()
    on = False
    for line in f_sentences:
        if line.startswith("#"):
            continue
        if line.startswith("VALID"):
            on = True
            continue
        sentence = Sentence(line.strip(), "ORG", "LOC", 6, 1, 2, tagger)
        for rel in sentence.relationships:
            t = Tuple(rel.e1, rel.e2, rel.sentence, rel.before, rel.between, rel.after)
            if on is True:
                valid.append(t)
            elif on is False:
                invalid.append(t)
    f_sentences.close()

    for v in valid:
        for i in invalid:
            score = similarity_3_contexts(v, i)
            print "VALID", v.e1, v.e2, '\t', v.bet_words
            print "INVALID", i.e1, i.e2, '\t', i.bet_words
            print score

if __name__ == "__main__":
    main()