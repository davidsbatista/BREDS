#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'dsbatista'
__email__ = "dsbatista@inesc-id.pt"

import re
import random
import fileinput
import sys

from nltk import PunktWordTokenizer, pos_tag, map_tag
from nltk.corpus import stopwords

from BREDS.Tuple import Tuple
from Common.ReVerb import Reverb
from Common.Sentence import Sentence

from numpy.linalg import norm
from numpy import dot, zeros

from gensim.models import Word2Vec
from gensim import matutils


VECTOR_DIM = 200
# http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html
# select everything except stopwords, ADJ and ADV
filter_pos = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB']
stopwords = stopwords.words('english')


class TupleVectors(object):

    def __init__(self, before, between, after):
        self.bef_vector = before
        self.bet_vector = between
        self.aft_vector = after


def read_sentences(data):
    relationships = list()
    for line in fileinput.input(data):
        if line.startswith("sentence: "):
            sentence = Sentence(line.split("sentence: ")[1].strip(), 'ORG', 'ORG', 6, 1, 2)

            for r in sentence.relationships:
                relationships.append(r)
    fileinput.close()
    return relationships


def sample(patterns, n):
    samples = list()
    for i in range(0, n):
        e = random.choice(patterns)
        samples.append(e)
        del patterns[patterns.index(e)]
    return samples


def pattern2vector_sum(tokens):
    """
    Generate word2vec vectors based on words that mediate the relationship
    which can be ReVerb patterns or the words around the entities
    """
    # sum each word
    pattern_vector = zeros(VECTOR_DIM)

    if len(tokens) > 1:
        for t in tokens:
            try:
                vector = word2vec[t.strip()]
                pattern_vector += vector
            except KeyError:
                continue

    elif len(tokens) == 1:
        try:
            pattern_vector = word2vec[tokens[0].strip()]
        except KeyError:
            pass

    return pattern_vector


def construct_words_vectors(words):
    # split text into tokens and tag them using NLTK's default English tagger
    # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
    text_tokens = PunktWordTokenizer().tokenize(words)
    tags_ptb = pos_tag(text_tokens)

    pattern = [t[0] for t in tags_ptb if t[0].lower() not in stopwords and t[1] not in filter_pos]
    words_vector = pattern2vector_sum(pattern)

    return words_vector


def construct_pattern_vector(pattern_tags):
    # remove stopwords and adjectives
    pattern = [t[0] for t in pattern_tags if t[0].lower() not in stopwords and t[1] not in filter_pos]
    if len(pattern) >= 1:
        words_vector = pattern2vector_sum(pattern)

    return words_vector


def process_sentence(rel):
    """
    - PoS-taggs a sentence
    - Splits the sentence into 3 contexts: BEFORE,BETWEEN,AFTER
    """

    print rel.sentence
    # tag the sentence, using the default NTLK English tagger
    # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
    entities_regex = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)
    tags_regex = re.compile('</?[A-Z]+>', re.U)

    sentence_no_tags = re.sub(tags_regex, "", rel.sentence)
    text_tokens = PunktWordTokenizer().tokenize(sentence_no_tags)
    tagged = pos_tag(text_tokens)

    # List of PoS-tags here: http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html
    tags = []
    for t in tagged:
        #tag = map_tag('en-ptb', 'universal', t[1])
        #tags.append((t[0], tag))
        tags.append((t[0], t[1]))

    # find named-entities offset
    matches = []
    for m in re.finditer(entities_regex, rel.sentence):
        matches.append(m)

    # extract contexts along with PoS-Tags
    for x in range(0, len(matches)-1):
        if x == 0:
            start = 0
        if x > 0:
            start = matches[x-1].end()
        try:
            end = matches[x+2].start()
        except IndexError:
            end = len(rel.sentence)-1

        before = rel.sentence[start:matches[x].start()]
        between = rel.sentence[matches[x].end():matches[x+1].start()]
        after = rel.sentence[matches[x+1].end(): end]
        ent1 = matches[x].group()
        ent2 = matches[x+1].group()
        arg1 = re.sub("</?[A-Z]+>", "", ent1)
        arg2 = re.sub("</?[A-Z]+>", "", ent2)
        rel.arg1 = arg1
        rel.arg2 = arg2
        quote = False
        #TODO: isto tem que ser generalizado
        #TODO: testar este caso
        # "Private equity firm <ORG>Terra Firma Capital Partners</ORG> acquired <ORG>EMI</ORG> last week for $ 4.9 billion ( euro3 .55 billion ) ."
        bgn_e2 = rel.sentence.index("<ORG>")
        end_e2 = rel.sentence.index("</ORG>")
        if (rel.sentence[bgn_e2-1]) == "'":
            quote = True
        if (rel.sentence[end_e2+len("</e2>")]) == "'":
            quote = True
        arg1_parts = arg1.split()
        arg2_parts = arg2.split()
        if quote:
            new = []
            for e in arg2_parts:
                if e.startswith("'") or e.endswith("'"):
                    e = '"'+e+'"'
                    new.append(e)
            arg2_parts = new
        rel.before = before
        rel.between = between
        rel.after = after

        # to split the tagged sentence into contexts, preserving the PoS-tags
        # has to take into consideration multi-word entities
        # NOTE: this works, but probably can be done in a much cleaner way
        before_i = 0
        for i in range(0, len(tags)):
            j = i
            z = 0
            while (z <= len(arg1_parts)-1) and tags[j][0] == arg1_parts[z]:
                j += 1
                z += 1
            if z == len(arg1_parts):
                before_i = i
                break
        for i in range(before_i, len(tags)):
            j = i
            z = 0
            while (z <= len(arg2_parts)-1) and tags[j][0] == arg2_parts[z]:
                j += 1
                z += 1
            if z == len(arg2_parts):
                after_i = i
                break
        before_tags = tags[:before_i]

        if len(arg1_parts) > 1:
            between_tags = tags[before_i+2:after_i]
            after_tags = tags[after_i+1:]
        elif len(arg2_parts) > 1:
            between_tags = tags[before_i+1:after_i]
            after_tags = tags[after_i+2:]
        else:
            between_tags = tags[before_i+1:after_i]
            after_tags = tags[after_i+1:]

        print ent1
        print ent2
        print "BEF", before_tags
        print "BET", between_tags
        print "AFT", after_tags
        print "\n"


def construct_vectors(relationships, reverb):
    vectors = list()
    for r in relationships:
        bef_vector = None
        aft_vector = None
        """
        bet_vector = None
        print r.sentence
        print r.before
        print r.between
        print r.after
        """

        patterns_bet_tags = reverb.extract_reverb_patterns_ptb(r.between)
        if len(patterns_bet_tags) > 0:
            passive_voice = reverb.detect_passive_voice(patterns_bet_tags)
            # forced hack since _'s_ is always tagged as VBZ, (u"'s", 'VBZ') and causes ReVerb to identify
            # a pattern which is wrong, if this happens, ignore that a pattern was extracted
            if patterns_bet_tags[0][0] == "'s":
                bet_vector = construct_words_vectors(r.between)
            else:
                bet_vector = construct_pattern_vector(patterns_bet_tags)
        else:
            bet_vector = construct_words_vectors(r.between)

            # extract words before the first entity, and words after the second entity
            if len(r.before) > 0:
                bef_vector = construct_words_vectors(r.before)

            if len(r.after) > 0:
                aft_vector = construct_words_vectors(r.after)

        t = TupleVectors(bef_vector, bet_vector, aft_vector)
        vectors.append(t)

    return vectors


def similarity_3_contexts(p1, p2):
        (bef, bet, aft) = (0, 0, 0)

        """
        alpha = 0.2
        beta = 0.6
        gamma = 0.2
        """

        alpha = 0.0
        beta = 1.0
        gamma = 0.0

        if p1.bef_vector is not None and p2.bef_vector is not None:
            bef = dot(matutils.unitvec(p1.bef_vector), matutils.unitvec(p2.bef_vector))

        if p1.bet_vector is not None and p2.bet_vector is not None:
            bet = dot(matutils.unitvec(p1.bet_vector), matutils.unitvec(p2.bet_vector))

        if p1.aft_vector is not None and p2.aft_vector is not None:
            aft = dot(matutils.unitvec(p1.aft_vector), matutils.unitvec(p2.aft_vector))

        return alpha*bef + beta*bet + gamma*aft


def main():
    #word2vec_path=/home/dsbatista/word2vec-read-only/vectors.bin
    #word2vec_path=/home/dsbatista/GoogleNews-vectors-negative300.bin
    #word2vec_path=/home/dsbatista/gigaword/word2vec/afp_apw_vectors.bin
    #word2vec_path=/home/dsbatista/gigaword/word2vec/afp_apw_xing_vectors.bin
    #model = "/home/dsbatista/gigaword/word2vec/afp_apw_xing200.bin"
    #print "Loading word2vec model"
    #global word2vec
    #word2vec = Word2Vec.load_word2vec_format(model, binary=True)
    reverb = Reverb()
    correct = read_sentences(sys.argv[1])
    incorrect = read_sentences(sys.argv[2])

    for r in correct:
        process_sentence(r)

    """
    positive = construct_vectors(correct, reverb)
    negative = construct_vectors(incorrect, reverb)
    print "positive instances", len(positive)
    print "negative insances", len(negative)
    correct_s = sample(positive, 10)
    incorrect_s = sample(negative, 10)
    for p1 in correct_s:
        for p2 in incorrect_s:
            if p1 == p2:
                continue
            else:
                # objective:
                # for positive versus negative patterns:
                #   find a configuration where the global scores in a boxplot where similairity is minimum
                # for positive versus positive or negative versus negative patterns:
                #   find a configuration where the global scores in a boxplot where similairity is maximum
                score = similarity_3_contexts(p1, p2)
                print p1, p2, score
    """

if __name__ == "__main__":
    main()