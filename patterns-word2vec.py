#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'dsbatista'
__email__ = "dsbatista@inesc-id.pt"

import sys
import random
import re
import fileinput
import matplotlib as mpl
## agg backend is used to create plot as a .png file
mpl.use('agg')
import matplotlib.pyplot as plt

from nltk import PunktWordTokenizer, pos_tag
from nltk.corpus import stopwords
from Common.ReVerb import Reverb
from numpy import dot, zeros
from gensim.models import Word2Vec
from gensim import matutils


WINDOW_SIZE = 2
VECTOR_DIM = 200
# http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html
# select everything except stopwords, ADJ and ADV
filter_pos = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB']
stopwords = stopwords.words('english')


class TupleVectors(object):

    def __init__(self, _before, _between, _after, _sentence, _ent1, _ent2):
        self.before = _before
        self.between = _between
        self.after = _after
        self.sentence = _sentence
        self.ent1 = _ent1
        self.ent2 = _ent2
        self.bef_vector = zeros(VECTOR_DIM)
        self.bet_vector = zeros(VECTOR_DIM)
        self.aft_vector = zeros(VECTOR_DIM)
        self.passive = None


def read_sentences(data):
    sentences = list()
    for line in fileinput.input(data):
        if line.startswith("sentence: "):
            sentence = line.split("sentence: ")[1].strip()
            t = process_sentence(sentence)
            if t is None:
                print "TupleVector failed"
                print line
                sys.exit(0)
            else:
                sentences.append(t)
    fileinput.close()
    return sentences


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


def construct_words_vectors(tagged_words):
    pattern = [t[0] for t in tagged_words if t[0].lower() not in stopwords and t[1] not in filter_pos]
    words_vector = pattern2vector_sum(pattern)
    return words_vector


def construct_pattern_vector(pattern_tags):
    # remove stopwords and adjectives
    words_vector = zeros(VECTOR_DIM)
    pattern = [t[0] for t in pattern_tags if t[0].lower() not in stopwords and t[1] not in filter_pos]
    if len(pattern) >= 1:
        words_vector = pattern2vector_sum(pattern)
    return words_vector


def process_sentence(sentence):
    """
    - PoS-taggs a sentence
    - Splits the sentence into 3 contexts: BEFORE,BETWEEN,AFTER
    """

    # tag the sentence, using the default NTLK English tagger
    # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
    entities_regex = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)
    tags_regex = re.compile('</?[A-Z]+>', re.U)

    sentence_no_tags = re.sub(tags_regex, "", sentence)
    text_tokens = PunktWordTokenizer().tokenize(sentence_no_tags)
    tagged = pos_tag(text_tokens)

    # find named-entities offset
    matches = []
    for m in re.finditer(entities_regex, sentence):
        matches.append(m)

    if len(matches) > 2:
        print sentence
        print "\n"

    # extract contexts along with PoS-Tags
    for x in range(0, len(matches)-1):
        if x == 0:
            start = 0
        if x > 0:
            start = matches[x-1].end()
        try:
            end = matches[x+2].start()
        except IndexError:
            end = len(sentence)-1

        #before = sentence[start:matches[x].start()]
        #between = sentence[matches[x].end():matches[x+1].start()]
        #after = sentence[matches[x+1].end(): end]
        ent1 = matches[x].group()
        ent2 = matches[x+1].group()
        arg1 = re.sub("</?[A-Z]+>", "", ent1)
        arg2 = re.sub("</?[A-Z]+>", "", ent2)
        arg1_parts = arg1.split()
        arg2_parts = arg2.split()

        if arg1 == arg2:
            continue

        #TODO: fixe de um bug para quando há _'_ dentro da entidade, confirmar que eh necessario
        """
        quote = False
        bgn_e2 = rel.sentence.index("<ORG>")
        end_e2 = rel.sentence.index("</ORG>")
        if (rel.sentence[bgn_e2-1]) == "'":
            quote = True
        if (rel.sentence[end_e2+len("</ORG>")]) == "'":
            quote = True
        if quote:
            new = []
            for e in arg2_parts:
                if e.startswith("'") or e.endswith("'"):
                    e = '"'+e+'"'
                    new.append(e)
            arg2_parts = new
        """

        # to split the tagged sentence into contexts, preserving the PoS-tags
        # has to take into consideration multi-word entities
        # NOTE: this works, but probably can be done in a much cleaner way
        before_i = 0

        for i in range(0, len(tagged)):
            j = i
            z = 0
            while (z <= len(arg1_parts)-1) and tagged[j][0] == arg1_parts[z]:
                j += 1
                z += 1
            if z == len(arg1_parts):
                before_i = i
                break

        for i in range(before_i, len(tagged)):
            j = i
            z = 0
            while (z <= len(arg2_parts)-1) and tagged[j][0] == arg2_parts[z]:
                j += 1
                z += 1
            if z == len(arg2_parts):
                after_i = i
                break

        before_tags = tagged[:before_i]
        between_tags = tagged[before_i+len(arg1_parts):after_i]
        after_tags = tagged[after_i+len(arg2_parts):]
        before_tags_cut = before_tags[-WINDOW_SIZE:]
        after_tags_cut = after_tags[:WINDOW_SIZE]
        t = TupleVectors(before_tags_cut, between_tags, after_tags_cut, sentence, ent1, ent2)
        return t


def construct_vectors(relationships, reverb):
    vectors = list()
    for r in relationships:
        patterns_bet_tags = reverb.extract_reverb_patterns_tagged_ptb(r.between)
        if len(patterns_bet_tags) > 0:
            r.passive = reverb.detect_passive_voice(patterns_bet_tags)
            r.bet_vector = construct_pattern_vector(patterns_bet_tags)
        else:
            r.passive = False
            r.bet_vector = construct_words_vectors(r.between)
            # extract words before the first entity, and words after the second entity
            if len(r.before) > 0:
                r.bef_vector = construct_words_vectors(r.before)

            if len(r.after) > 0:
                r.aft_vector = construct_words_vectors(r.after)
        """
        print r.sentence
        print "BEF", r.before
        print "BET", r.between
        print "AFT", r.after
        print "passive", r.passive
        print "ReVerb:", patterns_bet_tags
        print "\n"
        """
        vectors.append(r)

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

    model = "/home/dsbatista/gigaword/word2vec/afp_apw_xing200.bin"
    print "Loading word2vec model"
    global word2vec
    word2vec = Word2Vec.load_word2vec_format(model, binary=True)
    reverb = Reverb()
    print "Processing positive sentences..."
    correct = read_sentences(sys.argv[1])
    print "Processing negative sentences..."
    incorrect = read_sentences(sys.argv[2])
    print "Constructing vector representations for positive..."
    positive = construct_vectors(correct, reverb)
    print "Constructing vector representations for negative..."
    negative = construct_vectors(incorrect, reverb)
    print "positive instances", len(positive)
    print "negative insances", len(negative)

    correct_s = sample(positive, 10)
    incorrect_s = sample(negative, 10)

    data_to_plot = list()

    # for positive versus negative patterns:
    #   find a configuration where the global scores in a boxplot where similairity is minimum
    print "Correct with Incorrect"
    scores = list()
    for p1 in correct_s:
        for p2 in incorrect_s:
            if p1 == p2:
                continue
            else:
                score = similarity_3_contexts(p1, p2)
                scores.append(score)
                #print p1.before, p1.between, p1.after, '\t', p2.before, p2.between, p2.after, '\t', score
                print p1.between, '\t', p2.between, '\t', score
    data_to_plot.append(scores)

    print "Correct with Correct"
    scores = list()
    for p1 in correct_s:
        for p2 in correct_s:
            if p1 == p2:
                continue
            else:
                score = similarity_3_contexts(p1, p2)
                scores.append(score)
                #print p1.before, p1.between, p1.after, '\t', p2.before, p2.between, p2.after, '\t', score
                print p1.between, '\t', p2.between, '\t', score
    data_to_plot.append(scores)

    print "Incorrect with Incorrect"
    for p1 in incorrect_s:
        for p2 in incorrect_s:
            if p1 == p2:
                continue
            else:
                score = similarity_3_contexts(p1, p2)
                scores.append(score)
                #print p1.between, p1.after, '\t', p2.before, p2.between, p2.after, '\t', score
                print p1.between, '\t', p2.between, '\t', score
    data_to_plot.append(scores)

    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    bp = ax.boxplot(data_to_plot)

    # Save the figure
    fig.savefig('fig1.png', bbox_inches='tight')


if __name__ == "__main__":
    main()