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

from collections import defaultdict
from nltk import PunktWordTokenizer, pos_tag
from nltk.corpus import stopwords
from Common.ReVerb import Reverb
from numpy import dot, zeros
from gensim.models import Word2Vec
from gensim import matutils


SAMPLE_SIZE = 40
THRESHOLD = 0.6
WINDOW_SIZE = 2
ALPHA = 0.2
BETA = 0.6
GAMMA = 0.2

# http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html
# select everything except stopwords, ADJ and ADV
filter_pos = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB']
stopwords = stopwords.words('english')


class Relationship(object):

    def __init__(self, _before, _between, _after, _sentence, _ent1, _ent2):
        self.before = _before
        self.between = _between
        self.after = _after
        self.bef_words = list()
        self.bet_words = list()
        self.aft_words = list()
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


def read_sentences_bunescu(data):
    # Dataset created by Bunescu et al. 2007 (http://www.cs.utexas.edu/~ml/papers/bunescu-acl07.pdf)
    # Available at: http://knowitall.cs.washington.edu/hlt-naacl08-data.txt
    relationships = defaultdict(list)
    rel_type = None
    for line in fileinput.input(data):
        if len(line) == 1:
            continue
        elif line.startswith('#'):
            rel_type = line.split("#")[1].strip()
        else:
            """
            - PoS-taggs a sentence
            - Splits the sentence into 3 contexts: BEFORE,BETWEEN,AFTER
            """
            # tag the sentence, using the default NTLK English tagger
            # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
            entities_regex = re.compile(r'<p[1-2]>[^<]+</p[1-2]>', re.U)

            # find named-entities
            matches = []
            for m in re.finditer(entities_regex, line):
                matches.append(m)

            tags_regex = re.compile('</?p[1-2]>', re.U)
            sentence_no_tags = re.sub(tags_regex, "", line)
            text_tokens = PunktWordTokenizer().tokenize(sentence_no_tags)

            entities_indices = list()
            for e in matches:
                entity = e.group()
                arg = re.sub("</?p[1-2]>", "", entity).strip()
                indices = text_tokens.index(arg)
                """
                print indices
                if len(indices) > 1:
                    print "entity found more than 1 time"
                    print line.encode("utf8")
                    sys.exit(0)
                elif len(indices) == 0:
                    print "entity not found"
                    print line.encode("utf8")
                    print arg.encode("utf8")
                    sys.exit(0)
                else:
                """
                entities_indices.append(indices)

            tagged = pos_tag(text_tokens)
            assert len(tagged) == len(text_tokens)

            try:
                before = tagged[0:entities_indices[0]]
                between = tagged[entities_indices[0]+1:entities_indices[1]]
                after = tagged[entities_indices[1]+1:]
                rel = Relationship(before, between, after, line, matches[0], matches[1])
                relationships[rel_type].append(rel)
            except IndexError:
                print "error processing file"
                sys.exit(0)

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
                vector = word2vec[t[0].strip()]
                pattern_vector += vector
            except KeyError:
                continue

    elif len(tokens) == 1:
        try:
            pattern_vector = word2vec[tokens[0][0].strip()]
        except KeyError:
            pass

    return pattern_vector


def construct_words_vectors(r, tagged_words, context):
    # remove stopwords and adjective
    words_vector = zeros(VECTOR_DIM)
    pattern = [t for t in tagged_words if t[0].lower() not in stopwords and t[1] not in filter_pos]
    if len(pattern) >= 1:
        words_vector = pattern2vector_sum(pattern)
        if context == 'before':
            r.bef_words = pattern
        elif context == 'between':
            r.bet_words = pattern
        elif context == 'after':
            r.aft_words = pattern

    return words_vector


def construct_pattern_vector(r):
    # remove stopwords and adjectives
    words_vector = zeros(VECTOR_DIM)
    pattern = [t[0] for t in r.patterns_bet_tags if t[0].lower() not in stopwords and t[1] not in filter_pos]
    if len(pattern) >= 1:
        words_vector = pattern2vector_sum(pattern)
        r.bet_words = pattern
    return words_vector


def all_indices(value, qlist):
    indices = []
    idx = -1
    while True:
        try:
            idx = qlist.index(value, idx+1)
            indices.append(idx)
        except ValueError:
            break
    return indices


def process_sentence(sentence):
    """
    - PoS-taggs a sentence
    - Splits the sentence into 3 contexts: BEFORE,BETWEEN,AFTER
    """

    # tag the sentence, using the default NTLK English tagger
    # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
    entities_regex = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)

    # find named-entities
    matches = []
    for m in re.finditer(entities_regex, sentence):
        matches.append(m)

    tags_regex = re.compile('</?[A-Z]+>', re.U)
    sentence_no_tags = re.sub(tags_regex, "", sentence)
    text_tokens = PunktWordTokenizer().tokenize(sentence_no_tags)

    print text_tokens

    for e in matches:
        entity = e.group()
        print entity
        arg = re.sub("</?[A-Z]+>", "", entity)
        indices = all_indices(arg, text_tokens)
        print indices

    tagged = pos_tag(text_tokens)

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

        t = Relationship(before_tags_cut, between_tags, after_tags_cut, sentence, ent1, ent2)
        return t


def construct_vector(rel, reverb):
    rel.patterns_bet_tags = reverb.extract_reverb_patterns_tagged_ptb(rel.between)
    if len(rel.patterns_bet_tags) > 0:
        rel.passive = reverb.detect_passive_voice(rel.patterns_bet_tags)
        rel.bet_vector = construct_pattern_vector(rel)
    else:
        rel.passive = False
        rel.bet_vector = construct_words_vectors(rel, rel.between, "between")
        # extract words before the first entity, and words after the second entity
        if len(rel.before) > 0:
            rel.bef_vector = construct_words_vectors(rel, rel.before, "before")
        if len(rel.after) > 0:
            rel.aft_vector = construct_words_vectors(rel, rel.after, "after")

    print rel.sentence
    print "BEF", rel.before
    print "BET", rel.between
    print "AFT", rel.after
    print "passive", rel.passive
    print "ReVerb:", rel.patterns_bet_tags
    print "\n"


def similarity_3_contexts(p1, p2):
        (bef, bet, aft) = (0, 0, 0)

        if p1.bef_vector is not None and p2.bef_vector is not None:
            bef = dot(matutils.unitvec(p1.bef_vector), matutils.unitvec(p2.bef_vector))

        if p1.bet_vector is not None and p2.bet_vector is not None:
            bet = dot(matutils.unitvec(p1.bet_vector), matutils.unitvec(p2.bet_vector))

        if p1.aft_vector is not None and p2.aft_vector is not None:
            aft = dot(matutils.unitvec(p1.aft_vector), matutils.unitvec(p2.aft_vector))

        return ALPHA*bef+BETA*bet+GAMMA*aft, bef, bet, aft


def main():
    #word2vec_path=/home/dsbatista/word2vec-read-only/vectors.bin
    #word2vec_path=/home/dsbatista/GoogleNews-vectors-negative300.bin
    #word2vec_path=/home/dsbatista/gigaword/word2vec/afp_apw_vectors.bin
    #word2vec_path=/home/dsbatista/gigaword/word2vec/afp_apw_xing_vectors.bin

    """
    model = "/home/dsbatista/gigaword/word2vec/afp_apw_xing200.bin"
    print "Loading word2vec model"
    global word2vec
    word2vec = Word2Vec.load_word2vec_format(model, binary=True)
    """
    global VECTOR_DIM
    #VECTOR_DIM = word2vec.layer1_size
    VECTOR_DIM = 200

    reverb = Reverb()
    print "Processing sentences..."
    sentences = read_sentences_bunescu(sys.argv[1])
    for rel_type in sentences.keys():
        print rel_type, len(sentences[rel_type])

    print "Constructing vector representations"
    for rel in sentences.keys():
        print rel in sentences[rel_type]
        construct_vector(rel, reverb)

    """
    correct_s = sample(positive, SAMPLE_SIZE)
    incorrect_s = sample(negative, SAMPLE_SIZE)
    data_to_plot = list()


    # for positive versus negative patterns:
    #   find a configuration where the global scores in a boxplot where similairity is minimum
    print "Correct with Incorrect"
    scores = list()
    above = 0
    total = 0
    for p1 in correct_s:
        for p2 in incorrect_s:
            if p1 == p2:
                continue
            else:
                total += 1
                score_total, score_bef, score_bet, score_aft = similarity_3_contexts(p1, p2)
                scores.append(score_total)
                #print p1.before, p1.between, p1.after, '\t', p2.before, p2.between, p2.after, '\t', score
                if score_total >= THRESHOLD:
                    above += 1
                    print p1.sentence
                    print p2.sentence
                    print score_total
                    print p1.bef_words, '\t', p2.bef_words, '\t', score_bef
                    print p1.bet_words, '\t', p2.bet_words, '\t', score_bet
                    print p1.aft_words, '\t', p2.aft_words, '\t', score_aft
                    print "\n"
    data_to_plot.append(scores)

    high_sim = (float(above) / float(total)) * 100
    sys.stdout.write(str(high_sim)+"% percent above threshold "+str(THRESHOLD)+"\n")

    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    bp = ax.boxplot(data_to_plot)

    # Save the figure
    fig.savefig('fig1.png', bbox_inches='tight')
    """


if __name__ == "__main__":
    main()