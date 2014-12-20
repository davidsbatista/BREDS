#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import fileinput
import re
import StringIO

from nltk import pos_tag
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag.mapping import map_tag
from nltk.tokenize.punkt import PunktWordTokenizer


class Reverb(object):

    @staticmethod
    def extract_reverb_patterns(text):
        """
        Extract ReVerb relational patterns
        http://homes.cs.washington.edu/~afader/bib_pdf/emnlp11.pdf

        VERB - verbs (all tenses and modes)
        NOUN - nouns (common and proper)
        PRON - pronouns
        ADJ - adjectives
        ADV - adverbs
        ADP - adpositions (prepositions and postpositions)
        CONJ - conjunctions
        DET - determiners
        NUM - cardinal numbers
        PRT - particles or other function words
        X - other: foreign words, typos, abbreviations
        . - punctuation

        # extract ReVerb patterns:
        # V | V P | V W*P
        # V = verb particle? adv?
        # W = (noun | adj | adv | pron | det)
        # P = (prep | particle | inf. marker)
        """

        # split text into tokens
        text_tokens = PunktWordTokenizer().tokenize(text)

        # tag the sentence, using the default NTLK English tagger
        # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
        tags_ptb = pos_tag(text_tokens)

        # convert the tags to reduced tagset (Petrov et al. 2012)
        # http://arxiv.org/pdf/1104.2086.pdf
        tags = []
        for t in tags_ptb:
            tag = map_tag('en-ptb', 'universal', t[1])
            tags.append((t[0], tag))

        patterns = []
        patterns_tags = []
        i = 0
        limit = len(tags)-1

        while i <= limit:
            tmp = StringIO.StringIO()
            tmp_tags = []

            # a ReVerb pattern always starts with a verb
            if tags[i][1] == 'VERB':
                tmp.write(tags[i][0]+' ')
                t = (tags[i][0], tags[i][1])
                tmp_tags.append(t)
                i += 1

                # V = verb particle? adv? (also capture auxiliary verbs)
                while i <= limit and tags[i][1] in ['VERB', 'PRT', 'ADV']:
                    tmp.write(tags[i][0]+' ')
                    t = (tags[i][0], tags[i][1])
                    tmp_tags.append(t)
                    i += 1

                # W = (noun | adj | adv | pron | det)
                while i <= limit and tags[i][1] in ['NOUN', 'ADJ', 'ADV', 'PRON', 'DET']:
                    tmp.write(tags[i][0]+' ')
                    t = (tags[i][0],tags[i][1])
                    tmp_tags.append(t)
                    i += 1

                # P = (prep | particle | inf. marker)
                while i <= limit and tags[i][1] in ['ADP', 'PRT']:
                    tmp.write(tags[i][0]+' ')
                    t = (tags[i][0],tags[i][1])
                    tmp_tags.append(t)
                    i += 1
                # add the build pattern to the list collected patterns
                patterns.append(tmp.getvalue())
                patterns_tags.append(tmp_tags)
            i += 1

        return patterns, patterns_tags

    @staticmethod
    def test_reverb_patterns_extraction(sentences):
        for line in fileinput.input(sentences):
            #s = line.split('sentence:')[1].strip()
            text_tokens = word_tokenize(re.sub(r"</?e[1-2]>|\"", "", line))
            tagged = pos_tag(text_tokens)

            # convert the tags to reduced tagset (Petrov et al. 2012)
            # http://arxiv.org/pdf/1104.2086.pdf
            tags = []
            for t in tagged:
                tag = map_tag('en-ptb', 'universal', t[1])
                tags.append((t[0], tag))

            #r = Relationship(None, s, None, None, None)
            #extractRelationalWords(r)
            print tags

    @staticmethod
    def normalize_reverb_patterns(pattern):
        """
        Normalize relation phrase by removing: auxiliary verbs, adjectives, and adverbs
        TODO: if we are dealing with direction, do not remove aux verbs, can be used to
          detect passive voice
        """
        lmtzr = WordNetLemmatizer()
        aux_verbs = ['be', 'do', 'have', 'will', 'would', 'can', 'could', 'may', \
        'might', 'must', 'shall', 'should', 'will', 'would']
        norm_pattern = []

        #print rel_type
        #print "pattern  :",pattern

        # remove ADJ, ADV, and auxialiary VERB
        for i in range(0,len(pattern)):
            if pattern[i][1] == 'ADJ':
                continue
            elif pattern[i][1] == 'ADV':
                continue
            else:
                norm_pattern.append(pattern[i])
            """
            elif pattern[i][1]== 'VERB':
                verb = lmtzr.lemmatize(pattern[i][0],'v')
                if (verb in aux_verbs):
                    if (len(pattern)>=i+2):
                        if (pattern[i+1][1]=='VERB'):
                            continue
                        else:
                            norm_pattern.append(pattern[i])
                    else:
                        norm_pattern.append(pattern[i])
                else:
                    norm_pattern.append(pattern[i])
            """


        # if last word of pattern is a NOUN remove
        #while norm_pattern[-1][1]=='NOUN':
        #    del norm_pattern[-1]

        #print "norm     :",norm_pattern

        # Lemmatize VERB using WordNet
        """
        lemma_pattern = []
        for i in range(0,len(norm_pattern)):
            if norm_pattern[i][1]== 'VERB':
                norm = WordNetLemmatizer().lemmatize(norm_pattern[i][0],'v')
                lemma_pattern.append((norm,'VERB'))
            elif norm_pattern[i][0] == "'s":
                continue
            else:
                lemma_pattern.append(norm_pattern[i])
        #print "lemma    :",lemma_pattern
        """

        # Short Version: just VERBs and ending ADP, PRT, DET
        short = []
        #short.append(lemma_pattern[0])
        #if (len(lemma_pattern)>1) and lemma_pattern[1][1]=='VERB':
        #    short.append(lemma_pattern[1])

        short.append(norm_pattern[0])
        if (len(norm_pattern)>1) and norm_pattern[1][1]=='VERB':
            short.append(norm_pattern[1])
        tmp = []
        for w in reversed(norm_pattern):
            if (w[1] == 'ADP' or w[1] == 'NOUN' or w[1] == 'PRT' or w[1] == 'DET'):
                tmp.append(w)

        for w in reversed(tmp):
            short.append(w)

        if len(short)==1 and short[0][1]== 'VERB':
            verb = lmtzr.lemmatize(short[0][0],'v')
            if (verb in aux_verbs):
                return None

        else:
            return short


"""
- PoS-taggs a sentence
- Extract ReVerB patterns
- Splits the sentence into 3 contexts: BEFORE,BETWEEN,AFTER
- Fills in the attributes in the Relationship class with this information
def processSentencesSemEval(rel):
    # remove the tags and extract tokens
    text_tokens = word_tokenize(re.sub(r"</?e[1-2]>|\"", "", rel.sentence))
    regex = re.compile(r'<e[1-2]>[^<]+</e[1-2]>',re.U)

    # tag the sentence, using the default NTLK English tagger
    # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
    tagged = pos_tag(text_tokens)

    # convert the tags to reduced tagset (Petrov et al. 2012)
    # http://arxiv.org/pdf/1104.2086.pdf
    tags = []
    for t in tagged:
       tag = map_tag('en-ptb', 'universal', t[1])
       tags.append((t[0],tag))

    # extract contexts along with PoS-Tags
    matches = []

    for m in re.finditer(regex,rel.sentence):
        matches.append(m)

    for x in range(0,len(matches)-1):
        if x == 0:
            start = 0
        if x>0:
            start = matches[x-1].end()
        try:
            end = matches[x+2].start()
        except:
            end = len(rel.sentence)-1

        before = rel.sentence[ start :matches[x].start()]
        between = rel.sentence[matches[x].end():matches[x+1].start()]
        after = rel.sentence[matches[x+1].end(): end ]
        ent1 = matches[x].group()
        ent2 = matches[x+1].group()

        arg1 = re.sub("</?e[1-2]>","",ent1)
        arg2 = re.sub("</?e[1-2]>","",ent2)

        rel.arg1 = arg1
        rel.arg2 = arg2

        quote = False
        bgn_e2 = rel.sentence.index("<e2>")
        end_e2 = rel.sentence.index("</e2>")
        if (rel.sentence[bgn_e2-1])=="'":
            quote = True
        if (rel.sentence[end_e2+len("</e2>")])=="'":
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

        before_tags = []
        between_tags = []
        after_tags = []

        # to split the tagged sentence into contexts, preserving the PoS-tags
        # has to take into consideration multi-word entities
        # NOTE: this works, but probably can be done in a much cleaner way

        before_i = 0
        for i in range(0,len(tags)):
            j = i
            z = 0
            while ( (z<=len(arg1_parts)-1) and tags[j][0]==arg1_parts[z]):
                j += 1
                z += 1
            if (z==len(arg1_parts)):
                before_i = i
                break;

        for i in range(before_i,len(tags)):
            j = i
            z = 0
            while ( (z<=len(arg2_parts)-1) and tags[j][0]==arg2_parts[z]):
                j += 1
                z += 1
            if (z==len(arg2_parts)):
                after_i = i
                break;

        before_tags = tags[:before_i]

        if len(arg1_parts)>1:
            between_tags = tags[before_i+2:after_i]
            after_tags = tags[after_i+1:]

        elif len(arg2_parts)>1:
            between_tags = tags[before_i+1:after_i]
            after_tags = tags[after_i+2:]

        else:
            between_tags = tags[before_i+1:after_i]
            after_tags = tags[after_i+1:]

        # fill attributes with contextual information in Relationship class
        rel.before_tags  = before_tags
        rel.between_tags = between_tags
        rel.after_tags   = after_tags

        # extract ReVerb patterns from each context
        rel.patterns_bef, rel.patterns_bef_tags = extractReVerbPatterns(before_tags)
        rel.patterns_bet, rel.patterns_bet_tags = extractReVerbPatterns(between_tags)
        rel.patterns_aft, rel.patterns_aft_tags = extractReVerbPatterns(after_tags)
"""













