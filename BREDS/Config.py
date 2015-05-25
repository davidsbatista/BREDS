#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import fileinput
import os
import re
import codecs
import cPickle
import sys

from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk import word_tokenize
from Common.Seed import Seed
from Common.ReVerb import Reverb
from Common.Stanford import StanfordParser
from gensim.models import Word2Vec
from gensim import corpora
from Word2VecWrapper import Word2VecWrapper
from Common.StanfordDependencies import StanfordDependencies


class Config(object):

    def __init__(self, config_file, seeds_file, negative_seeds, similarity, confidance, sentences_file):

        self.entities_regex = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)
        self.tags_regex = re.compile('</?[A-Z]+>', re.U)
        self.e_types = {'ORG': 3, 'LOC': 4, 'PER': 5}

        self.seed_tuples = set()
        self.negative_seed_tuples = set()
        self.vec_dim = 0
        self.e1_type = None
        self.e2_type = None
        self.stopwords = stopwords.words('english')
        self.lmtzr = WordNetLemmatizer()
        self.threshold_similarity = similarity
        self.instance_confidance = confidance
        self.word2vecwrapper = Word2VecWrapper()
        self.reverb = Reverb()
        self.dictionary = None

        for line in fileinput.input(config_file):
            if line.startswith("#") or len(line) == 1:
                continue

            if line.startswith("wUpdt"):
                self.wUpdt = float(line.split("=")[1])

            if line.startswith("wUnk"):
                self.wUnk = float(line.split("=")[1])

            if line.startswith("wNeg"):
                self.wNeg = float(line.split("=")[1])

            if line.startswith("number_iterations"):
                self.number_iterations = int(line.split("=")[1])

            if line.startswith("use_RlogF"):
                self.use_RlogF = bool(line.split("=")[1])

            if line.startswith("min_pattern_support"):
                self.min_pattern_support = int(line.split("=")[1])

            if line.startswith("max_tokens_away"):
                self.max_tokens_away = int(line.split("=")[1])

            if line.startswith("min_tokens_away"):
                self.min_tokens_away = int(line.split("=")[1])

            if line.startswith("context_window_size"):
                self.context_window_size = int(line.split("=")[1])

            if line.startswith("embeddings"):
                self.embeddings = line.split("=")[1].strip()

            if line.startswith("single_vector"):
                self.single_vector = line.split("=")[1].strip()

            if line.startswith("similarity"):
                self.similarity = line.split("=")[1].strip()

            if line.startswith("word2vec_path"):
                self.word2vecmodelpath = line.split("=")[1].strip()

            if line.startswith("vector"):
                self.vector = line.split("=")[1].strip()

            if line.startswith("alpha"):
                self.alpha = float(line.split("=")[1])

            if line.startswith("beta"):
                self.beta = float(line.split("=")[1])

            if line.startswith("gamma"):
                self.gamma = float(line.split("=")[1])

            if line.startswith("semantic_drift"):
                self.semantic_drift = line.split("=")[1].strip()

        self.read_seeds(seeds_file)
        self.read_negative_seeds(negative_seeds)
        fileinput.close()

        print "Configuration parameters"
        print "========================"

        print "Relationship Representation"
        print "e1 type              :", self.e1_type
        print "e2 type              :", self.e2_type
        print "context window       :", self.context_window_size
        print "max tokens away      :", self.max_tokens_away
        print "min tokens away      :", self.min_tokens_away
        print "Word2Vec Model       :", self.word2vecmodelpath

        print "\nVectors"
        print "embeddings type      :", self.embeddings
        print "alpha                :", self.alpha
        print "beta                 :", self.beta
        print "gamma                :", self.gamma

        print "\nSeeds:"
        print "positive seeds       :", len(self.seed_tuples)
        print "negative seeds       :", len(self.negative_seed_tuples)
        print "negative seeds wNeg  :", self.wNeg
        print "unknown seeds wUnk   :", self.wUnk

        print "\nParameters and Thresholds"
        print "threshold_similarity :", self.threshold_similarity
        print "instance confidence  :", self.instance_confidance
        print "min_pattern_support  :", self.min_pattern_support
        print "iterations           :", self.number_iterations
        print "iteration wUpdt      :", self.wUpdt
        print "semantic drift filter:", self.semantic_drift
        print "\n"

        if self.embeddings == 'fcm':
            # Load Stanford Parser using NLTK interface and PyStanfordDependencies to get the syntactic dependencies
            # JAVA_HOME needs to be set, calling 'java -version' should show: java version "1.8.0_45" or higher
            # PARSER and STANFORD_MODELS enviroment variables need to be set
            os.environ['STANFORD_PARSER'] = '/home/dsbatista/stanford-parser-full-2015-04-20/'
            os.environ['STANFORD_MODELS'] = '/home/dsbatista/stanford-parser-full-2015-04-20/'
            self.parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
            #self.sd = StanfordDependencies.get_instance(backend='subprocess', jar_filename='/home/dsbatista/stanford-parser-full-2015-04-20/stanford-parser.jar')

            if os.path.isfile("vocabulary_words.pkl"):
                print "Loading vocabulary from disk"
                f = open("vocabulary_words.pkl")
                self.dictionary = cPickle.load(f)
                f.close()
            else:
                # generate a dictionary of all the words
                self.generate_dictionary(sentences_file)
                f = open("vocabulary_words.pkl", "w")
                cPickle.dump(self.dictionary, f)
                f.close()

            print len(self.dictionary.token2id), "unique tokens"

        print "\n\nLoading word2vec model ...\n"
        self.word2vec = Word2Vec.load_word2vec_format(self.word2vecmodelpath, binary=True)
        self.vec_dim = self.word2vec.layer1_size

    def generate_dictionary(self, sentences_file):
        f_sentences = codecs.open(sentences_file, encoding='utf-8')
        documents = list()
        count = 0
        print "Generating vocabulary index from sentences..."
        for line in f_sentences:
            line = re.sub('<[A-Z]+>[^<]+</[A-Z]+>', '', line)
            # remove stop words and tokenize
            document = [word for word in word_tokenize(line.lower()) if word not in self.stopwords]
            #document = [word for word in word_tokenize(line.lower())]
            documents.append(document)
            count += 1
            if count % 10000 == 0:
                sys.stdout.write(".")

        f_sentences.close()
        self.dictionary = corpora.Dictionary(documents)

    def read_seeds(self, seeds_file):
        for line in fileinput.input(seeds_file):
            if line.startswith("#") or len(line) == 1:
                continue
            if line.startswith("e1"):
                self.e1_type = line.split(":")[1].strip()
            elif line.startswith("e2"):
                self.e2_type = line.split(":")[1].strip()
            else:
                e1 = line.split(";")[0].strip()
                e2 = line.split(";")[1].strip()
                seed = Seed(e1, e2)
                self.seed_tuples.add(seed)

    def read_negative_seeds(self, negative_seeds):
        for line in fileinput.input(negative_seeds):
                if line.startswith("#") or len(line) == 1:
                    continue
                if line.startswith("e1"):
                    self.e1_type = line.split(":")[1].strip()
                elif line.startswith("e2"):
                    self.e2_type = line.split(":")[1].strip()
                else:
                    e1 = line.split(";")[0].strip()
                    e2 = line.split(";")[1].strip()
                    seed = Seed(e1, e2)
                    self.negative_seed_tuples.add(seed)