__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import codecs
import re

from gensim import corpora
from nltk.tokenize.punkt import PunktWordTokenizer
from gensim.models import TfidfModel


class VectorSpaceModel(object):

    def __init__(self, sentences_file, stopwords):
        self.dictionary = None
        self.corpus = None
        f_sentences = codecs.open(sentences_file, encoding='utf-8')
        documents = list()
        print "Gathering sentences and removing stopwords"
        for line in f_sentences:
            line = re.sub('<[A-Z]+>[^<]+</[A-Z]+>', '', line)

            # TODO: remove punctuation, commas, etc.
            # remove common words and tokenize
            document = [word for word in PunktWordTokenizer().tokenize(line.lower()) if word not in stopwords]
            documents.append(document)

            # TODO: avoid keeping all documents in memory
            #dictionary = corpora.Dictionary(line.lower().split() for line in open('mycorpus.txt'))
        f_sentences.close()

        """
        print "Removing tokens that appear only once"
        # remove words that appear only once
        # TODO: ver qual eh a frequencia de corte no word2vec, e fazer o mesmo
        all_tokens = sum(documents, [])
        tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
        documents = [[word for word in text if word not in tokens_once] for text in documents]
        """

        self.dictionary = corpora.Dictionary(documents)
        self.corpus = [self.dictionary.doc2bow(text) for text in documents]
        self.tf_idf_model = TfidfModel(self.corpus)

        print len(documents), "documents red"
        print len(self.dictionary), " unique tokens"
