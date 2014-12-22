__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import codecs
import re

from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktWordTokenizer

from gensim.corpora import TextCorpus
from gensim.models import TfidfModel


def tf_idf(sentences_file):
    stoplist = stopwords.words('english')
    f_sentences = codecs.open(sentences_file, encoding='utf-8')
    documents = list()
    print "Gathering sentences and removing stopwords"
    for line in f_sentences:
        line = re.sub('<[A-Z]+>[^<]+</[A-Z]+>', '', line)
        # remove common words and tokenize
        # TODO: remove punctuation, commas, etc.
        document = [word for word in PunktWordTokenizer().tokenize(line.lower()) if word not in stoplist]
        documents.append(document)
        #dictionary = corpora.Dictionary(line.lower().split() for line in open('mycorpus.txt'))

    """
    print "Removing tokens that appear only once"
    # remove words that appear only once
    # TODO: ver qual eh a frequencia de corte no word2vec, e fazer o mesmo
    all_tokens = sum(documents, [])
    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
    documents = [[word for word in text if word not in tokens_once] for text in documents]
    """

    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(text) for text in documents]
    print len(dictionary)
    print len(documents)
    print len(corpus)
    tf_idf_model = TfidfModel(corpus)
    return tf_idf_model

