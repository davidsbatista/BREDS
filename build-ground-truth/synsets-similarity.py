__author__ = 'dsbatista'

from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic


def main():
    brown_ic = wordnet_ic.ic('ic-brown.dat')

    wn.synsets('founded')
    wn.synsets('leader')

    dog = wn.synset('dog.n.01')
    cat = wn.synset('cat.n.01')
    dog.jcn_similarity(cat, brown_ic)
    dog.jcn_similarity(cat, brown_ic)


if __name__ == "__main__":
    main()