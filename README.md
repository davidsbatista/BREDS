![example event parameter](https://github.com/davidsbatista/BREDS/actions/workflows/code_checks.yml/badge.svg?event=pull_request)
![code coverage](https://raw.githubusercontent.com/davidsbatista/BREDS/coverage-badge/coverage.svg?raw=true)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

## Bootstrapping Relationship Extraction with Distributional Semantics

BREDS is a bootstrapping system for relationship extraction relying on word vector representations (i.e., word embeddings). For more details please refer to:

- David S Batista, Bruno Martins, and Mário J Silva. , [Semi-Supervised Bootstrapping of Relationship Extractors with Distributional Semantics](http://davidsbatista.net/assets/documents/publications/breds-emnlp_15.pdf). In Empirical Methods in Natural Language Processing. ACL, 2015. (Honorable Mention for Best Short Paper)

- David S Batista, Ph.D. Thesis, [Large-Scale Semantic Relationship Extraction for Information Discovery (Chapter 5)](http://davidsbatista.net/assets/documents/publications/dsbatista-phd-thesis-2016.pdf), Instituto Superior Técnico, University of Lisbon, 2016

- [Presentation at PyData Berlin 2017](https://www.youtube.com/watch?v=Ra15lX-wojg)

  &nbsp;&nbsp;[![Presentation at PyData Berlin 2017](https://img.youtube.com/vi/Ra15lX-wojg/default.jpg)](https://www.youtube.com/watch?v=Ra15lX-wojg)


### Usage:

    python breds.py parameters sentences positive_seeds negative_seeds similarity confidence

**parameters**

A sample configuration is provided in `parameters.cfg`. The file contains values for different parameters:

    max_tokens_away=6           # maximum number of tokens between the two entities
    min_tokens_away=1           # minimum number of tokens between the two entities
    context_window_size=2       # number of tokens to the left and right

    wUpdt=0.5                   # < 0.5 trusts new examples less on each iteration
    number_iterations=4         # number of bootstrap iterations
    wUnk=0.1                    # weight given to unknown extracted relationship instances
    wNeg=2                      # weight given to extracted relationship instances
    min_pattern_support=2       # minimum number of instances in a cluster to be considered a pattern

    word2vec_path=vectors.bin   # path to a word2vecmodel in binary format

    alpha=0.2                   # weight of the BEF context in the similarity function
    beta=0.6                    # weight of the BET context in the similarity function
    gamma=0.2                   # weight of the AFT context in the similarity function




**sentences**

A sample configuration is provided in `sentences.txt`, a text file containing sentences, one per line, with tags 
identifying the named type of named-entities, e.g.:
 
    The tech company <ORG>Soundcloud</ORG> is based in <LOC>Berlin</LOC>, capital of Germany.
    <ORG>Pfizer</ORG> says it has hired <ORG>Morgan Stanley</ORG> to conduct the review.
    <ORG>Allianz</ORG>, based in <LOC>Munich</LOC>, said net income rose to EUR 1.32 billion.
    <ORG>Pfizer</ORG>, based in <LOC>New York City</LOC> , employs about 90,000 workers.

**positive_seeds**

A file with examples of the relationships to be bootstrapped. The file must also specify the semantic type of the
entities in the relationships. The first two lines specify that first entity in the relationship is of type ORG
and that the second is of type LOC. Then a seed relationship is specified per line, e.g.:

    e1:ORG
    e2:LOC

    Nokia;Espoo
    Pfizer;New York
    Google;Mountain View
    Microsoft;Redmond

**negative_seeds**

The same thing as for positive relationships, but containing seeds that do not represent the relationships to be
bootstrapped.

**similarity**

The threshold similarity real value [0,1] for clustering/extracting instances, e.g.:

    0.6

**confidance_threshold**

The confidence threshold real value [0,1] for an instance to be used as seed, e.g.:

    0.8


Demo
====

You need to specify a word2vec model in the `parameters.cfg` file, the model used in my experiments is available for 
download. It was generated from the sub collections of the English Gigaword Collection, namely the AFP, APW and XIN. 
The model is available here: 

[afp_apw_xin_embeddings.bin](http://data.davidsbatista.net/afp_apw_xin_embeddings.bin)

A sample file containing sentences where the named-entities are already tagged, which has 1 million sentences taken 
from the New York Times articles part of the English Gigaword Collection, is available here: 

[sentences.txt.bz2](http://data.davidsbatista.net/sentences.txt.bz2)

The golden standard used for evaluation is available here: 

[relationships_gold.zip](http://data.davidsbatista.net/relationships_gold.zip)


To extract the locations/headquarters of companies from `sentences.txt` based on the seeds examples given in 
`seeds_positive`, run the following command: 

    python breds.py parameters.cfg sentences.txt seeds_positive.txt seeds_negative.txt 0.7 0.7

In the first step BREDS pre-processes the `sentences.txt` file, generating word vector representations of 
relationships (i.e.: `processed_tuples.pkl`). This is done so that then you can experiment with different seed 
examples without having to repeat the process of generating word vectors representations. Just use `processed_tuples.pkl`
as the second argument to `BREDS.py` instead of `sentences.txt`.

Running the whole bootstrap process, depending on your hardware, sentences input size and number of iterations, 
can take very long time (i.e., a few hours). You can reduce the size of `sentences.txt` file, or you can also use 
a multicore version of BREDS. In the multicore version finding seed matches and clustering them is done in parallel, 
levering multicore architectures. You must specify at the end how many cores you want to use:

    python breds-parallel.py parameters.cfg sentences.txt seeds_positive.txt seeds_negative.txt 0.7 0.7 #cpus

The output should be in a `relationships.txt` file. The file contains a list of the relationships extracted, 
containing the confidence score, the sentence where the relationship was found, the patterns that extracted the 
relationship and whether the passive voice is present in the relationship, e.g.:

    instance: DynCorp       Reston  score:0.998
    sentence: Because <ORG>DynCorp</ORG> , headquartered in <LOC>Reston</LOC> , <LOC>Va.</LOC> , gets 98 percent of its revenue from government work .
    pattern_bef: Because
    pattern_bet: , headquartered in
    pattern_aft: , Va.
    passive voice: False

    instance: Handspring    Silicon Valley  score:0.893
    sentence: There will be more firms like <ORG>Handspring</ORG> , a company based in <LOC>Silicon Valley</LOC> that looks as if it is about to become a force in handheld computers , despite its lack of machinery .
    pattern_bef: firms like
    pattern_bet: , a company based in
    pattern_aft: that looks
    passive voice: False

## Development

```sh
git clone https://github.com/breds/
cd pybadges
python -m virtualenv venv
source venv/bin/activate
# Installs in edit mode and with development dependencies.
pip install -e .[dev]
nox
```

If you'd like to contribute your changes back to breds, please read the [contributor guide.](CONTRIBUTING.md)