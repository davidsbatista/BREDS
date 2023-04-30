![example event parameter](https://github.com/davidsbatista/BREDS/actions/workflows/code_checks.yml/badge.svg?event=pull_request)
![code coverage](https://raw.githubusercontent.com/davidsbatista/BREDS/coverage-badge/coverage.svg?raw=true)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Pull Requests Welcome](https://img.shields.io/badge/pull%20requests-welcome-brightgreen.svg)](https://github.com/davidsbatista/BREDS/blob/main/CONTRIBUTING.md)

## BREDS: Bootstrapping Relationship Extraction with Distributional Semantics

BREDS extracts relationships from text using a bootstrapping/semi-supervised approach by relying on seed examples and on
distributional semantics to generalize the relationship.


### Extracting headquarter locations of companies:

    python runner.py --sentences=sentences.txt --positive_seeds=seeds_positive.txt --similarity=0.6 --confidence=0.6


`sentences.txt` is a text file containing one sentence per line with named-entities tagged, e.g.: 
 
    The tech company <ORG>Soundcloud</ORG> is based in <LOC>Berlin</LOC>, capital of Germany.
    <ORG>Pfizer</ORG> says it has hired <ORG>Morgan Stanley</ORG> to conduct the review.
    <ORG>Allianz</ORG>, based in <LOC>Munich</LOC>, said net income rose to EUR 1.32 billion.
    <ORG>Pfizer</ORG>, based in <LOC>New York City</LOC> , employs about 90,000 workers.


`seeds_positive.txt` is a text file containing one seed example per line, e.g.:

    e1:ORG
    e2:LOC

    Nokia;Espoo
    Pfizer;New York
    Google;Mountain View
    Microsoft;Redmond

`--similarity=0.6` and `--confidence=0.6` are parameters controlling similarity and confidence thresholds. 


The output of the process is a file `relationships.txt`: 

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




## Reference
- [Semi-Supervised Bootstrapping of Relationship Extractors with Distributional Semantics, EMNLP'15](https://aclanthology.org/D15-1056/)
```
@inproceedings{batista-etal-2015-semi,
    title = "Semi-Supervised Bootstrapping of Relationship Extractors with Distributional Semantics",
    author = "Batista, David S.  and Martins, Bruno  and Silva, M{\'a}rio J.",
    booktitle = "Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing",
    month = sep,
    year = "2015",
    address = "Lisbon, Portugal",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D15-1056",
    doi = "10.18653/v1/D15-1056",
    pages = "499--504",
}
```
- [Large-Scale Semantic Relationship Extraction for Information Discovery (Chapter 5), David S Batista, Ph.D. Thesis](http://davidsbatista.net/assets/documents/publications/dsbatista-phd-thesis-2016.pdf)
```
@incollection{phd-dsbatista2016
  title = {Large-Scale Semantic Relationship Extraction for Information Discovery},
    author = {Batista, David S.},
  school = {Instituto Superior Técnico, Universidade de Lisboa},
  year = {2016}
}
```


## Presentation at PyData Berlin 2017

  [![Presentation at PyData Berlin 2017](https://img.youtube.com/vi/Ra15lX-wojg/hqdefault.jpg)](https://www.youtube.com/watch?v=Ra15lX-wojg)



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