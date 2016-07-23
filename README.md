Bootstrapping Relationship Extraction with Distributional Semantics
===================================================================

BREDS is a bootstrapping system for relationship extraction relying on word vector representations (i.e., word embeddings). For more details please refer to:

- David S Batista, Bruno Martins, and Mário J Silva. , [Semi-Supervised Bootstrapping of Relationship Extractors with Distributional Semantics](http://davidsbatista.github.io/publications/breds-emnlp_15.pdf). In Empirical Methods in Natural Language Processing. ACL, 2015. (Honorable Mention for Best Short Paper)

- David S Batista, Ph.D. Thesis, [Large-Scale Semantic Relationship Extraction for Information Discovery (Chapter 5)](http://davidsbatista.github.io/publications/dsbatista-phd-thesis-2016.pdf), Instituto Superior Técnico, University of Lisbon, 2016


Usage:

    BREDS.py parameters sentences positive_seeds negative_simties similarity confidence

**parameters**

A sample configuration is provided in `parameters.cfg`. The file contains values for differentes parameters:

    max_tokens_away=6           # maximum number of tokens between the two entities
    min_tokens_away=1           # maximum number of tokens between the two entities
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

A sample configuration is provided in `sentences.txt`, a text file containing sentences, one per line, with tags identifing the named type of named-entities, e.g.:
 
    The tech company <ORG>Soundcloud</ORG> is based in <LOC>Berlin</LOC>, capital of Germany.
    <ORG>Pfizer</ORG> says it has hired <ORG>Morgan Stanley</ORG> to conduct the review.
    <ORG>Allianz</ORG>, based in <LOC>Munich</LOC>, said net income rose to EUR 1.32 billion ($1.96 billion).
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


Quick Demo
=============

To run a quick demo-example, extracting the locations or headquarters of companies from `sentences.txt` issue the following command:

    BREDS.py parameters.cfg sentences.txt has-installations.txt has-installations_negative.txt 0.6 0.8

The output should be a `relationships.txt`, with a list of relationships extracted, containing the confidence score, and the sentence where the relationship was found, the patterns that extracted the relationship and wether the passive voice is present in the relationship:

    instance: DynCorp       Reston  score:0.998397435897
    sentence: Because <ORG>DynCorp</ORG> , headquartered in <LOC>Reston</LOC> , <LOC>Va.</LOC> , gets 98 percent of its revenue from government work .
    pattern_bef: Because
    pattern_bet: , headquartered in
    pattern_aft: , Va.
    passive voice: False

    instance: Handspring    Silicon Valley  score:0.998397435897
    sentence: There will be more firms like <ORG>Handspring</ORG> , a company based in <LOC>Silicon Valley</LOC> that looks as if it is about to become a force in handheld computers , despite its lack of machinery .
    pattern_bef: firms like
    pattern_bet: , a company based in
    pattern_aft: that looks
    passive voice: False

Dependencies
============

**Numpy**: http://www.numpy.org/

**NLTK**: http://www.nltk.org/

**Gensim**: https://radimrehurek.com/gensim/

**Word2Vec Model**: You also need to specify a word2vec model in the `parameters.cfg` file, the one used in my experiments is available [here](https://drive.google.com/file/d/0B0CbnDgKi0PyZHRtVS1xWlVnekE/view?usp=sharing)

Notes
=====
`BREDS-parallel.py` is a different version of the algorihtm that exploits multiple cores architectures. It launches several processees to finding instances matching the seeds and also when generating extraction patterns.
