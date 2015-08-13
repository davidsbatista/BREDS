Bootstrapping Relationship Extractors with Distributional Semantics
===================================================================


Usage:

BREDS.py parameters.cfg sentences_file seeds_file_positive seeds_file_negative similarity_threshold confidance_threshold


parameters.cfg:
    a sample configuration is provided in parameters.cfg


sentences_file:
    A text file containing the documents, one sentence per line with tags identifing the named entities, e.g.:
    ''The tech company <ORG>Soundcloud</ORG> is based in <LOC>Berlin</LOC>, capital of Germany.''


seeds_file_positive
    A file with examples of the relationships to be bootstrapped. The file must also specify the semantic type of the
    entities in the relationships. The first two lines specify that first entity in the relationship is of type ORG
    and that the second is of type LOC. e.g:

    e1:ORG
    e2:LOC

    Next a seed relationship is specified per line, e.g.:

    Nokia;Espoo
    Pfizer;New York


seeds_file_negative
    The same thing as for positive relationships, but containing seeds that do not represent the relationships to be
     bootstrapped


similarity_threshold
    The threshold similarity for clustering/extracting instances, e.g. 0.6


confidance_threshold
    The confidence threshold of an instance to used as seed, e.g. 0.8