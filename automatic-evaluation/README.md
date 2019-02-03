Automatic Evaluation of Relation Extraction Systems on Large-scale
===================================================================

This is an implementation of a framework for large-scale evaluation of relation extraction systems based
on an automatic annotator. For more details, please refer to:

- Mirko Bronzi, Zhaochen Guo, Filipe Mesquita, Denilson Barbosa, Paolo Merialdo, [Automatic Evaluation of Relation Extraction Systems on Large-scale](https://akbcwekex2012.files.wordpress.com/2012/05/8_paper.pdf). In Proceedings of the Joint Workshop on Automatic Knowledge Base Construction and Web-scale Knowledge Extraction, AKBC-WEKEX 2012

Unfortunately this code lacks of good principles and maintenance, and currently I lack the time
to refactor it. It contains poor code and lots of hard-coded configurations like paths.

I will nevertheless try to describe the process. The idea is to create a gold standard of relationships based in 
list of sentences where at least two named-entities are present and a Knowledge Base (KB)

Two packages are needed:

jellyfish>=0.7.1
Whoosh>=2.7.4


##### `easy_freebase_clean.py`: 

This script selects relationships from a KB, in this case [FreeBase Easy](http://freebase-easy.cs.uni-freiburg.de/browse/).



##### `select_sentences.py`: 

Selects sentences for which two entities present also occur in a relationship in the KB.


##### `index_whoosh.py`:

Indexes in [Woosh](https://pypi.org/project/Whoosh/) the relationships extracted.
