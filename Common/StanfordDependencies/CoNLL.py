# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import namedtuple
import re

# picks out (tag, word) from Penn Treebank-style trees
import sys

ptb_tags_and_words_re = re.compile(r'\(\s*([^\s()]+)\s+([^\s()]+)\s*\)')

# picks out (deprel, gov, govindex, dep, depindex) from Stanford
# Dependencies text (e.g., "nsubj(word-1, otherword-2)")
deps_re = re.compile(r'^\s*([^\s()]+)\(([^\s()]+)-(\d+),\s+'
                     r'([^\s()]+)-(\d+)\)\s*$',
                     re.M)

# CoNLL-X field names
FIELD_NAMES = ('index', 'form', 'lemma', 'cpos', 'pos', 'feats', 'head',
               'deprel', 'phead', 'pdeprel')

class Token(namedtuple('Token', FIELD_NAMES)):
    """CoNLL-X style dependency token. Fields include:
    - form (the word form)
    - lemma (the word's base form or lemma) -- empty for SubprocessBackend
    - pos (part of speech tag)
    - index (index of the token in the sentence)
    - head (index of the head of this token), and
    - deprel (the dependency relation between this token and its head)

    There are other fields but they typically won't be populated by
    StanfordDependencies.

    See http://ilk.uvt.nl/conll/#dataformat for a complete description."""
    def __repr__(self):
        """Represent this Token as Python code. Note that the resulting
        representation may not be a valid Python call since this skips
        fields with empty values."""
        # slightly different from the official tuple __repr__ in that
        # we skip any fields with None as their value
        items = [(field, getattr(self, field, None)) for field in FIELD_NAMES]
        fields = ['%s=%r' % (k, v) for k, v in items if v is not None]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(fields))
    def as_conll(self):
        """Represent this Token as a line as a string in CoNLL-X format."""
        def get(field):
            value = getattr(self, field)
            if value is None:
                value = '_'
            elif field == 'feats':
                value = '|'.join(value)
            return str(value)
        return '\t'.join([get(field) for field in FIELD_NAMES])
    @classmethod
    def from_conll(this_class, text):
        """Construct a Token from a line in CoNLL-X format."""
        fields = text.split('\t')
        fields[0] = int(fields[0]) # index
        fields[6] = int(fields[6]) # head index
        if fields[5] != '_': # feats
            fields[5] = tuple(fields[5].split('|'))
        fields = [value if value != '_' else None for value in fields]
        return this_class(**dict(zip(FIELD_NAMES, fields)))

class Sentence(list):
    """Sequence of Token objects."""
    def as_conll(self):
        """Represent this Sentence as a string in CoNLL-X format."""
        return '\n'.join(token.as_conll() for token in self)
    def as_asciitree(self, str_func=None):
        """Represent this Sentence as an ASCII tree string. Requires
        the asciitree package. A default token stringifier is provided
        but for custom formatting, specify a str_func which should take
        a single Token and return a string."""
        import asciitree
        from collections import defaultdict
        children = defaultdict(list)
        # since erased nodes may be missing, multiple tokens may have same
        # index (CCprocessed), etc.
        token_to_index = {}
        roots = []
        for token in self:
            children[token.head].append(token)
            token_to_index[token] = token.index
            if token.head == 0:
                roots.append(token)
        assert roots, "Couldn't find root Token(s)"

        if len(roots) > 1:
            # multiple roots so we make a fake one to be their parent
            root = Token(0, 'ROOT', 'ROOT-LEMMA', 'ROOT-CPOS', 'ROOT-POS',
                         None, None, 'ROOT-DEPREL', None, None)
            token_to_index[root] = 0
            children[0] = roots
        else:
            root = roots[0]

        def child_func(token):
            index = token_to_index[token]
            return children[index]
        if not str_func:
            def str_func(token):
                return ' %s [%s]' % (token.form, token.deprel)

        return asciitree.draw_tree(root, child_func, str_func)
    def as_dotgraph(self, digraph_kwargs=None, id_prefix=None,
                    node_formatter=None, edge_formatter=None):
        """Returns this sentence as a graphviz.Digraph. Requires the
        graphviz Python package and graphviz itself. There are several
        ways to customize. Graph level keyword arguments can be passed
        as a dictionary to digraph_kwargs. If you're viewing multiple
        Sentences in the same graph, you'll need to set a unique prefix
        string in id_prefix. Lastly, you can change the formatting of
        nodes and edges with node_formatter and edge_formatter. Both
        take a single Token as an argument (for edge_formatter, the
        Token represents the child token) and return a dictionary of
        keyword arguments which are passed to the node and edge creation
        functions in graphviz. The node_formatter will also be called
        with None as its token when adding the root."""
        digraph_kwargs = digraph_kwargs or {}
        id_prefix = id_prefix or ''

        node_formatter = node_formatter or (lambda token: {})
        edge_formatter = edge_formatter or (lambda token: {})

        import graphviz
        graph = graphviz.Digraph(**digraph_kwargs)
        # add root node
        graph.node(id_prefix + '0', 'root', **node_formatter(None))

        # add remaining nodes and edges
        already_added = set()
        for token in self:
            token_id = id_prefix + str(token.index)
            parent_id = id_prefix + str(token.head)
            if token_id not in already_added:
                graph.node(token_id, token.form, **node_formatter(token))
            graph.edge(parent_id, token_id, label=token.deprel,
                       **edge_formatter(token))
            already_added.add(token_id)
        return graph

    @classmethod
    def from_conll(this_class, stream):
        """Construct a Sentence. stream is an iterable over strings where
        each string is a line in CoNLL-X format. If there are multiple
        sentences in this stream, we only return the first one."""
        stream = iter(stream)
        sentence = this_class()
        for line in stream:
            line = line.strip()
            if line:
                sentence.append(Token.from_conll(line))
            elif sentence:
                return sentence
        return sentence
    @classmethod
    def from_stanford_dependencies(this_class, stream, tree,
                                   include_erased=False, include_punct=True):
        """Construct a Sentence. stream is an iterable over strings
        where each string is a line representing a Stanford Dependency
        as in the output of the command line Stanford Dependency tool:

            deprel(gov-index, dep-depindex)

        The corresponding Penn Treebank formatted tree must be provided
        as well."""
        stream = iter(stream)
        sentence = this_class()
        covered_indices = set()
        tags_and_words = ptb_tags_and_words_re.findall(str(tree))
        for line in stream:
            if not line.strip():
                if sentence:
                    # empty line means the sentence is over
                    break
                else:
                    continue
            matches = deps_re.findall(line)
            try:
                assert len(matches) == 1
                deprel, gov_form, head, form, index = matches[0]
                index = int(index)
                tag, word = tags_and_words[index - 1]
                assert form == word
                covered_indices.add(index)

                if not include_punct and deprel == 'punct':
                    continue
                token = Token(index, form, None, tag, tag, None, int(head),
                              deprel, None, None)
                sentence.append(token)
            except AssertionError:
                print "Error parsing dependencies"
                for l in stream:
                    print l
                print line
                sys.exit(0)

        if include_erased:
            # look through words in the tree to see if any of them
            # were erased
            for index, (tag, word) in enumerate(tags_and_words, 1):
                if index in covered_indices:
                    continue
                token = Token(index, word, None, tag, tag, None, 0,
                              'erased', None, None)
                sentence.append(token)

        sentence.sort()
        return sentence

class Corpus(list):
    """Sequence of Sentence objects."""
    def as_conll(self):
        """Represent the entire corpus as a string in CoNLL-X format."""
        return '\n'.join(sentence.as_conll() for sentence in self)
    @classmethod
    def from_conll(this_class, stream):
        """Construct a Corpus. stream is an iterable over strings where
        each string is a line in CoNLL-X format."""
        stream = iter(stream)
        corpus = this_class()
        while 1:
            # read until we get an empty sentence
            sentence = Sentence.from_conll(stream)
            if sentence:
                corpus.append(sentence)
            else:
                break
        return corpus
    @classmethod
    def from_stanford_dependencies(this_class, stream, trees,
                                   include_erased=False, include_punct=True):
        """Construct a Corpus. stream is an iterable over strings where
        each string is a line representing a Stanford Dependency as in
        the output of the command line Stanford Dependency tool:

            deprel(gov-index, dep-depindex)

        Sentences are separated by blank lines. A corresponding list of
        Penn Treebank formatted trees must be provided as well."""
        stream = iter(stream)
        corpus = this_class()
        for tree in trees:
            sentence = Sentence.from_stanford_dependencies(stream,
                                                           tree,
                                                           include_erased,
                                                           include_punct)
            corpus.append(sentence)
        return corpus
