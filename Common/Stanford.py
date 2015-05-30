# -*- coding: utf-8 -*-
# Natural Language Toolkit: Interface to the Stanford Parser
#
# Copyright (C) 2001-2014 NLTK Project
# Author: Steven Xu <xxu@student.unimelb.edu.au>
#
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

from __future__ import unicode_literals

import tempfile
import os
import re
from subprocess import PIPE

from nltk import compat
from nltk.internals import find_jar, find_jar_iter, config_java, java, _java_options

from nltk.parse.api import ParserI
from nltk.tree import Tree

from Common.StanfordDependencies.CoNLL import Corpus,Sentence,Token

_stanford_url = 'http://nlp.stanford.edu/software/lex-parser.shtml'


class StanfordParser(ParserI):
    r"""
    Interface to the Stanford Parser

    >>> parser=StanfordParser(
    ...     model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"
    ... )
    >>> parser.raw_parse_sents((
    ...     "the quick brown fox jumps over the lazy dog",
    ...     "the quick grey wolf jumps over the lazy fox"
    ... ))
    [Tree('ROOT', [Tree('NP', [Tree('NP', [Tree('DT', ['the']), Tree('JJ', ['quick']), Tree('JJ', ['brown']),
    Tree('NN', ['fox'])]), Tree('NP', [Tree('NP', [Tree('NNS', ['jumps'])]), Tree('PP', [Tree('IN', ['over']),
    Tree('NP', [Tree('DT', ['the']), Tree('JJ', ['lazy']), Tree('NN', ['dog'])])])])])]), Tree('ROOT', [Tree('NP',
    [Tree('NP', [Tree('DT', ['the']), Tree('JJ', ['quick']), Tree('JJ', ['grey']), Tree('NN', ['wolf'])]), Tree('NP',
    [Tree('NP', [Tree('NNS', ['jumps'])]), Tree('PP', [Tree('IN', ['over']), Tree('NP', [Tree('DT', ['the']),
    Tree('JJ', ['lazy']), Tree('NN', ['fox'])])])])])])]

    >>> parser.parse_sents((
    ...     "I 'm a dog".split(),
    ...     "This is my friends ' cat ( the tabby )".split(),
    ... ))
    [Tree('ROOT', [Tree('S', [Tree('NP', [Tree('PRP', ['I'])]), Tree('VP', [Tree('VBP', ["'m"]),
    Tree('NP', [Tree('DT', ['a']), Tree('NN', ['dog'])])])])]), Tree('ROOT', [Tree('S', [Tree('NP',
    [Tree('DT', ['This'])]), Tree('VP', [Tree('VBZ', ['is']), Tree('NP', [Tree('NP', [Tree('NP', [Tree('PRP$', ['my']),
    Tree('NNS', ['friends']), Tree('POS', ["'"])]), Tree('NN', ['cat'])]), Tree('PRN', [Tree('-LRB-', ['-LRB-']),
    Tree('NP', [Tree('DT', ['the']), Tree('NN', ['tabby'])]), Tree('-RRB-', ['-RRB-'])])])])])])]

    >>> parser.tagged_parse_sents((
    ...     (
    ...         ("The", "DT"),
    ...         ("quick", "JJ"),
    ...         ("brown", "JJ"),
    ...         ("fox", "NN"),
    ...         ("jumped", "VBD"),
    ...         ("over", "IN"),
    ...         ("the", "DT"),
    ...         ("lazy", "JJ"),
    ...         ("dog", "NN"),
    ...         (".", "."),
    ...     ),
    ... ))
    [Tree('ROOT', [Tree('S', [Tree('NP', [Tree('DT', ['The']), Tree('JJ', ['quick']), Tree('JJ', ['brown']),
    Tree('NN', ['fox'])]), Tree('VP', [Tree('VBD', ['jumped']), Tree('PP', [Tree('IN', ['over']), Tree('NP',
    [Tree('DT', ['the']), Tree('JJ', ['lazy']), Tree('NN', ['dog'])])])]), Tree('.', ['.'])])])]
    """
    _MODEL_JAR_PATTERN = r'stanford-parser-(\d+)(\.(\d+))+-models\.jar'
    _JAR = 'stanford-parser.jar'

    def __init__(self, path_to_jar=None, path_to_models_jar=None,
                 model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz',
                 encoding='UTF-8', verbose=False, java_options='-mx30G'):

        self._stanford_jar = find_jar(
            self._JAR, path_to_jar,
            env_vars=('STANFORD_PARSER',),
            searchpath=(), url=_stanford_url,
            verbose=verbose
        )

        # find the most recent model
        self._model_jar=max(
            find_jar_iter(
                self._MODEL_JAR_PATTERN, path_to_models_jar,
                env_vars=('STANFORD_MODELS',),
                searchpath=(), url=_stanford_url,
                verbose=verbose, is_regex=True
            ),
            key=lambda model_name: re.match(self._MODEL_JAR_PATTERN, model_name)
        )

        self.model_path = model_path
        self._encoding = encoding
        self.java_options = java_options

    @staticmethod
    def _parse_trees_output(output_):
        res = []
        cur_lines = []
        for line in output_.splitlines(False):
            if line == '':
                res.append(Tree.fromstring('\n'.join(cur_lines)))
                cur_lines = []
            else:
                cur_lines.append(line)
        return res

    @staticmethod
    def _parse_deps_output(output_):
        res_trees = []
        res_deps = []
        cur_lines = []
        inside_tree = False
        inside_deps = False
        tree_processed = False
        deps_processed = False

        print "Stanford parser finished"
        print "Converting Dependencies to Tokens"

        for line in output_.splitlines(False):
            if line.startswith("(ROOT"):
                inside_tree = True

            if inside_tree is True:
                if line == '':
                    res_trees.append(Tree.fromstring('\n'.join(cur_lines)))
                    cur_lines = []
                    tree_processed = True
                    inside_tree = False
                else:
                    cur_lines.append(line)

            if tree_processed is True and inside_deps is False:
                inside_deps = True
                continue

            if inside_deps is True:
                if line == '':
                    res_deps.append(Sentence.from_stanford_dependencies(cur_lines, str(res_trees[-1])))
                    cur_lines = []
                    deps_processed = True
                    inside_deps = False
                else:
                    # For some sentences Stanford Parser outputs repeated depenencies with an ' added
                    # skip those, e.g:
                    # nsubj(led-2, That-1)
                    # nsubj(led-2', That-1)
                    if re.match('.*[0-9]+\'', line):
                        pass
                    else:
                        cur_lines.append(line)

            if inside_deps is False and deps_processed is True:
                #reset all, finishing processing one sentence
                cur_lines = []
                inside_tree = False
                inside_deps = False
                tree_processed = False
                deps_processed = False

        return res_trees, res_deps

    def parse_all(self, sentence, verbose=False):
        """
        Use StanfordParser to parse a sentence. Takes a sentence as a list of
        words; it will be automatically tagged with this StanfordParser instance's
        tagger.

        :param sentence: Input sentence to parse
        :type sentence: list(str)
        :rtype: Tree
        """
        return self.parse_sents([sentence], verbose)

    def parse_sents(self, sentences, verbose=False):
        """
        Use StanfordParser to parse multiple sentences. Takes multiple sentences as a
        list where each sentence is a list of words.
        Each sentence will be automatically tagged with this StanfordParser instance's
        tagger.
        If whitespaces exists inside a token, then the token will be treated as
        separate tokens.

        :param sentences: Input sentences to parse
        :type sentences: list(list(str))
        :rtype: list(Tree)
        """
        cmd = [
            'edu.stanford.nlp.parser.lexparser.LexicalizedParser',
            '-model', self.model_path,
            '-sentences', 'newline',
            '-outputFormat', 'penn',
            '-tokenized',
            '-escaper', 'edu.stanford.nlp.process.PTBEscapingProcessor',
        ]
        return self._parse_trees_output(self._execute(
            cmd, '\n'.join(' '.join(sentence) for sentence in sentences), verbose))

    def raw_parse(self, sentence, verbose=False):
        """
        Use StanfordParser to parse a sentence. Takes a sentence as a string;
        before parsing, it will be automatically tokenized and tagged by
        the Stanford Parser.

        :param sentence: Input sentence to parse
        :type sentence: str
        :rtype: Tree
        """
        return self.raw_parse_sents((sentence,), verbose)

    def raw_parse_sents(self, sentences, verbose=False):
        """
        Use StanfordParser to parse multiple sentences. Takes multiple sentences as a
        list of strings.
        Each sentence will be automatically tokenized and tagged by the Stanford Parser.

        :param sentences: Input sentences to parse
        :type sentences: list(str)
        :rtype: list(Tree)
        """
        cmd = [
            'edu.stanford.nlp.parser.lexparser.LexicalizedParser',
            '-model', self.model_path,
            '-nthreads', '12',
            '-sentences', 'newline',
            '-outputFormat', 'penn',
        ]
        return self._parse_trees_output(self._execute(cmd, '\n'.join(sentences), verbose))

    def raw_parse_sents_deps(self, sentences, verbose=False):
        """
        Use StanfordParser to parse multiple sentences. Takes multiple sentences as a
        list of strings.
        Each sentence will be automatically tokenized and tagged by the Stanford Parser.

        :param sentences: Input sentences to parse
        :type sentences: list(str)
        :rtype: list(Tree)
        """
        cmd = [
            'edu.stanford.nlp.parser.lexparser.LexicalizedParser',
            '-model', self.model_path,
            '-nthreads', '12',
            '-sentences', 'newline',
            '-encoding', 'UTF-8',
            '-outputFormat', 'penn',
        ]
        #'-outputFormat', 'penn,typedDependencies',
        return self._parse_trees_output(self._execute(cmd, '\n'.join(sentences), verbose))

    def convert_trees(self, ptb_trees, representation='basic',
                      include_punct=True, include_erased=False, verbose=False):
        """Convert a list of Penn Treebank formatted trees (ptb_trees)
        into Stanford Dependencies. The dependencies are represented
        as a list of sentences, where each sentence is itself a list of
        Token objects.

        Currently supported representations are 'basic', 'collapsed',
        'CCprocessed', and 'collapsedTree' which behave the same as they
        in the CoreNLP command line tools.

        Setting debug=True will cause debugging information (including
        the java command run to be printed."""
        #self._raise_on_bad_representation(representation)
        #input_file = tempfile.NamedTemporaryFile(delete=False)

        trees = list()
        for ptb_tree in ptb_trees:
            #input_file.write(str(ptb_tree) + '\n')
            trees.append(str(ptb_tree)+'\n')

        encoding = self._encoding
        cmd = [
            'edu.stanford.nlp.trees.EnglishGrammaticalStructure',
            '-nthreads', '12',
            '-keepPunct',
            '-encoding', encoding,
            '-' + representation, '-treeFile']

        # if we're including erased, we want to include punctuation
        # since otherwise we won't know what SD considers punctuation
        #if include_punct or include_erased:
        stdout = self._execute(cmd, '\n'.join(trees), verbose, convert=True)

        sentences = Corpus.from_stanford_dependencies(stdout.splitlines(),
                                                      ptb_trees,
                                                      include_erased,
                                                      include_punct)
        assert len(sentences) == len(ptb_trees), \
            "Only got %d sentences from Stanford Dependencies when " \
            "given %d trees." % (len(sentences), len(ptb_trees))
        return sentences

    def tagged_parse(self, sentence, verbose=False):
        """
        Use StanfordParser to parse a sentence. Takes a sentence as a list of
        (word, tag) tuples; the sentence must have already been tokenized and
        tagged.

        :param sentence: Input sentence to parse
        :type sentence: list(tuple(str, str))
        :rtype: Tree
        """
        return self.tagged_parse_sents([sentence], verbose)[0]

    def tagged_parse_sents(self, sentences, verbose=False):
        """
        Use StanfordParser to parse multiple sentences. Takes multiple sentences
        where each sentence is a list of (word, tag) tuples.
        The sentences must have already been tokenized and tagged.

        :param sentences: Input sentences to parse
        :type sentences: list(list(tuple(str, str)))
        :rtype: Tree
        """
        tag_separator = '/'
        cmd = [
            'edu.stanford.nlp.parser.lexparser.LexicalizedParser',
            '-model', self.model_path,
            '-sentences', 'newline',
            '-outputFormat', 'penn',
            '-tokenized',
            '-tagSeparator', tag_separator,
            '-tokenizerFactory', 'edu.stanford.nlp.process.WhitespaceTokenizer',
            '-tokenizerMethod', 'newCoreLabelTokenizerFactory',
        ]
        # We don't need to escape slashes as "splitting is done on the last instance of the character in the token"
        return self._parse_trees_output(self._execute(
            cmd, '\n'.join(' '.join(tag_separator.join(tagged) for tagged in sentence) for sentence in sentences), verbose))

    def _execute(self, cmd, input_, verbose=False, convert=False):
        encoding = self._encoding
        if convert is False:
            cmd.extend(['-encoding', encoding])

        default_options = ' '.join(_java_options)

        # Configure java.
        config_java(options=self.java_options, verbose=verbose)

        # Windows is incompatible with NamedTemporaryFile() without passing in delete=False.
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as input_file:
            # Write the actual sentences to the temporary input file
            if isinstance(input_, compat.text_type) and encoding:
                input_ = input_.encode(encoding)
            input_file.write(input_)
            input_file.flush()

            cmd.append(input_file.name)

            # Run the tagger and get the output.
            stdout, stderr = java(cmd, classpath=(self._stanford_jar, self._model_jar),
                                  stdout=PIPE, stderr=PIPE)
            #stdout = stdout.decode(encoding)

        os.unlink(input_file.name)

        # Return java configurations to their default values.
        config_java(options=default_options, verbose=False)

        return stdout


def setup_module(module):
    from nose import SkipTest

    try:
        StanfordParser(
            model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz'
        )
    except LookupError:
        raise SkipTest('doctests from nltk.parse.stanford are skipped because the stanford parser jar doesn\'t exist')
