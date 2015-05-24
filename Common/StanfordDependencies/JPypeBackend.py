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

import jpype
from .StanfordDependencies import (StanfordDependencies,
                                   JavaRuntimeVersionError)
from .CoNLL import Token, Sentence

class JPypeBackend(StanfordDependencies):
    """Faster backend than SubprocessBackend but requires you to install
    jpype ('pip install JPype1', not 'JPype'). May be less stable. There's
    no speed benefit of using convert_trees() over convert_tree() for this
    backend. In terms of output, should be identical to SubprocessBackend
    except that all string fields will be unicode. Additionally, has
    the option to run the lemmatizer (see convert_tree())."""
    def __init__(self, jar_filename=None, download_if_missing=False,
                 version=None, extra_jvm_args=None, start_jpype=True,
                 jvm_path=None):
        """extra_jvm_args can be set to a list of strings which will
        be passed to your JVM.  If start_jpype is True, we will start
        a JVM via JPype if one hasn't been started already. The user is
        responsible for stopping the JVM (jpype.shutdownJVM()) when they
        are done converting. Once the JVM has been shutdown, you'll need
        to create a new JPypeBackend in order to convert after that.
        jvm_path is the path to libjvm.so (if None, will use JPype's
        default JRE path)."""
        StanfordDependencies.__init__(self, jar_filename, download_if_missing,
                                      version)
        if start_jpype and not jpype.isJVMStarted():
            jpype.startJVM(jvm_path or jpype.getDefaultJVMPath(),
                           '-ea',
                           '-Djava.class.path=' + self.jar_filename,
                           *(extra_jvm_args or []))
        self.corenlp = jpype.JPackage('edu').stanford.nlp
        try:
            self.acceptFilter = self.corenlp.util.Filters.acceptFilter()
        except TypeError:
            # this appears to be caused by a mismatch between CoreNLP and JRE
            # versions since this method changed to return a Predicate.
            version = jpype.java.lang.System.getProperty("java.version")
            self._report_version_error(version)
        trees = self.corenlp.trees
        self.treeReader = trees.Trees.readTree
        self.grammaticalStructure = trees.EnglishGrammaticalStructure
        self.stemmer = self.corenlp.process.Morphology.stemStaticSynchronized
        puncFilterInstance = trees.PennTreebankLanguagePack(). \
            punctuationWordRejectFilter()
        try:
            self.puncFilter = puncFilterInstance.test
        except AttributeError:
            self.puncFilter = puncFilterInstance.accept
        self.lemma_cache = {}
    def convert_tree(self, ptb_tree, representation='basic',
                     include_punct=True, include_erased=False,
                     add_lemmas=False):
        """Arguments are as in StanfordDependencies.convert_trees but with
        the addition of add_lemmas. If add_lemmas=True, we will run the
        Stanford CoreNLP lemmatizer and fill in the lemma field."""
        self._raise_on_bad_representation(representation)
        tree = self.treeReader(ptb_tree)
        deps = self._get_deps(tree, include_punct, representation)

        tagged_yield = self._listify(tree.taggedYield())
        indices_to_words = dict(enumerate(tagged_yield, 1))
        tokens = Sentence()
        covered_indices = set()

        def add_token(index, form, head, deprel):
            tag = indices_to_words[index].tag()
            if add_lemmas:
                lemma = self.stem(form, tag)
            else:
                lemma = None
            token = Token(index=index, form=form, lemma=lemma, cpos=tag,
                          pos=tag, feats=None, head=head, deprel=deprel,
                          phead=None, pdeprel=None)
            tokens.append(token)

        # add token for each dependency
        for dep in deps:
            index = dep.dep().index()
            head = dep.gov().index()
            deprel = dep.reln().toString()
            form = indices_to_words[index].value()

            add_token(index, form, head, deprel)
            covered_indices.add(index)
        if include_erased:
            # see if there are any tokens that were erased
            # and add them as well
            all_indices = set(indices_to_words.keys())
            for index in all_indices - covered_indices:
                form = indices_to_words[index].value()
                if not include_punct and not self.puncFilter(form):
                    continue
                add_token(index, form, head=0, deprel='erased')
            # erased generally disrupt the ordering of the sentence
            tokens.sort()
        return tokens
    def stem(self, form, tag):
        """Returns the stem of word with specific form and part-of-speech
        tag according to the Stanford lemmatizer. Lemmas are cached."""
        key = (form, tag)
        if key not in self.lemma_cache:
            lemma = self.stemmer(*key).word()
            self.lemma_cache[key] = lemma
        return self.lemma_cache[key]

    def _get_deps(self, tree, include_punct, representation):
        """Get a list of dependencies from a Stanford Treee for a specific
        Stanford Dependencies representation."""
        if include_punct:
            egs = self.grammaticalStructure(tree, self.acceptFilter)
        else:
            egs = self.grammaticalStructure(tree)

        if representation == 'basic':
            deps = egs.typedDependencies()
        elif representation == 'collapsed':
            deps = egs.typedDependenciesCollapsed(True)
        elif representation == 'CCprocessed':
            deps = egs.typedDependenciesCCprocessed(True)
        else:
            # _raise_on_bad_representation should ensure that this
            # assertion doesn't fail
            assert representation == 'collapsedTree'
            deps = egs.typedDependenciesCollapsedTree()
        return self._listify(deps)

    @staticmethod
    def _listify(collection):
        """This is a workaround where Collections are no longer iterable
        when using JPype."""
        new_list = []
        for index in range(len(collection)):
            new_list.append(collection[index])
        return new_list

    @staticmethod
    def _report_version_error(version):
        print "Your Java version:", version
        if version.split('.')[:2] < ['1', '8']:
            print "The last CoreNLP release for Java 1.6/1.7 was 3.4.1"
            print "Try using: StanfordDependencies.get_instance(" \
                  "backend='jpype', version='3.4.1')"
            print
        raise JavaRuntimeVersionError()
