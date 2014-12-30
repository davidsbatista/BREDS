# ====================================================================
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# ====================================================================

from unittest import TestCase, main
from lucene import *

class PositionIncrementTestCase(TestCase):
    """
    Unit tests ported from Java Lucene
    """

    def testSetPosition(self):

        class _analyzer(PythonAnalyzer):
            def tokenStream(_self, fieldName, reader):
                class _tokenStream(PythonTokenStream):
                    def __init__(self_):
                        super(_tokenStream, self_).__init__()
                        self_.TOKENS = ["1", "2", "3", "4", "5"]
                        self_.INCREMENTS = [1, 2, 1, 0, 1]
                        self_.i = 0
                        self_.posIncrAtt = self_.addAttribute(PositionIncrementAttribute.class_)
                        self_.termAtt = self_.addAttribute(TermAttribute.class_)
                        self_.offsetAtt = self_.addAttribute(OffsetAttribute.class_)
                    def incrementToken(self_):
                        if self_.i == len(self_.TOKENS):
                            return False
                        self_.termAtt.setTermBuffer(self_.TOKENS[self_.i])
                        self_.offsetAtt.setOffset(self_.i, self_.i)
                        self_.posIncrAtt.setPositionIncrement(self_.INCREMENTS[self_.i])
                        self_.i += 1
                        return True
                    def end(self_):
                        pass
                    def reset(self_):
                        pass
                    def close(self_):
                        pass
                return _tokenStream()

        analyzer = _analyzer()

        store = RAMDirectory()
        writer = IndexWriter(store, analyzer, True, 
                             IndexWriter.MaxFieldLength.LIMITED)
        d = Document()
        d.add(Field("field", "bogus",
                    Field.Store.YES, Field.Index.ANALYZED))
        writer.addDocument(d)
        writer.optimize()
        writer.close()

        searcher = IndexSearcher(store, True)

        pos = searcher.getIndexReader().termPositions(Term("field", "1"))
        pos.next()
        # first token should be at position 0
        self.assertEqual(0, pos.nextPosition())
    
        pos = searcher.getIndexReader().termPositions(Term("field", "2"))
        pos.next()
        # second token should be at position 2
        self.assertEqual(2, pos.nextPosition())
    
        q = PhraseQuery()
        q.add(Term("field", "1"))
        q.add(Term("field", "2"))
        hits = searcher.search(q, None, 1000).scoreDocs
        self.assertEqual(0, len(hits))

        # same as previous, just specify positions explicitely.
        q = PhraseQuery() 
        q.add(Term("field", "1"), 0)
        q.add(Term("field", "2"), 1)
        hits = searcher.search(q, None, 1000).scoreDocs
        self.assertEqual(0, len(hits))

        # specifying correct positions should find the phrase.
        q = PhraseQuery()
        q.add(Term("field", "1"), 0)
        q.add(Term("field", "2"), 2)
        hits = searcher.search(q, None, 1000).scoreDocs
        self.assertEqual(1, len(hits))

        q = PhraseQuery()
        q.add(Term("field", "2"))
        q.add(Term("field", "3"))
        hits = searcher.search(q, None, 1000).scoreDocs
        self.assertEqual(1, len(hits))

        q = PhraseQuery()
        q.add(Term("field", "3"))
        q.add(Term("field", "4"))
        hits = searcher.search(q, None, 1000).scoreDocs
        self.assertEqual(0, len(hits))

        # phrase query would find it when correct positions are specified. 
        q = PhraseQuery()
        q.add(Term("field", "3"), 0)
        q.add(Term("field", "4"), 0)
        hits = searcher.search(q, None, 1000).scoreDocs
        self.assertEqual(1, len(hits))

        # phrase query should fail for non existing searched term 
        # even if there exist another searched terms in the same searched
        # position.
        q = PhraseQuery()
        q.add(Term("field", "3"), 0)
        q.add(Term("field", "9"), 0)
        hits = searcher.search(q, None, 1000).scoreDocs
        self.assertEqual(0, len(hits))

        # multi-phrase query should succed for non existing searched term
        # because there exist another searched terms in the same searched
        # position.

        mq = MultiPhraseQuery()
        mq.add([Term("field", "3"), Term("field", "9")], 0)
        hits = searcher.search(mq, None, 1000).scoreDocs
        self.assertEqual(1, len(hits))

        q = PhraseQuery()
        q.add(Term("field", "2"))
        q.add(Term("field", "4"))
        hits = searcher.search(q, None, 1000).scoreDocs
        self.assertEqual(1, len(hits))

        q = PhraseQuery()
        q.add(Term("field", "3"))
        q.add(Term("field", "5"))
        hits = searcher.search(q, None, 1000).scoreDocs
        self.assertEqual(1, len(hits))

        q = PhraseQuery()
        q.add(Term("field", "4"))
        q.add(Term("field", "5"))
        hits = searcher.search(q, None, 1000).scoreDocs
        self.assertEqual(1, len(hits))

        q = PhraseQuery()
        q.add(Term("field", "2"))
        q.add(Term("field", "5"))
        hits = searcher.search(q, None, 1000).scoreDocs
        self.assertEqual(0, len(hits))

        # should not find "1 2" because there is a gap of 1 in the index
        qp = QueryParser(Version.LUCENE_CURRENT, "field",
                         StopWhitespaceAnalyzer(False))
        q = PhraseQuery.cast_(qp.parse("\"1 2\""))
        hits = searcher.search(q, None, 1000).scoreDocs
        self.assertEqual(0, len(hits))

        # omitted stop word cannot help because stop filter swallows the
        # increments.
        q = PhraseQuery.cast_(qp.parse("\"1 stop 2\""))
        hits = searcher.search(q, None, 1000).scoreDocs
        self.assertEqual(0, len(hits))

        # query parser alone won't help, because stop filter swallows the
        # increments.
        qp.setEnablePositionIncrements(True)
        q = PhraseQuery.cast_(qp.parse("\"1 stop 2\""))
        hits = searcher.search(q, None, 1000).scoreDocs
        self.assertEqual(0, len(hits))

        # stop filter alone won't help, because query parser swallows the
        # increments.
        qp.setEnablePositionIncrements(False)
        q = PhraseQuery.cast_(qp.parse("\"1 stop 2\""))
        hits = searcher.search(q, None, 1000).scoreDocs
        self.assertEqual(0, len(hits))
      
        # when both qp qnd stopFilter propagate increments, we should find
        # the doc.
        qp = QueryParser(Version.LUCENE_CURRENT, "field",
                         StopWhitespaceAnalyzer(True))
        qp.setEnablePositionIncrements(True)
        q = PhraseQuery.cast_(qp.parse("\"1 stop 2\""))
        hits = searcher.search(q, None, 1000).scoreDocs
        self.assertEqual(1, len(hits))

    def testPayloadsPos0(self):

        dir = RAMDirectory()
        writer = IndexWriter(dir, TestPayloadAnalyzer(), True,
                             IndexWriter.MaxFieldLength.LIMITED)

        doc = Document()
        doc.add(Field("content",
                      StringReader("a a b c d e a f g h i j a b k k")))
        writer.addDocument(doc)

        r = writer.getReader()

        tp = r.termPositions(Term("content", "a"))
        count = 0
        self.assert_(tp.next())
        # "a" occurs 4 times
        self.assertEqual(4, tp.freq())

        expected = 0
        self.assertEqual(expected, tp.nextPosition())
        self.assertEqual(1, tp.nextPosition())
        self.assertEqual(3, tp.nextPosition())
        self.assertEqual(6, tp.nextPosition())

        # only one doc has "a"
        self.assert_(not tp.next())

        searcher = IndexSearcher(r)
    
        stq1 = SpanTermQuery(Term("content", "a"))
        stq2 = SpanTermQuery(Term("content", "k"))
        sqs = [stq1, stq2]
        snq = SpanNearQuery(sqs, 30, False)

        count = 0
        sawZero = False

        pspans = snq.getSpans(searcher.getIndexReader())
        while pspans.next():
            payloads = pspans.getPayload()
            sawZero |= pspans.start() == 0

            it = payloads.iterator()
            while it.hasNext():
                count += 1
                it.next()

        self.assertEqual(5, count)
        self.assert_(sawZero)

        spans = snq.getSpans(searcher.getIndexReader())
        count = 0
        sawZero = False
        while spans.next():
            count += 1
            sawZero |= spans.start() == 0

        self.assertEqual(4, count)
        self.assert_(sawZero)
		
        sawZero = False
        psu = PayloadSpanUtil(searcher.getIndexReader())
        pls = psu.getPayloadsForQuery(snq)
        count = pls.size()
        it = pls.iterator()
        while it.hasNext():
            bytes = JArray('byte').cast_(it.next())
            s = bytes.string_
            sawZero |= s == "pos: 0"

        self.assertEqual(5, count)
        self.assert_(sawZero)
        writer.close()
        searcher.getIndexReader().close()
        dir.close()


class StopWhitespaceAnalyzer(PythonAnalyzer):

    def __init__(self, enablePositionIncrements):
        super(StopWhitespaceAnalyzer, self).__init__()

        self.enablePositionIncrements = enablePositionIncrements
        self.a = WhitespaceAnalyzer()

    def tokenStream(self, fieldName, reader):

        ts = self.a.tokenStream(fieldName, reader)
        set = HashSet()
        set.add("stop")

        return StopFilter(self.enablePositionIncrements, ts, set)


class TestPayloadAnalyzer(PythonAnalyzer):

    def tokenStream(self, fieldName, reader):

        result = LowerCaseTokenizer(reader)
        return PayloadFilter(result, fieldName)


class PayloadFilter(PythonTokenFilter):

    def __init__(self, input, fieldName):
        super(PayloadFilter, self).__init__(input)
        self.input = input

        self.fieldName = fieldName
        self.pos = 0
        self.i = 0
        self.posIncrAttr = input.addAttribute(PositionIncrementAttribute.class_)
        self.payloadAttr = input.addAttribute(PayloadAttribute.class_)
        self.termAttr = input.addAttribute(TermAttribute.class_)

    def incrementToken(self):

        if self.input.incrementToken():
            bytes = JArray('byte')("pos: %d" %(self.pos))
            self.payloadAttr.setPayload(Payload(bytes))

            if self.i % 2 == 1:
                posIncr = 1
            else:
                posIncr = 0

            self.posIncrAttr.setPositionIncrement(posIncr)
            self.pos += posIncr
            self.i += 1
            return True

        return False


if __name__ == "__main__":
    import sys, lucene
    lucene.initVM()
    if '-loop' in sys.argv:
        sys.argv.remove('-loop')
        while True:
            try:
                main()
            except:
                pass
    else:
         main()