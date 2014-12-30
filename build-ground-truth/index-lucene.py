#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import lucene
from lucene import SimpleFSDirectory, System, File, Document, Field, StandardAnalyzer, IndexWriter, Version

def main():
    lucene.initVM()
    index_dir = SimpleFSDirectory(File("/tmp/REMOVEME.index-dir"))
    analyzer = StandardAnalyzer(Version.LUCENE_30)
    writer = IndexWriter(index_dir, analyzer, True, IndexWriter.MaxFieldLength(512))
    print >> sys.stderr, "Currently there are %d documents in the index..." % writer.numDocs()
    print >> sys.stderr, "Reading lines from sys.stdin..."
    regex = re.compile('<[A-Z]+>([^<]+)</[A-Z]+>', re.U)
    for l in sys.stdin:
        doc = Document()
        matches = []
        # before indexing documents, change spaces withing entities to "_", e.g.:
        # In <LOC>Bonn</LOC> , the head of the <ORG>German Social Democratic Party</ORG>
        # becomes:
        # In <LOC>Bonn</LOC> , the head of the <ORG>German_Social_Democratic_Party</ORG>

        for m in re.finditer(regex, l):
            matches.append(m)
            for x in range(0, len(matches)):
                new = re.sub(r'\s', "_", matches[x].group())
                l = l.replace(matches[x].group(), new)

        print l

        doc.add(Field("text", l, Field.Store.YES, Field.Index.ANALYZED))
        writer.addDocument(doc)

    print >> sys.stderr, "Indexed lines from stdin (%d documents in index)" % (writer.numDocs())
    print >> sys.stderr, "About to optimize index of %d documents..." % writer.numDocs()
    writer.optimize()
    print >> sys.stderr, "...done optimizing index of %d documents" % writer.numDocs()
    print >> sys.stderr, "Closing index of %d documents..." % writer.numDocs()
    writer.close()
    #print >> sys.stderr, "...done closing index of %d documents" % writer.numDocs()

if __name__ == "__main__":
    main()