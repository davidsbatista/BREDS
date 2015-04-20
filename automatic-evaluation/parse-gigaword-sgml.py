#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import sgmllib
import sys


class ExtractText(sgmllib.SGMLParser):

    def __init__(self, verbose=0):
        sgmllib.SGMLParser.__init__(self, verbose)
        self.data = None
        self.doc_id = None
        self.paragraphs = list()
        self.inside_p = False

    def start_doc(self, attrs):
        self.doc_id = attrs[0][1]

    def end_doc(self):
        if len(self.paragraphs)>0:
            f = open(self.doc_id+'.txt','w')
            for p in self.paragraphs:
                f.write(p+'\n')
            f.close()
        self.data = None
        self.doc_id = None
        self.paragraphs = list()
        self.inside_p = False

    def handle_data(self, data):
        if self.data is not None and self.inside_p is True:
            self.paragraphs.append(data.strip().replace("\n", " "))

    def start_text(self, attrs):
        self.data = []

    def start_p(self, attrs):
        self.inside_p = True

    def end_p(self):
        self.inside_p = False

    def end_text(self):
        pass


def main():
    parser = ExtractText()
    f = open(sys.argv[1])
    print "Processing", sys.argv[1]
    sgml_file = f.read()
    parser.feed(sgml_file)
    parser.close()

if __name__ == "__main__":
    main()