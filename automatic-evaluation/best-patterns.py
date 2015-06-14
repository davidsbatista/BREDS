#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import operator
import fileinput
import sys


def process_output(data, threshold):
    patterns = dict()
    for line in fileinput.input(data):
        if line.startswith('instance'):
            null, score = line.split("score:")

        if line.startswith('pattern_bet:'):
            bet = line.split("pattern_bet:")[1].strip()

        if line.startswith('\n') and float(score) >= threshold:
            try:
                patterns[bet] += 1
            except KeyError:
                patterns[bet] = 1

    fileinput.close()
    return patterns


def main():
    patterns = process_output(sys.argv[1], float(sys.argv[2]))

    sorted_patterns = sorted(patterns.items(), key=operator.itemgetter(1), reverse=True)

    for k in sorted_patterns:
        print k

if __name__ == "__main__":
    main()