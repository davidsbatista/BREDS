#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'dsbatista'

import fileinput
import sys
import fnmatch
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    relationship = sys.argv[1]
    directory = sys.argv[2]
    print relationship
    files = [filename for filename in os.listdir(directory) if fnmatch.fnmatch(filename, 'evaluation*'+relationship+'*.txt')]
    print sorted(files)
    similarity = list()
    threshold = list()
    f1 = list()
    print "Conf. Threshold\tSimilarity\tF1"
    for f in sorted(files):
        parts = f.split("_")
        similarity = float(parts[-2].strip())
        threshold = float(parts[-1].split(".txt")[0].strip())
        for line in fileinput.input(directory+'/'+f):
            if line.startswith("|a|"):
                a = line.strip()
            if line.startswith("|b|"):
                b = line.strip()
            if line.startswith("|c|"):
                c = line.strip()
            if line.startswith("|d|"):
                d = line.strip()
            if line.startswith("|S|"):
                S = line.strip()
            if line.startswith("Relationships not evaluated"):
                not_evaluated = line.strip()
            if line.startswith("Precision"):
                precision = line.split()[1]
            if line.startswith("Recall"):
                recall = line.split()[1]
            if line.startswith("F1"):
                f1 = line.split()[2]
        print str(threshold)+'\t'+str(similarity)+'\t'+str(f1)
        """
        fileinput.close()
        x = [row.split(' ')[0] for row in data]
        y = [row.split(' ')[1] for row in data]
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title("Plot title...")
        ax1.set_xlabel('your x label..')
        ax1.set_ylabel('your y label...')
        ax1.plot(x,y, c='r', label='the data')
        leg = ax1.legend()
        plt.show()
        """


if __name__ == "__main__":
    main()
