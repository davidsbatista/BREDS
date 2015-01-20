#!/usr/bin/env python

import codecs
import fileinput
import multiprocessing
import sys
import mmap

from os import listdir
from os.path import isfile, join
from Snowball.Sentence import Sentence

__author__ = 'dsbatista'

MAX_TOKENS_AWAY = 6
MIN_TOKENS_AWAY = 1
CONTEXT_WINDOW = 2

manager = multiprocessing.Manager()


def load_freebase_entities(directory):
    entities = manager.dict()
    onlyfiles = [data_file for data_file in listdir(directory) if isfile(join(directory, data_file))]
    print onlyfiles
    for data_file in onlyfiles:
        print "Processing", directory+data_file
        count = 0
        for line in fileinput.input(directory+data_file):
            e1, r, e2 = line.split('\t')
            entities[e1] = 'dummy'
            entities[e2] = 'dummy'
            count += 1
            if count % 100000 == 0:
                print count, "processed"
        fileinput.close()
    return entities


def load_sentences(data_file):
    sentences = manager.Queue()
    with codecs.open(data_file, 'rb', encoding='utf-8') as f:
        # Size 0 will read the ENTIRE file into memory!
        m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) #File is open read-only

        # Proceed with your code here -- note the file is already in memory
        # so "readine" here will be as fast as could be
        data = m.readline()
        count = 0
        while data:
            # Do stuff
            data = m.readline()
            sentences.put(data.strip())
            count += 1
            if count % 100000 == 0:
                print count, "processed"
        return sentences


def get_sentences(sentences, freebase, results):
    count = 0
    while True:
        sentence = sentences.get_nowait()
        count += 1
        s = Sentence(sentence.strip(), None, None, MAX_TOKENS_AWAY, MIN_TOKENS_AWAY, CONTEXT_WINDOW)
        for r in s.relationships:
            if r.between == " , " or r.between == " ( " or r.between == " ) ":
                continue
            else:
                try:
                    freebase[r.ent1]
                    results.append(sentence)
                    print "MATCHED:", r.ent1, '\t', r.ent2
                except KeyError:
                    try:
                        freebase[r.ent2]
                        results.append(sentence)
                        print "MATCHED:", r.ent1, '\t', r.ent2
                    except KeyError:
                        pass

        if count % 1000 == 0:
            print multiprocessing.current_process(), "queue size", sentences.qsize()
        if sentences.empty is True:
            break


def main():
    # load freebase entities into a shared data structure
    freebase = load_freebase_entities(sys.argv[1])
    print len(freebase), " freebase entities loaded"
    print "Loading sentences"
    sentences = load_sentences(sys.argv[2])
    print sentences.qsize(), " sentences loaded"

    print "Looking for sentences with Freebase entities"
    # launch different processes, each reads a sentence from AFP/APW news corpora
    # transform sentence into relationships, if both entities in relationship occur in freebase
    # select sentence
    num_cpus = multiprocessing.cpu_count()
    results = [manager.list() for _ in range(num_cpus)]

    processes = [multiprocessing.Process(target=get_sentences, args=(sentences, freebase, results[i])) for i in range(num_cpus)]
    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()

    selected_sentences = set()
    for l in results:
        selected_sentences.update(l)

    print "Writing sentences to disk"
    f = open("sentences_matched_freebase.txt", "w")
    for s in selected_sentences:
        f.write(s.encode("utf8")+'\n')
    f.close

if __name__ == "__main__":
    main()