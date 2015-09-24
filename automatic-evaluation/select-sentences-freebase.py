#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import cPickle
import os
import codecs
import fileinput
import multiprocessing
import sys
import Queue

from os import listdir
from os.path import isfile, join
from Sentence import Sentence

MAX_TOKENS_AWAY = 6
MIN_TOKENS_AWAY = 1
CONTEXT_WINDOW = 2

manager = multiprocessing.Manager()


def load_relationships(directory):
    entities = dict()
    onlyfiles = [data_file for data_file in listdir(directory) if isfile(join(directory, data_file))]
    for data_file in onlyfiles:
        print "Processing", directory+data_file
        count = 0

        if data_file.startswith("dbpedia_"):
            for line in fileinput.input(directory+data_file):
                if line.startswith("#"):
                    continue
                try:
                    e1, r, e2, t = line.split()
                    e1 = e1.replace('http://dbpedia.org/resource/', '').replace('<', '').replace('>', '').strip()
                    r = e2.replace('http://dbpedia.org/resource/', '').replace('<', '').replace('>', '')
                    e2 = r.replace('http://dbpedia.org/ontology/', '').replace('<', '').replace('>', '').strip()
                    entities[e1] = 'dummy'
                    entities[e2] = 'dummy'
                    # TODO: limpar "_" e ","
                    count += 1
                    if count % 100000 == 0:
                        print count, "processed"
                except Exception, e:
                    print e
                    print line
                    sys.exit(0)
            fileinput.close()

        elif data_file.startswith("freebase_"):
            for line in fileinput.input(directory+data_file):
                if line.startswith("#"):
                    continue
                try:
                    e1, r, e2 = line.split('\t')
                    # TODO: freebase limpar "Inc."
                    entities[e1.strip()] = 'dummy'
                    entities[e2.strip()] = 'dummy'
                    count += 1
                    if count % 100000 == 0:
                        print count, "processed"
                except Exception, e:
                    print e
                    print line
                    sys.exit(0)
            fileinput.close()

        elif data_file.startswith("yago_"):
            for line in fileinput.input(directory+data_file):
                if line.startswith("#"):
                    continue
                try:
                    e1, r, e2 = line.split('\t')
                    # TODO: freebase limpar "Inc.", "_", ",", "("
                    e1 = e1.replace('<', '').replace('>', '').strip()
                    r = r.replace('<', '').replace('>', '')
                    e2 = e2.split(" ")[0].replace('<', '').replace('>', '').strip()
                    entities[e1] = 'dummy'
                    entities[e2] = 'dummy'
                    count += 1
                    if count % 100000 == 0:
                        print count, "processed"
                except Exception, e:
                    print e
                    print line
                    sys.exit(0)
            fileinput.close()

    return entities


def load_sentences(data_file):
    sentences = manager.Queue()
    count = 0
    with codecs.open(data_file, 'rb', encoding='utf-8') as f:
        for line in f:
            sentences.put(line.strip())
            count += 1
            if count % 100000 == 0:
                print count, "processed"
        return sentences


def get_sentences(sentences, entities, results):
    count = 0
    while True:
        try:
            sentence = sentences.get_nowait()
            discard = False
            count += 1
            s = Sentence(sentence.strip(), MAX_TOKENS_AWAY, MIN_TOKENS_AWAY, CONTEXT_WINDOW)
            for r in s.relationships:
                if r.between == " , " or r.between == " ( " or r.between == " ) ":
                    discard = True
                    break
                elif r.e1 not in entities or r.e2 not in entities:
                    discard = True
                    break

            if len(s.relationships) == 0 or discard == len(s.relationships):
                discard = True

            if discard is False:
                results.append(sentence)

            if count % 50000 == 0:
                print multiprocessing.current_process(), "queue size", sentences.qsize()

        except Queue.Empty:
            break


def main():
    # load freebase entities into a shared data structure
    if os.path.isfile("entities.pkl"):
        f = open("entities.pkl", "r")
        print "Loading pre-processed entities"
        entities = cPickle.load(f)
        f.close()
        print len(entities), " entities loaded"
        print "Loading sentences"
        sentences = load_sentences(sys.argv[1])
        print sentences.qsize(), " sentences loaded"
    else:
        entities = load_relationships(sys.argv[1])
        print len(entities), " entities loaded"
        f = open("entities.pkl", "wb")
        cPickle.dump(entities, f)
        f.close()
        print "Loading sentences"
        sentences = load_sentences(sys.argv[2])
        print sentences.qsize(), " sentences loaded"

    print "Selecting sentences with entities in the KB"
    # launch different processes, each reads a sentence from AFP/APW news corpora
    # transform sentence into relationships, if both entities in relationship occur in DB sentence is selected
    num_cpus = multiprocessing.cpu_count()
    results = [manager.list() for _ in range(num_cpus)]
    entities_shr_dict = manager.dict(entities)
    print len(entities_shr_dict), " entities loaded"

    processes = [multiprocessing.Process(target=get_sentences, args=(sentences, entities_shr_dict, results[i]))
                 for i in range(num_cpus)]

    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()

    selected_sentences = set()
    for l in results:
        selected_sentences.update(l)

    print "Writing sentences to disk"
    f = open("sentences_matched_output.txt", "w")
    for s in selected_sentences:
        try:
            f.write(s.encode("utf8")+'\n')
        except Exception, e:
            print e
            print type(s)
            print s
    f.close()

if __name__ == "__main__":
    main()