#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import os
import codecs
import fileinput
import multiprocessing
import sys
import queue

from os import listdir
from os.path import isfile, join
from Sentence import Sentence

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

MAX_TOKENS_AWAY = 6
MIN_TOKENS_AWAY = 1
CONTEXT_WINDOW = 2

manager = multiprocessing.Manager()


def load_relationships(directory):
    entities = dict()
    only_files = [data_file for data_file in
                  listdir(directory) if isfile(join(directory, data_file))]

    for data_file in only_files:
        print("Processing", directory+data_file)
        count = 0

        # DBpedia
        if data_file.startswith("dbpedia_"):
            for line in fileinput.input(directory+data_file):
                if line.startswith("#"):
                    continue
                try:
                    e1, r, e2, t = line.split()
                    e1 = e1.replace('http://dbpedia.org/resource/', '').\
                        replace('<', '').replace('>', '').strip()
                    r = e2.replace('http://dbpedia.org/resource/', '').\
                        replace('<', '').replace('>', '')
                    e2 = r.replace('http://dbpedia.org/ontology/', '').\
                        replace('<', '').replace('>', '').strip()
                    entities[e1] = 'dummy'
                    entities[e2] = 'dummy'
                    count += 1
                    if count % 100000 == 0:
                        print(count, "processed")
                except Exception as e:
                    print(e)
                    print(line)
                    sys.exit(0)
            fileinput.close()

        # Freebase
        elif data_file.startswith("freebase_"):
            for line in fileinput.input(directory+data_file):
                if line.startswith("#"):
                    continue
                try:
                    e1, r, e2 = line.split('\t')
                    entities[e1.strip()] = 'dummy'
                    entities[e2.strip()] = 'dummy'
                    count += 1
                    if count % 100000 == 0:
                        print(count, "processed")
                except Exception as e:
                    print(e)
                    print(line)
                    sys.exit(0)
            fileinput.close()

        # YAGO
        elif data_file.startswith("yago_"):
            for line in fileinput.input(directory+data_file):
                if line.startswith("#"):
                    continue
                try:
                    e1, r, e2 = line.split('\t')
                    e1 = e1.replace('<', '').replace('>', '').strip()
                    r = r.replace('<', '').replace('>', '')
                    e2 = e2.split(" ")[0].replace('<', '').\
                        replace('>', '').strip()
                    entities[e1] = 'dummy'
                    entities[e2] = 'dummy'
                    count += 1
                    if count % 100000 == 0:
                        print(count, "processed")
                except Exception as e:
                    print(e)
                    print(line)
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
                print(count, "processed")
        return sentences


def get_sentences(sentences, entities, child_conn):
    count = 0
    selected = list()
    discarded = list()
    while True:
        try:
            sentence = sentences.get_nowait()
            discard = False
            count += 1
            s = Sentence(sentence.strip(), MAX_TOKENS_AWAY, MIN_TOKENS_AWAY,
                         CONTEXT_WINDOW)
            for r in s.relationships:
                if r.between == " , " or r.between == " ( " \
                        or r.between == " ) ":
                    discard = True
                    break
                elif r.e1 not in entities or r.e2 not in entities:
                    discard = True
                    break

            if len(s.relationships) == 0 or discard == len(s.relationships):
                discard = True

            if discard is False:
                selected.append(sentence)

            elif discard is True:
                discarded.append(sentence)

            if count % 50000 == 0:
                print(multiprocessing.current_process(), "queue size", \
                    sentences.qsize())

        except queue.Empty:
            print(multiprocessing.current_process(), "Queue is Empty")
            print(multiprocessing.current_process(), "selected", len(selected))
            print(multiprocessing.current_process(), "discarded", len(discarded))
            pid = multiprocessing.current_process().pid
            child_conn.send((pid, selected, discarded))
            break


def main():
    # load freebase entities into a shared data structure
    if os.path.isfile("entities.pkl"):
        f = open("entities.pkl", "r")
        print("Loading pre-processed entities")
        entities = pickle.load(f)
        f.close()
        print(len(entities), " entities loaded")
        print("Loading sentences")
        sentences = load_sentences(sys.argv[1])
        print(sentences.qsize(), " sentences loaded")
    else:
        entities = load_relationships(sys.argv[1])
        print(len(entities), " entities loaded")
        f = open("entities.pkl", "wb")
        pickle.dump(entities, f)
        f.close()
        print("Loading sentences")
        sentences = load_sentences(sys.argv[2])
        print(sentences.qsize(), " sentences loaded")

    print("Selecting sentences with entities in the KB")

    # launch different processes, each reads a sentence and extracts
    # relationships, if both entities in the relationship occur in the KB
    num_cpus = multiprocessing.cpu_count()
    entities_shr_dict = manager.dict(entities)
    print(len(entities_shr_dict), " entities loaded")

    pipes = [multiprocessing.Pipe(False) for _ in range(num_cpus)]
    processes = [multiprocessing.Process(
        target=get_sentences,
        args=(sentences, entities_shr_dict, pipes[i][1]))
                 for i in range(num_cpus)]

    print("Running", len(processes), " processes")
    for proc in processes:
        proc.start()

    selected_sentences = set()
    discarded_sentences = set()

    for i in range(len(pipes)):
        data = pipes[i][0].recv()
        child_pid = data[0]
        selected = data[1]
        discarded = data[2]
        print(child_pid, "selected", len(selected), "discarded", len(discarded))
        selected_sentences.update(selected)
        discarded_sentences.update(discarded)

    for proc in processes:
        proc.join()

    print("Writing sentences to disk")
    f = open("sentences_matched_output.txt", "w")
    for s in selected_sentences:
        try:
            f.write(s.encode("utf8")+'\n')
        except Exception as e:
            print(e)
            print(type(s))
            print(s)
    f.close()

    f = open("sentences_discarded_output.txt", "w")
    for s in discarded_sentences:
        try:
            f.write(s.encode("utf8")+'\n')
        except Exception as e:
            print(e)
            print(type(s))
            print(s)
    f.close()

if __name__ == "__main__":
    main()
