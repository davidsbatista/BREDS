#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fileinput
import sys
import re
import jellyfish
import codecs

from collections import defaultdict
from query_freebase import queryFreebase
from query_freebase import parseResponse
from cache_freebase import parseCache

results = defaultdict(dict)
ground_truth = defaultdict(list)
acronyms = defaultdict(list)
total_uniq = set()
pessoa = set()

country_codes = dict()
country_cities = defaultdict(list)
city_countries = defaultdict(list)
countries = set()

ORGS_to_ignore = []

def loadDBpediaRelations(data):
    for line in fileinput.input(data):
        try:
            e1,rel,e2,p = line.split()
            e1 = e1.split('<http://dbpedia.org/resource/')[1].replace(">","")
            e2 = e2.split('<http://dbpedia.org/resource/')[1].replace(">","")
            e1 = re.sub("_"," ",e1)
            e2 = re.sub("_"," ",e2)

        except Exception, e:
            print line
            sys.exit(0)

        if "(" in e1 or "(" in e2:
            e1 = re.sub("\(.*\)","",e1)
            e2 = re.sub("\(.*\)","",e2)
            e1 = e1.decode("utf8").upper().strip()
            e2 = e2.decode("utf8").upper().strip()
            ground_truth[e1].append(e2)

        else:
            e1 = e1.decode("utf8").upper().strip()
            e2 = e2.decode("utf8").upper().strip()
            ground_truth[e1].append(e2)

    fileinput.close()


def loadTuples(threshold):
    total_tuples = 0
    f_discarded = open("discarded.txt","w")
    for line in fileinput.input(sys.argv[1]):
        if line.startswith("tuple"):
            parts = line.split('\t')
            e1 = parts[0].replace("tuple:","")
            e2 = parts[1]
            if e1 not in ORGS_to_ignore:
                score = parts[2]
                total_tuples += 1
                score = float(score)
                if not (score<threshold):
                    t = (e1.strip(),e2.strip())
                    try:
                        results[t].append(score)
                    except:
                        results[t] = []
                        results[t].append(score)

                # to keep track of total unique tuples
                # regardless of the score

                t = (e1.strip(),e2.strip())
                total_uniq.add(t)

            else:
                f_discarded.write(line+'\n')

    f_discarded.close()
    fileinput.close()
    return total_tuples


def loadAcronyms(data):
    for line in fileinput.input(data):
        parts = line.split('\t')
        acronym = parts[0].strip()
        expanded = parts[-1].strip()
        acronyms[acronym].append(expanded)
    fileinput.close()


def loadCountryCodes(data):
    for line in fileinput.input(data):
        parts = line.split('\t')
        code = parts[0]
        country = parts[4]
        country_codes[code] = country
    fileinput.close()


def loadGeoNamesCities(data):
    for line in fileinput.input(data):
        parts = line.split('\t')
        city = parts[1]
        country_code = parts[8]
        country = country_codes[country_code]
        country_cities[country.upper().strip()].append(city.upper().strip())
        city_countries[city.upper().strip()].append(country.upper().strip())
    global countries
    countries = country_cities.keys()
    fileinput.close()



def member():
    positive = 0
    negative = 0
    not_found = 0

    f_not_found = open("not_found.txt","w")
    f_negative  = open("negative.txt","w")
    f_positive  = open("positive.txt","w")

    tuples_not_found = set()

    for t in total_uniq:
        # try a direct match
        per_extracted = t[0].decode("utf8").upper().strip()
        org_truth = ground_truth.get(per_extracted)
        found = False;
        if org_truth:
            # if there is a direct look for similar organisations
            for org in org_truth:
                score = jellyfish.jaro_winkler(org.encode("utf8"),t[1].upper())
                if score>=0.8:
                    f_positive.write(t[0]+'\t'+t[1]+'\n')
                    positive += 1
                    found = True
                    break;

            if found == False:
                negative += 1
                f_negative.write(t[0]+'\t'+t[1]+'\t\t:'+';'.join(org_truth).encode("utf8")+'\n')

        else:
            tuples_not_found.add(t)
            not_found += 1

    for t in tuples_not_found:
        f_not_found.write(t[0]+'\t'+t[1]+'\n')

    return positive, negative, not_found



def headquarters():
    positive = 0
    negative = 0
    not_found = 0

    f_not_found = open("not_found.txt","w")
    f_negative  = open("negative.txt","w")
    f_positive  = open("positive.txt","w")

    tuples_not_found = set()

    for t in results:
        # first, try a direct match
        org_extracted = t[0].decode("utf8").upper().strip()
        locations_groundtruth = ground_truth.get(org_extracted)

        # if its a direct match with a ground truth organization, compare the locations
        if locations_groundtruth:
            loc_extracted = t[1].decode("utf8").upper().strip()
            found = False;
            for locations in locations_groundtruth:
                # some locations in DBpedia contain diferente references, e.g., city,state
                # e.g.,: AUBURN HILLS, MICHIGAN
                # split and compare with both

                # in case it was found and got outside the for-loop below
                # no need to check more references
                if found==True:
                    break;
                locations_parts =  locations.split(",")
                for loc in locations_parts:
                    # match locations with Jaro-Winkler, keep those >=0.8 similarity score
                    score = jellyfish.jaro_winkler(loc_extracted.encode("utf8"),loc.strip().encode("utf8"))
                    if score>=0.8:
                        f_positive.write(t[0]+'\t'+t[1]+'\n')
                        positive += 1
                        found = True
                        break;

                    # if ground-truth (from DBpedia) is a country, and extracted is a city
                    # check if the city is in that country
                    elif loc in countries:
                        if loc_extracted.encode("utf8") in country_cities[loc]:
                            f_positive.write(t[0]+'\t'+t[1]+'\t'+'\n')
                            positive += 1
                            found = True
                            break;

                    # if ground-truth (from DBpedia) is a city, and extracted location is a country
                    # check if that city is located in that country only
                    # elif

            if found == False:
                negative += 1
                f_negative.write(t[0]+'\t'+t[1]+'\t\t:'+';'.join(locations_groundtruth).encode("utf8")+'\n')

        else:
            tuples_not_found.add(t)

    # try to expand the acronyms
    names_found = set()
    for name in tuples_not_found:
        # if it is a single token with all uppercase letters
        if len(name[0].split())==1 and name[0].isupper():
            found = False
            # get all the possible expansions that match this acronym
            expansions = acronyms.get(name[0])
            if expansions:
                # check if any of these expansions is an organization in the 
                # ground_truth database and if it is, extract the locations
                for e in expansions:
                    locations_groundtruth = ground_truth.get(e.upper())
                    if locations_groundtruth:
                        for location in locations_groundtruth:
                            locations_parts =  location.split(",")
                            for loc in locations_parts:
                                # approximate similarity
                                score = jellyfish.jaro_winkler(loc.encode("utf8"),name[1].upper())
                                if score>=0.8:
                                    #f_positive.write(name[0]+' ('+e+')\t'+name[1]+'\t'+str(avg_score)+'\n')
                                    f_positive.write(name[0]+' ('+e+')\t'+name[1]+'\n')
                                    positive += 1
                                    found = True
                                    names_found.add(name)
                                    break;

                        if (found == True):
                            break;

    for n in names_found:
        tuples_not_found.remove(n)

    # for tuples not found query Freebase
    # cache of strings that were already queried to Freebase
    queried = []
    for line in fileinput.input('/home/dsbatista/gigaword/ground-truth/freebase-queried.txt'):
        queried.append(line.strip())
    fileinput.close()

    # file to save Freebase query results
    output = codecs.open('/home/dsbatista/gigaword/ground-truth/freebase-output.txt','a',"utf-8")

    # open file for append, update 'freebase-queried.txt' with new issue queries
    f_queried = open('/home/dsbatista/gigaword/ground-truth/freebase-queried.txt',"a");

    tuples_found = []

    for t in tuples_not_found:
        org = t[0].strip()
        # for now do not query acronyms to Freebase with ~=, too many false positives
        if not (len(t[0].split())==1 and name[0].isupper()):
            # first check if that query string was already issued to Freebase
            # if not, query Freebase and save the result
            if org not in queried:
                if org=="Star-Times": continue
                response = queryFreebase(org)
                queried.append(org)
                if response!='error':
                    try:
                        if response['result']:
                            print "found:\t",org
                            parseResponse(org,response,output)
                        else:
                            print "not found:\t",org
                        f_queried.write(org+'\n')
                        f_queried.flush()

                    except TypeError, e:
                        print org
                        print e
                        print response
                        f_queried.close()
                        output.close()
                        sys.exit(0)

                    except Exception, e:
                        print org
                        print e
                        print response
                        f_queried.close()
                        output.close()
                        sys.exit(0)


    f_queried.close()
    output.close()

    # look trough Freebase cached query results
    orgs_freebase = parseCache()

    for t in tuples_not_found:
        org = t[0].strip()
        location = t[1].strip()
        found = False
        matches = orgs_freebase[org]
        for match in matches:
            if found==True:
                break;
            for e in match:
                if e.startswith('location:'):
                    locations = e.split('location:')[1]
                    parts = locations.split(",")
                    if location in parts:
                        positive +=1
                        found = True
                        tuples_found.append(t)
                        break;

    for t in tuples_found:
        tuples_not_found.remove(t)

    not_found = len(tuples_not_found)
    for t in tuples_not_found:
        f_not_found.write(t[0]+'\t'+t[1]+'\n')


    """
    # there are extracted tuples with the same organisation name but different locations
    # aggregate the tuples by organisation name
    no_matches = defaultdict(list)

    for t in tuples_not_found:
        no_matches[t[0]].append(t)

    print len(no_matches.keys())

    # for direct matches that failed try approximate string matching
    for org in ground_truth.keys():
        for name in name_tuples_not_found:
            # if its not an acronym
            if not (len(name[0].split())==1 and name[0].isupper()):
                # only compare if first letter is the same to avoid
                # comparision with all
                if name[0][0].upper() == org[0].upper() and name[0][1].upper() == org[1].upper():
                    score = jellyfish.jaro_winkler(name[0],org.encode("utf8"))
                    if score>=0.8:
                        print name[0],'\t',org.encode("utf8"),score
                        print name[1],ground_truth[org]
                        print
    """



    """
    # TODO: pre-process output using 'Sorted Neighborhood Method'
    print len(name_tuples_not_found)
    sorted_ = sorted(name_tuples_not_found, key=lambda tup: (len(tup[0]),tup[0]))
    for n in sorted_:
        print n
    """
    f_not_found.close()
    f_negative.close()
    f_positive.close()

    return positive,negative,not_found


def ignoreORGs(data):
    for line in fileinput.input(data):
        ORGS_to_ignore.append(line.strip())
    fileinput.close()


def main():
    threshold = float(sys.argv[2])
    relationship = sys.argv[3]

    ORGS_to_ignore = ignoreORGs("/home/dsbatista/gigaword/ground-truth/ignore-ORGS.txt")

    total_tuples = loadTuples(threshold)

    if relationship=='member':
        # PER - ORG
        loadDBpediaRelations("/home/dsbatista/gigaword/ground-truth/occupation.txt")
        loadDBpediaRelations("/home/dsbatista/gigaword/ground-truth/party.txt")

    elif relationship=='headquarters':
        # ORG - LOC
        print "Loading DBpedia triples"
        loadDBpediaRelations("/home/dsbatista/gigaword/ground-truth/headquarters.txt")
        loadDBpediaRelations("/home/dsbatista/gigaword/ground-truth/location.txt")

        # load GeoNames data
        print "Loading GeoName data"
        loadCountryCodes("/home/dsbatista/gigaword/ground-truth/countryInfo.txt")
        loadGeoNamesCities("/home/dsbatista/gigaword/ground-truth/cities1000.txt")

        # If its ORG-LOC or PER-ORG, load companies acronyms
        print "Loading list of acronyms"
        loadAcronyms("/home/dsbatista/gigaword/ground-truth/wikipedia-acronyms.txt")

    # LOC - LOC
    elif relationship == 'part-of':
        sys.exit(0)

    else:
        print "compare-results.py tuples.txt threshold_value 'headquarters|member'"
        sys.exit(0)

    print len(ground_truth.keys()),"ground truth tuples loaded"
    total = len(ground_truth.keys())

    if relationship=='headquarters':
        positive,negative,not_found = headquarters()

    elif relationship=='member':
        positive,negative,not_found = member()


    print "\nTuples to evaluate: ", total_tuples,"\n"
    unique = float(len(total_uniq)/float(total_tuples))*100

    print "unique          : ", str(round(unique,2))+"%","("+str(len(total_uniq))+")"
    print "threshold       : ", threshold
    print "below threshold : ", str(round(float((len(total_uniq)-len(results))/float(len(total_uniq)))*100,2))+"%","("+str(len(total_uniq)-len(results))+")", "\n"
    print "for evaluation  : ", str(round(float(len(results)/float(len(total_uniq)))*100,2))+"%","("+str(len(results))+")"
    print "not found       : ", str(round(float(not_found)/len(results)*100,2))+"%","("+str(not_found)+")"
    print "positive        : ", positive
    print "negative        : ", negative
    print "accuracy        : ", float((positive)) / float((positive + negative))


if __name__ == "__main__":
    main()