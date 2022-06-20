from pyexpat.errors import XML_ERROR_PARTIAL_CHAR
import lxml.etree as ET
import igraph
import networkx as nx
import itertools as IT
import os
from cdlib import algorithms, viz, evaluation, NodeClustering
from datetime import datetime
import numpy as np
from io import BytesIO
import matplotlib
import time
import argparse
import sys
from networkx.readwrite import json_graph
import json

from pyparsing import col

# CLEAN UP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('testset_arg', type=int)
    parser.add_argument('thickness_arg', type=int)
    parser.add_argument('comm_alg', type=int) #10 for all
    parser.add_argument('xml_path', type=str)
    parser.add_argument('output_path', type=str)

    args = parser.parse_args()
    if len(sys.argv) > 1:
        testset = args.testset_arg
        timeslice_thickness = args.thickness_arg

        #if args.comm_alg < 10:
        community_alg = args.comm_alg
        #else:
            #community_alg = 0
    else:
        testset = 1 # 1, 2 Atzmüller; 3, 4 Chimani
        timeslice_thickness = 3
        community_alg = 0
    
    filename = args.xml_path
    print(filename)
    file = open(filename, 'rb')
    xml_file = BytesIO(file.read())
    author_lst = []
    collab_graphs = {}
    authors_yearly = {} # auch speichern in welcher Konferenz sie aktiv sind
    author_conferences = {}
    author_lst_local = []
    start = datetime.now()
    flag = 0
    start_year = 0
    end_year = 0
    for event, elem in ET.iterparse(xml_file, dtd_validation=True):
        if elem.tag == "inproceedings":
            if testset == 1:
                if "conf/lpnmr/" in elem.attrib['key']:
                    flag = 'lpnmr'
                if "conf/inap/" in elem.attrib['key']:
                    flag = 'inap'
                if "conf/ilp/" in elem.attrib['key']:
                    flag = 'ilp'
                if "conf/iclp/" in elem.attrib['key']:
                    flag = 'iclp'
                if "conf/tplp/" in elem.attrib['key']:
                    flag = 'tplp'
            if testset == 2:
                if "conf/kdd/" in elem.attrib['key']:
                    flag = 'kdd'
                if "conf/icdm/" in elem.attrib['key']:
                    flag = 'icdm'
                if "conf/icml/" in elem.attrib['key']:
                    flag = 'icml'
            if testset == 3 or testset == 4:
                if "conf/stoc/" in elem.attrib['key']:
                    flag = 'stoc'
                if "conf/focs/" in elem.attrib['key']:
                    flag = 'focs'
                if "conf/soda/" in elem.attrib['key']:
                    flag = 'soda'
                if "conf/esa/" in elem.attrib['key']:
                    flag = 'esa'
                if "conf/isaac/" in elem.attrib['key']:
                    flag = 'isaac'
                if "conf/icalp/" in elem.attrib['key']:
                    flag = 'icalp'
                if "conf/mfcs/" in elem.attrib['key']:
                    flag = 'mfcs'
                if "conf/socg/" in elem.attrib['key']:
                    flag = 'socg'
                if "conf/gd/" in elem.attrib['key']:
                    flag = 'gd'
                if "conf/stacs/" in elem.attrib['key']:
                    flag = 'stacs'
                if "conf/approx/" in elem.attrib['key']:
                    flag = 'approx'
                if "conf/random/" in elem.attrib['key']:
                    flag = 'random'
                if "conf/wg/" in elem.attrib['key']:
                    flag = 'wg'
                if "conf/alenex/" in elem.attrib['key']:
                    flag = 'alenex'
            if testset == 4:
                if "conf/sea/" in elem.attrib['key']:
                    flag = 'sea'
                if "conf/latin/" in elem.attrib['key']:
                    flag = 'latin'
                if "conf/cocoa/" in elem.attrib['key']:
                    flag = 'cocoa'
                if "conf/iwoca/" in elem.attrib['key']:
                    flag = 'iwoca'
                if "conf/waoa/" in elem.attrib['key']:
                    flag = 'waoa'
                if "conf/walcom/" in elem.attrib['key']:
                    flag = 'walcom'
                if "conf/wads/" in elem.attrib['key']:
                    flag = 'wads'
        if elem.tag == "author" and flag != 0:
                author_lst_local.append(elem.text)
                if elem.text not in author_conferences:
                    author_conferences[elem.text] = set()
                author_conferences[elem.text].add(flag)
        if elem.tag == "year" and flag != 0:
            if start_year == 0:
                start_year = int(elem.text)
            else:
                if int(elem.text) < start_year:
                    start_year = int(elem.text)
            if end_year == 0:
                end_year = int(elem.text)
            else:
                if int(elem.text) > end_year:
                    end_year = int(elem.text)
            
            for year in range(int(elem.text) - timeslice_thickness + 1, int(elem.text) + 1):
                if not year in collab_graphs:
                    collab_graphs[year] = nx.Graph()
                    authors_yearly[year] = []
                if len(author_lst_local) > 1:
                    author_lst.extend(author_lst_local)
                    for a in author_lst_local:
                        #choose proper graph by year
                        if a not in authors_yearly[year]:
                            authors_yearly[year].append(a)
                        collab_graphs[year].add_node(authors_yearly[year].index(a))
                    for combo in IT.combinations(author_lst_local, 2):
                        #same graph as above
                        if collab_graphs[year].has_edge(authors_yearly[year].index(combo[0]), authors_yearly[year].index(combo[1])):
                            collab_graphs[year][authors_yearly[year].index(combo[0])][authors_yearly[year].index(combo[1])]['weight'] = collab_graphs[year][authors_yearly[year].index(combo[0])][authors_yearly[year].index(combo[1])]['weight'] + 1
                        else:
                            collab_graphs[year].add_edge(authors_yearly[year].index(combo[0]), authors_yearly[year].index(combo[1]), weight = 1)
            author_lst_local = []
            flag = 0
        elem.clear()
    end = datetime.now()
    collab_graphs = {key: collab_graphs[key] for key in sorted(collab_graphs)}


    communities = {}
    # UNBEDINGT BEIDE REPRÄSENTATIONEN ERSTELLEN UND MIT ZAHLEN ALS INDEX VERWENDEN, SONST TOT // vllt auch nicht sonst tot lol
    comm_test_file = open(args.output_path + "comm_test_all.csv", 'a')

    for y in range(1995, 2021 - timeslice_thickness + 1):
        # else years with no collaborative papers mess up the comparison graph
        if len(collab_graphs[y]) > 0:
        # put different communities in
            communities[y] = []
            if community_alg == 0 or community_alg == 10:
                average_runtime = 0
                comm_set = set()
                for i in range(0, 10):
                    start = datetime.now()
                    commies = algorithms.louvain(collab_graphs[y].copy(), weight = 'weight', resolution=0.25*(i + 1), randomize=i).communities
                    end = datetime.now()
                    average_runtime += (end - start).total_seconds()
                    for community in commies:
                        if len(community) >= 3:
                            comm_set.add(tuple(sorted(community)))
                    # comm_set.update(algorithms.louvain(collab_graphs[y].copy(), weight = 'weight', randomize=i).communities)
                    print(y, ", 0,", i, ",", len(comm_set), file = comm_test_file)
                    communities[y].extend(commies)
                print("Average runtime: ", average_runtime/10, file = comm_test_file)
            if community_alg == 1 or community_alg == 10:
                average_runtime = 0
                #comm_set = set()
                #comm_test_file = open("../resources/comm_test_whispers.csv", 'a')
                for i in range(0,10):
                    start = datetime.now()
                    commies = algorithms.chinesewhispers(collab_graphs[y].copy(), iterations=2*(i + 1), seed=i).communities
                    end = datetime.now()
                    average_runtime += (end - start).total_seconds()
                    for community in commies:
                        if len(community) >= 3:
                            comm_set.add(tuple(sorted(community)))
                    print(y, ", 1,", i, ",", len(comm_set), file = comm_test_file)
                    communities[y].extend(commies)
                print("Average runtime: ", average_runtime/10, file = comm_test_file)
            if community_alg == 2 or community_alg == 10:
                average_runtime = 0
                #comm_set = set()
                #comm_test_file = open("../resources/comm_test_scd.csv", 'a')
                for i in range(0,10):
                    start = datetime.now()
                    commies = algorithms.scd(collab_graphs[y].copy(), iterations=3*(i+1), seed=i).communities
                    end = datetime.now()
                    average_runtime += (end - start).total_seconds()
                    for community in commies:
                        if len(community) >= 3:
                            comm_set.add(tuple(sorted(community)))
                    print(y, ", 2,", i , ",", len(comm_set), file = comm_test_file)
                    communities[y].extend(commies)
                print("Average runtime: ", average_runtime/10, file = comm_test_file)

            if community_alg == 3 or community_alg == 10:
                average_runtime = 0
                #comm_set = set()
                #comm_test_file = open("../resources/comm_test_pycombo.csv", 'a')
                for i in range(0,10):
                    start = datetime.now()
                    commies = algorithms.pycombo(collab_graphs[y].copy(), modularity_resolution=0.25*(i + 1), random_seed=i).communities
                    end = datetime.now()
                    average_runtime += (end - start).total_seconds()
                    for community in commies:
                        if len(community) >= 3:
                            comm_set.add(tuple(sorted(community)))
                    print(y, ", 3,", i, ",", len(comm_set), file = comm_test_file)
                    communities[y].extend(commies)
                print("Average runtime: ", average_runtime/10, file = comm_test_file)
            # if community_alg == 4 or community_alg == 10:
            #     comm_set = set()
            #     comm_test_file = open("../resources/comm_test_gemsec.csv", 'a')
            #     for i in range(0,2):
            #         commies = algorithms.gemsec(collab_graphs[y].copy(), seed=i).communities
            #         for community in commies:
            #             if len(community) >= 3:
            #                 comm_set.add(tuple(sorted(community)))
            #         print(y, ", ", len(comm_set), file = comm_test_file)
            #         communities[y].extend(algorithms.gemsec(collab_graphs[y].copy(), seed=i).communities) #cluster umstellen

if __name__ == "__main__":
    main()