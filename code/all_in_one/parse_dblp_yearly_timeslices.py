import lxml.etree as ET
import networkx as nx
import itertools as IT
import os
from cdlib import algorithms, evaluation, NodeClustering
from datetime import datetime, timedelta
import numpy as np
from io import BytesIO
import argparse
import sys
from networkx.readwrite import json_graph
from math import log
import json
from operator import itemgetter
from scan import SCAN_nx

from pyparsing import col

# CLEAN UP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('testset_arg', type=int)
    parser.add_argument('thickness_arg', type=int)
    parser.add_argument('number_of_sol_arg', type=int)
    parser.add_argument('target_function', type=int) # 10 für alle
    parser.add_argument('comm_alg', type=int) # 10 for all
    parser.add_argument('cutoff', type=str)
    parser.add_argument('preprocessing_type', type=int) # 0 for none, >0 for closer edge weights
    parser.add_argument('postprocessing_type', type=int) # 0 for standard, 1 for avg_degree
    parser.add_argument('xml_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('-c', '--core_remove', action='store_true') # do not use with multiple postprocessing types
    args = parser.parse_args()
    postprocessing_type_max = 4
    if len(sys.argv) > 1:
        testset = args.testset_arg
        timeslice_thickness = args.thickness_arg
        number_of_solutions = args.number_of_sol_arg
        if args.target_function < 10:
            target_fun = args.target_function
        else:
            target_fun = 0
        community_alg = args.comm_alg
        cutoff = float(args.cutoff)
        if args.postprocessing_type < 10:
            postprocessing_type = args.postprocessing_type
        else:
            postprocessing_type = 0
    else:
        testset = 1 # 1, 2 Atzmüller; 3, 4 Chimani
        timeslice_thickness = 3
        number_of_solutions = 1
        target_fun = 0
        community_alg = 0
        postprocessing_type = 0
    dblp_path = args.xml_path
    if not os.path.exists(dblp_path):
        print("Error finding dblp.xml, did you enter the right path?")
        sys.exit()
    output_path = args.output_path
    if not os.path.exists(output_path):
        print("Error finding output path, did you enter the right path?")
        sys.exit()
    file = open(dblp_path, 'rb')
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
                    for a in author_lst_local:
                        #choose proper graph by year
                        # save list of all authors for faster comparisons (index vs string)
                        if a not in author_lst:
                            author_lst.append(a)
                        a_index = author_lst.index(a)
                        if a_index not in authors_yearly[year]:
                            authors_yearly[year].append(a_index)
                        collab_graphs[year].add_node(authors_yearly[year].index(a_index))
                    for combo in IT.combinations(author_lst_local, 2):
                        #same graph as above
                        if collab_graphs[year].has_edge(authors_yearly[year].index(author_lst.index(combo[0])), authors_yearly[year].index(author_lst.index(combo[1]))):
                            collab_graphs[year][authors_yearly[year].index(author_lst.index(combo[0]))][authors_yearly[year].index(author_lst.index(combo[1]))]['weight'] = collab_graphs[year][authors_yearly[year].index(author_lst.index(combo[0]))][authors_yearly[year].index(author_lst.index(combo[1]))]['weight'] + 1
                        else:
                            collab_graphs[year].add_edge(authors_yearly[year].index(author_lst.index(combo[0])), authors_yearly[year].index(author_lst.index(combo[1])), weight = 1)
            author_lst_local = []
            flag = 0
        elem.clear()
    end = datetime.now()
    collab_graphs = {key: collab_graphs[key] for key in sorted(collab_graphs)}
    if args.preprocessing_type > 0:
        for y in collab_graphs:
            for u, v, d in collab_graphs[y].edges(data=True):
                d['weight'] = 1 + d['weight']*0.1

    print("Parsing xml and building collab graphs: ", end-start)
    start = datetime.now()
    communities = []
    for y in range(start_year, end_year + 1 - timeslice_thickness + 1):
        # else years with no collaborative papers mess up the comparison graph
        if len(collab_graphs[y]) > 0:
        # put different communities in
            communities.append([])
            if community_alg == 0 or community_alg == 10:
                for i in range(0, 10):
                    communities[y - start_year].extend(algorithms.louvain(collab_graphs[y].copy(), weight = 'weight', resolution=0.25*(i + 1), randomize=i).communities)
            if community_alg == 1 or community_alg == 10:
                for i in range(0,10):
                    communities[y - start_year].extend(algorithms.chinesewhispers(collab_graphs[y].copy(), iterations=2*(i + 1), seed=i).communities)
            if community_alg == 2 or community_alg == 10:
                for i in range(0,10):
                    communities[y - start_year].extend(algorithms.scd(collab_graphs[y].copy(), iterations=3*(i+1), seed=i).communities)
            if community_alg == 3 or community_alg == 10:
                for i in range(0,10):
                    scan_init = SCAN_nx(collab_graphs[y].copy(), epsilon = 0.25+(i*0.5), mu = 3, seed = i)
                    communities[y - start_year].extend(scan_init.execute())
            
            no_small_communities = []
            for c in communities[y - start_year]:
                if not len(c) < 3:
                    no_small_communities.append(c)
            communities[y - start_year] = no_small_communities
            communities[y - start_year].sort()
            communities[y - start_year] = list(communities[y - start_year] for communities[y - start_year],_ in IT.groupby(communities[y - start_year]))
    end = datetime.now()
    print(communities[0])
    print("Finding communities: ", end-start)
    # get reference for individual communities to store in comparison graph
    #TODO: wenn community liste von listen eine ebene tiefer knoten extrahieren
    id_to_community = []  # vllt nach Jahren aufsplitten damit funktioniert
    community_nodes = []
    #community_to_id = []
    flag = True
    for y in range(start_year, end_year + 1 - timeslice_thickness + 1):
        id_to_community.append([])
        community_nodes.append([])
        for c in communities[y - start_year]:
            nodes = set()
            for node in c:
                nodes.add(authors_yearly[y][node])
            id_to_community[y - start_year].append(c)
            community_nodes[y - start_year].append(nodes)
        #community_to_id.append(dict(zip(id_to_community[y - start_year], range(0, len(id_to_community[y-start_year])))))

    
    community_calcs = [] # add list of community, embeddedness, density
    for y in range(start_year, end_year + 1 - timeslice_thickness + 1):
        community_calcs.append([])
        for c_index in range(len(communities[y - start_year])):
            community_calcs[y - start_year].append([])
            clustering = NodeClustering([id_to_community[y - start_year][c_index]], graph=collab_graphs[y], method_name="egal")
            community_calcs[y - start_year][c_index].append(evaluation.avg_embeddedness(collab_graphs[y], clustering, summary=False)[0])
            community_calcs[y - start_year][c_index].append(evaluation.internal_edge_density(collab_graphs[y], clustering, summary=False)[0])

    while args.target_function >= 0:
        number_of_solutions = args.number_of_sol_arg
        output = open(output_path + "outputSingleSourceTarget" + str(testset) +  str(timeslice_thickness) + str(number_of_solutions) + str(target_fun) + str(community_alg) + str(cutoff) + ".txt", 'w')

        comparison_graph = nx.DiGraph()
        i = 0
        per_year_buffer = []
        comm_counter = 0
        epsilon = 0.001
        global compare_total
        compare_total = timedelta()
        calculate_total = timedelta()
        start = datetime.now()
        # fill comparison graph
        for y in range(start_year, end_year + 1 - timeslice_thickness + 1):
            if i == 0:
                comparison_graph.add_node("start")
                comparison_graph.add_node("end")
                # for c in communities[y - start_year]:
                #     comparison_graph.add_node((y, community_to_id[y - start_year][c]))
                #     comparison_graph.add_edge("start",  (y, community_to_id[y - start_year][c]), weight = (y - start_year)*cutoff) #irgendwas mit y drin wahrscheinlich?
                #     comm_counter = comm_counter + 1

                for c_index in range(len(communities[y - start_year])):
                    comparison_graph.add_node((y, c_index))
                    comparison_graph.add_edge("start",  (y, c_index), weight = (y - start_year)*cutoff) #irgendwas mit y drin wahrscheinlich?
                    comparison_graph.add_edge((y, c_index), "end", weight = (end_year + 1 - timeslice_thickness - y)*cutoff)
                    comm_counter = comm_counter + 1
            else:
                # for c in communities[y - start_year]:
                #     if (y, community_to_id[y - start_year][c]) in comparison_graph:
                #         comm_counter = comm_counter + 1
                #         continue
                #     comparison_graph.add_node((y, community_to_id[y - start_year][c]))
                #     comparison_graph.add_edge("start",  (y, community_to_id[y - start_year][c]), weight = (y - start_year)*cutoff) #irgendwas mit y drin wahrscheinlich?
                #     # if one huge difference between communities, they likely aren't the same one -> maybe flat values
                #     clustering = NodeClustering([c], graph=collab_graphs[y], method_name="egal")
                #     calculate_start = datetime.now()
                #     avg_embed = evaluation.avg_embeddedness(collab_graphs[y], clustering, summary=False)
                #     edge_den = evaluation.internal_edge_density(collab_graphs[y], clustering, summary=False)
                #     calculate_end = datetime.now()
                #     calculate_total += calculate_end - calculate_start
                #     comm_counter = comm_counter + 1
                #     for c2 in communities[previous_year]:
                #         base_weight = compare_communities(community_nodes[y - start_year][community_to_id[y - start_year][c]], community_nodes[previous_year - start_year][id_to_community[previous_year - start_year].index(c2)])
                #         if base_weight > epsilon and target_fun == 0:
                #             base_weight = 1/(base_weight * ((community_calcs[community_to_id[y - start_year][c]][0] + community_calcs[community_to_id[previous_year - start_year][c2]][0])/2) * ((community_calcs[community_to_id[y - start_year][c]][1] + community_calcs[community_to_id[previous_year - start_year][c2]][1])/2))
                #         elif base_weight > epsilon and target_fun == 1:
                #             base_weight = 1-(base_weight * ((community_calcs[community_to_id[y - start_year][c]][0] + community_calcs[community_to_id[previous_year - start_year][c2]][0])/2) * ((community_calcs[community_to_id[y - start_year][c]][1] + community_calcs[community_to_id[previous_year - start_year][c2]][1])/2))
                #         elif base_weight > epsilon and target_fun == 2:
                #             if base_weight < 0.5:
                #                 base_weight = 100000
                #             else:
                #                 base_weight = 1/(base_weight * ((community_calcs[community_to_id[y - start_year][c]][0] + community_calcs[community_to_id[previous_year - start_year][c2]][0])/2) * ((community_calcs[community_to_id[y - start_year][c]][1] + community_calcs[community_to_id[previous_year - start_year][c2]][1])/2))
                #         elif base_weight > epsilon and target_fun == 3:
                #             if base_weight < 0.5:
                #                 base_weight = 100000
                #             else:
                #                 base_weight = 1-(base_weight * ((community_calcs[community_to_id[y - start_year][c]][0] + community_calcs[community_to_id[previous_year - start_year][c2]][0])/2) * ((community_calcs[community_to_id[y - start_year][c]][1] + community_calcs[community_to_id[previous_year - start_year][c2]][1])/2))
                #         elif base_weight > epsilon and target_fun == 4:
                #             if base_weight < 0.5:
                #                 base_weight = 100000
                #             else:
                #                 base_weight = (1-base_weight)
                #         else:
                #             base_weight = 100000
                #         comparison_graph.add_edge((previous_year, community_to_id[previous_year - start_year][c2]), (y, community_to_id[y - start_year][c]), weight = base_weight)
                for c_index in range(len(communities[y - start_year])):
                    if (y, c_index) in comparison_graph:
                        comm_counter = comm_counter + 1
                        continue
                    comparison_graph.add_node((y, c_index))
                    comparison_graph.add_edge("start",  (y, c_index), weight = (y - start_year)*cutoff) #irgendwas mit y drin wahrscheinlich?
                    comparison_graph.add_edge((y, c_index), "end", weight = (end_year + 1 - timeslice_thickness - y)*cutoff)
                    # if one huge difference between communities, they likely aren't the same one -> maybe flat values
                    clustering = NodeClustering([id_to_community[y - start_year][c_index]], graph=collab_graphs[y], method_name="egal")
                    calculate_start = datetime.now()
                    avg_embed = evaluation.avg_embeddedness(collab_graphs[y], clustering, summary=False)
                    edge_den = evaluation.internal_edge_density(collab_graphs[y], clustering, summary=False)
                    calculate_end = datetime.now()
                    calculate_total += calculate_end - calculate_start
                    comm_counter = comm_counter + 1
                    for c2_index in range(len(communities[previous_year - start_year])):
                        base_weight = compare_communities([*itemgetter(*id_to_community[y - start_year][c_index])(authors_yearly[y])], [*itemgetter(*id_to_community[previous_year - start_year][c2_index])(authors_yearly[previous_year])])
                        if base_weight > epsilon and target_fun == 0:
                            if base_weight < 0.5:
                                base_weight = 100000
                            else:
                                base_weight = 1/((log(len(id_to_community[y - start_year][c_index])) + log((len(id_to_community[previous_year - start_year][c2_index]))))/2*20*(base_weight * ((community_calcs[y - start_year][c_index][0] + community_calcs[previous_year - start_year][c2_index][0])/2) * ((community_calcs[y - start_year][c_index][1] + community_calcs[previous_year - start_year][c2_index][1])/2)))
                        elif base_weight > epsilon and target_fun == 1:
                            base_weight = 1-(base_weight *((community_calcs[y - start_year][c_index][0] + community_calcs[previous_year - start_year][c2_index][0])/2) * ((community_calcs[y - start_year][c_index][1] + community_calcs[previous_year - start_year][c2_index][1])/2))
                        elif base_weight > epsilon and target_fun == 2:
                            if base_weight < 0.5:
                                base_weight = 100000
                            else:
                                base_weight = 1/(((log(len(id_to_community[y - start_year][c_index])) + log((len(id_to_community[previous_year - start_year][c2_index]))))/2)**2*base_weight) 
                        elif base_weight > epsilon and target_fun == 3:
                            if base_weight < 0.5:
                                base_weight = 100000
                            else:
                                base_weight = 1-(base_weight * ((community_calcs[y - start_year][c_index][0] + community_calcs[previous_year - start_year][c2_index][0])/2) * ((community_calcs[y - start_year][c_index][1] + community_calcs[previous_year - start_year][c2_index][1])/2))
                        elif base_weight > epsilon and target_fun == 4:
                            if base_weight < 0.5:
                                base_weight = 100000
                            else:
                                base_weight = (1-base_weight)
                        elif base_weight > epsilon and target_fun == 5:
                            if base_weight < 0.5:
                                base_weight = 100000
                            else:
                                base_weight = 1/(((log(len(id_to_community[y - start_year][c_index])) + log((len(id_to_community[previous_year - start_year][c2_index]))))/2)**2*base_weight*2) 
                        else:
                            base_weight = 100000
                        comparison_graph.add_edge((previous_year, c2_index), (y, c_index), weight = base_weight)
            previous_year = y
            i = i + 1
        end = datetime.now()

        print("Building comparison graph for target function ", target_fun, ": ", end-start)
        print(comparison_graph)
        print("Time spent comparing communities: ", compare_total)
        print("Time spent calculating useless stuff: ", calculate_total)
        num_communities = 0
        for y in range(len(communities)):
            num_communities = num_communities + len(communities[y])
        print("Total number of communities: ", num_communities, file = output)

        output_csv = open(output_path + "outputSingleSourceTarget" + str(testset) +  str(timeslice_thickness) + str(number_of_solutions) + str(target_fun) + str(community_alg) + str(cutoff) + ".csv", 'w')
        while number_of_solutions > 0:
            # find shortest path in comparison graph to find most similar communities in different years
            postprocessing_type_updated = postprocessing_type

            sizes = []
            shareds = []
            edge_dens = []
            embeds = []
            average_size = 0
            size_sd = 0
            average_edge_den = 0
            edge_den_sd = 0
            average_embed = 0
            embed_sd = 0
            counter = 0


            if not nx.has_path(comparison_graph, "start", "end"):
                break
            start = datetime.now()
            path = nx.astar_path(comparison_graph, "start", "end")
            end = datetime.now()
            print(target_fun, args.number_of_sol_arg-number_of_solutions, " Finding shortest path: ", end-start)
            start = datetime.now()
            for idx, n in enumerate(path):
                print(n, file = output)
                conferences_in_comm = set()
                if counter > 0:
                    print([*itemgetter(*itemgetter(*id_to_community[n[0] - start_year][n[1]])(authors_yearly[n[0]]))(author_lst)], file = output)
                    calc_clustering = NodeClustering([id_to_community[n[0] - start_year][n[1]]], graph=collab_graphs[n[0]], method_name="egal")
                    sizes.append(len(calc_clustering.communities[0]))
                    edge_den_old = average_edge_den
                    edge_dens.append(evaluation.internal_edge_density(calc_clustering.graph, calc_clustering, summary=False)[0])
                    embeds.append(evaluation.avg_embeddedness(calc_clustering.graph, calc_clustering, summary=False)[0])
                    for author in id_to_community[n[0] - start_year][n[1]]:
                            conferences_in_comm = conferences_in_comm.union(author_conferences[author_lst[authors_yearly[n[0]][author]]])
                    print(conferences_in_comm, file = output)
                    if counter >= len(path) - 2:
                        break
                    else:
                        next_item = path[idx + 1]
                        shareds.append(compare_communities([*itemgetter(*id_to_community[n[0] - start_year][n[1]])(authors_yearly[n[0]])], [*itemgetter(*id_to_community[next_item[0] - start_year][next_item[1]])(authors_yearly[next_item[0]])]))
                        print(comparison_graph.get_edge_data(n, path[idx + 1]), file = output)
                counter = counter + 1
                if counter == len(path) - 1:
                    break
            average_size = np.mean(sizes)
            size_sd = np.std(sizes)
            average_edge_den = np.mean(edge_dens)
            edge_den_sd = np.std(edge_dens)
            average_embed = np.mean(embeds)
            embed_sd = np.std(embeds)
            if len(path) > 3:
                average_shared = np.mean(shareds) # only count year transitions
                shared_sd = np.std(shareds)
            print(args.number_of_sol_arg - number_of_solutions, ", ", len(path) - 2, ", ", average_size, ", ", size_sd, ", ", average_shared, ", ", shared_sd, ", ", average_edge_den, ", ", edge_den_sd, ", ", average_embed, ", ", embed_sd, file = output_csv)
            # create output graph
            output_graph = nx.Graph()
            color_map = []
            core_community = set()
            output_graph_updated = nx.Graph()
            counter = 0
            for idx, n in enumerate(path):
                if counter > 0:
                    comm = id_to_community[n[0] - start_year][n[1]]
                    add_nodes = set(comm)
                    add_size = len(add_nodes)
                    core_community = set(comm) # add nodes from community chain to "core community"
                    comms_only_subgraph = nx.subgraph(collab_graphs[n[0]], core_community)
                    for node in comms_only_subgraph.nodes():
                        output_graph_updated.add_node(author_lst[authors_yearly[n[0]][node]])
                    for edge in comms_only_subgraph.edges():
                        if output_graph_updated.has_edge(author_lst[authors_yearly[n[0]][edge[0]]], author_lst[authors_yearly[n[0]][edge[1]]]):
                            output_graph_updated[author_lst[authors_yearly[n[0]][edge[0]]]][author_lst[authors_yearly[n[0]][edge[1]]]]['weight'] = output_graph_updated[author_lst[authors_yearly[n[0]][edge[0]]]][author_lst[authors_yearly[n[0]][edge[1]]]]['weight'] + comms_only_subgraph[edge[0]][edge[1]]['weight'] # /timeslice_thickness
                        else:
                            output_graph_updated.add_edge(author_lst[authors_yearly[n[0]][edge[0]]], author_lst[authors_yearly[n[0]][edge[1]]], weight = comms_only_subgraph[edge[0]][edge[1]]['weight']) # /timeslice_thickness)
                    for node in comm:
                        color_map.append(author_lst[authors_yearly[n[0]][node]])
                        add_nodes = add_nodes.union(set(collab_graphs[n[0]].neighbors(node)))
                    subgraph = nx.subgraph(collab_graphs[n[0]], add_nodes)
                    for node in subgraph.nodes():
                        output_graph.add_node(author_lst[authors_yearly[n[0]][node]])
                    for edge in subgraph.edges():
                        if output_graph.has_edge(author_lst[authors_yearly[n[0]][edge[0]]], author_lst[authors_yearly[n[0]][edge[1]]]):
                            output_graph[author_lst[authors_yearly[n[0]][edge[0]]]][author_lst[authors_yearly[n[0]][edge[1]]]]['weight'] = output_graph[author_lst[authors_yearly[n[0]][edge[0]]]][author_lst[authors_yearly[n[0]][edge[1]]]]['weight'] + subgraph[edge[0]][edge[1]]['weight'] # /timeslice_thickness
                        else:
                            output_graph.add_edge(author_lst[authors_yearly[n[0]][edge[0]]], author_lst[authors_yearly[n[0]][edge[1]]], weight = subgraph[edge[0]][edge[1]]['weight']) # /timeslice_thickness)
                if counter >= len(path) - 2:
                        break
                counter = counter + 1
            end = datetime.now()
            print(target_fun, args.number_of_sol_arg-number_of_solutions, " Creating output graph and csv content and writing communities in path to txt: ", end-start)
            before_postprocessing = output_graph_updated.copy()


            while postprocessing_type_updated < postprocessing_type_max:
                output_graph_updated = before_postprocessing.copy()
                if postprocessing_type_updated == 0:
                    # initially remove authors which are not present sufficiently often (factor subject to change) !!! might split community !!!
                    remove_nodes = []
                    for node in output_graph_updated:
                        max_edge_weight = 0
                        for edge in output_graph_updated.edges(node):
                            if output_graph_updated.get_edge_data(edge[0], edge[1])['weight'] > max_edge_weight:
                                max_edge_weight = output_graph_updated.get_edge_data(edge[0], edge[1])['weight']
                        if max_edge_weight < (len(path) - 2)/5:
                            remove_nodes.append(node)
                    output_graph_updated.remove_nodes_from(remove_nodes)

                    # remove authors that are not sufficiently connected to the core community
                    while True:
                        remove_nodes = []
                        for node in output_graph_updated:
                            if output_graph_updated.degree(node) < 2:
                                remove_nodes.append(node)
                        output_graph_updated.remove_nodes_from(remove_nodes)
                        if len(remove_nodes) < 1:
                            break

                    end = datetime.now()


                    # calculate fitness scores for core community
                    sizes_core = []
                    edge_dens_core = []
                    shareds_core = []
                    embeds_core = []
                    average_edge_den_core = 0
                    edge_den_sd_core = 0
                    average_size_core = 0
                    size_sd_core = 0
                    average_shared_core = 0
                    shared_sd_core = 0
                    average_embed_core = 0
                    embed_sd_core = 0
                    
                    authors_core_calc = []
                    for author_str in list(output_graph_updated):
                        authors_core_calc.append(author_lst.index(author_str))
                    for idx, n in enumerate(path):
                        if idx > 0:
                            author_indices = []
                            authors_current = []
                            for author in authors_core_calc:
                                if author in authors_yearly[n[0]]:
                                    if authors_yearly[n[0]].index(author) in id_to_community[n[0] - start_year][n[1]]:
                                        author_indices.append(authors_yearly[n[0]].index(author))
                                        authors_current.append(author)
                            if len(author_indices) > 0:
                                core_comm_clustering = NodeClustering([author_indices], collab_graphs[n[0]], "core_community")
                                edge_dens_core.append(evaluation.internal_edge_density(core_comm_clustering.graph, core_comm_clustering, summary=False)[0])
                                sizes_core.append(len(author_indices))
                                embeds_core.append(evaluation.avg_embeddedness(core_comm_clustering.graph, core_comm_clustering, summary=False)[0])
                                if idx > 1:
                                    shareds_core.append(compare_communities(authors_current, authors_previous))
                            authors_previous = authors_current.copy()
                        if idx >= len(path) - 2:
                            break
                    average_edge_den_core = np.mean(edge_dens_core)
                    edge_den_sd_core = np.std(edge_dens_core)
                    average_size_core = np.mean(sizes_core)
                    size_sd_core = np.std(sizes_core)
                    average_embed_core = np.mean(embeds_core)
                    embed_sd_core = np.std(embeds_core)
                    if len(path) > 3:
                        average_shared_core = np.mean(shareds_core)
                        shared_sd_core = np.std(shareds_core)
                    output_csv_core = open(output_path + "outputSingleSourceTargetSTD" + str(testset) +  str(timeslice_thickness) + str(args.number_of_sol_arg) + str(target_fun) + str(community_alg) + str(cutoff) + ".csv", 'a')
                    print(args.number_of_sol_arg - number_of_solutions, ", ", len(path) - 2, ", ", average_size_core, ", ", size_sd_core, ", ", average_shared_core, ", ", shared_sd_core, ", ", average_edge_den_core, ", ", edge_den_sd_core, ", ", average_embed_core, ", ", embed_sd_core, file = output_csv_core)
                    print(target_fun, args.number_of_sol_arg-number_of_solutions, " Postprocessing (if exists): ", end-start)
                    postprocessed_out = open(output_path + "outputSingleSourceTargetSTD" + str(testset) +  str(timeslice_thickness) + str(args.number_of_sol_arg) + str(target_fun) + str(community_alg) + str(cutoff) + str(args.number_of_sol_arg - number_of_solutions) + ".json", 'w')
                    print("Remaining nodes after postprocessing: ", output_graph_updated.nodes(), file=output)
                    color_map = list(output_graph_updated)
                    c = json.dumps(color_map)
                    print(c, file = postprocessed_out)

                if postprocessing_type_updated == 1:

                    start = datetime.now()
                    # initially remove authors which are not present sufficiently often (factor subject to change) !!! might split community !!!
                    remove_nodes = []
                    average_degree = 0
                    for node in output_graph_updated:
                        average_degree += output_graph_updated.degree(node)
                    average_degree /= output_graph_updated.number_of_nodes()
                    for node in output_graph_updated:
                        if output_graph_updated.degree(node) < average_degree:
                            remove_nodes.append(node)
                    output_graph_updated.remove_nodes_from(remove_nodes)

                    # remove authors that are not sufficiently connected to the core community
                    while True:
                        remove_nodes = []
                        for node in output_graph_updated:
                            if output_graph_updated.degree(node) < 2:
                                remove_nodes.append(node)
                        output_graph_updated.remove_nodes_from(remove_nodes)
                        if len(remove_nodes) < 1:
                            break
                    end = datetime.now()

                    # calculate fitness scores for core community
                    sizes_core = []
                    edge_dens_core = []
                    shareds_core = []
                    embeds_core = []
                    average_edge_den_core = 0
                    edge_den_sd_core = 0
                    average_size_core = 0
                    size_sd_core = 0
                    average_shared_core = 0
                    shared_sd_core = 0
                    average_embed_core = 0
                    embed_sd_core = 0
                    
                    authors_core_calc = []
                    for author_str in list(output_graph_updated):
                        authors_core_calc.append(author_lst.index(author_str))
                    for idx, n in enumerate(path):
                        if idx > 0:
                            author_indices = []
                            authors_current = []
                            for author in authors_core_calc:
                                if author in authors_yearly[n[0]]:
                                    if authors_yearly[n[0]].index(author) in id_to_community[n[0] - start_year][n[1]]:
                                        author_indices.append(authors_yearly[n[0]].index(author))
                                        authors_current.append(author)
                            if len(author_indices) > 0:
                                core_comm_clustering = NodeClustering([author_indices], collab_graphs[n[0]], "core_community")
                                edge_dens_core.append(evaluation.internal_edge_density(core_comm_clustering.graph, core_comm_clustering, summary=False)[0])
                                sizes_core.append(len(author_indices))
                                embeds_core.append(evaluation.avg_embeddedness(core_comm_clustering.graph, core_comm_clustering, summary=False)[0])
                                if idx > 1:
                                    shareds_core.append(compare_communities(authors_current, authors_previous))
                            authors_previous = authors_current.copy()
                        if idx >= len(path) - 2:
                            break
                    average_edge_den_core = np.mean(edge_dens_core)
                    edge_den_sd_core = np.std(edge_dens_core)
                    average_size_core = np.mean(sizes_core)
                    size_sd_core = np.std(sizes_core)
                    average_embed_core = np.mean(embeds_core)
                    embed_sd_core = np.std(embeds_core)
                    if len(path) > 3:
                        average_shared_core = np.mean(shareds_core)
                        shared_sd_core = np.std(shareds_core)
                    output_csv_core = open(output_path + "outputSingleSourceTargetAVD" + str(testset) +  str(timeslice_thickness) + str(args.number_of_sol_arg) + str(target_fun) + str(community_alg) + str(cutoff) + ".csv", 'a')
                    print(args.number_of_sol_arg - number_of_solutions, ", ", len(path) - 2, ", ", average_size_core, ", ", size_sd_core, ", ", average_shared_core, ", ", shared_sd_core, ", ", average_edge_den_core, ", ", edge_den_sd_core, ", ", average_embed_core, ", ", embed_sd_core, file = output_csv_core)
                    print(target_fun, args.number_of_sol_arg-number_of_solutions, " Postprocessing (if exists): ", end-start)
                    postprocessed_out = open(output_path + "outputSingleSourceTargetAVD" + str(testset) +  str(timeslice_thickness) + str(args.number_of_sol_arg) + str(target_fun) + str(community_alg) + str(cutoff) + str(args.number_of_sol_arg - number_of_solutions) + ".json", 'w')
                    print("Remaining nodes after postprocessing: ", output_graph_updated.nodes(), file=output)
                    color_map = list(output_graph_updated)
                    c = json.dumps(color_map)
                    print(c, file = postprocessed_out)

                if postprocessing_type_updated == 2:

                    start = datetime.now()
                    # initially remove authors which are not present sufficiently often (factor subject to change) !!! might split community !!!
                    remove_nodes = []
                    average_degree = 0
                    for node in output_graph_updated:
                        average_degree += output_graph_updated.degree(node)
                    average_degree /= output_graph_updated.number_of_nodes()
                    for node in output_graph_updated:
                        if output_graph_updated.degree(node) < average_degree*0.75:
                            remove_nodes.append(node)
                    output_graph_updated.remove_nodes_from(remove_nodes)

                    # remove authors that are not sufficiently connected to the core community
                    while True:
                        remove_nodes = []
                        for node in output_graph_updated:
                            if output_graph_updated.degree(node) < 2:
                                remove_nodes.append(node)
                        output_graph_updated.remove_nodes_from(remove_nodes)
                        if len(remove_nodes) < 1:
                            break
                    end = datetime.now()

                    # calculate fitness scores for core community
                    sizes_core = []
                    edge_dens_core = []
                    shareds_core = []
                    embeds_core = []
                    average_edge_den_core = 0
                    edge_den_sd_core = 0
                    average_size_core = 0
                    size_sd_core = 0
                    average_shared_core = 0
                    shared_sd_core = 0
                    average_embed_core = 0
                    embed_sd_core = 0
                    
                    authors_core_calc = []
                    for author_str in list(output_graph_updated):
                        authors_core_calc.append(author_lst.index(author_str))
                    for idx, n in enumerate(path):
                        if idx > 0:
                            author_indices = []
                            authors_current = []
                            for author in authors_core_calc:
                                if author in authors_yearly[n[0]]:
                                    if authors_yearly[n[0]].index(author) in id_to_community[n[0] - start_year][n[1]]:
                                        author_indices.append(authors_yearly[n[0]].index(author))
                                        authors_current.append(author)
                            if len(author_indices) > 0:
                                core_comm_clustering = NodeClustering([author_indices], collab_graphs[n[0]], "core_community")
                                edge_dens_core.append(evaluation.internal_edge_density(core_comm_clustering.graph, core_comm_clustering, summary=False)[0])
                                sizes_core.append(len(author_indices))
                                embeds_core.append(evaluation.avg_embeddedness(core_comm_clustering.graph, core_comm_clustering, summary=False)[0])
                                if idx > 1:
                                    shareds_core.append(compare_communities(authors_current, authors_previous))
                            authors_previous = authors_current.copy()
                        if idx >= len(path) - 2:
                            break
                    average_edge_den_core = np.mean(edge_dens_core)
                    edge_den_sd_core = np.std(edge_dens_core)
                    average_size_core = np.mean(sizes_core)
                    size_sd_core = np.std(sizes_core)
                    average_embed_core = np.mean(embeds_core)
                    embed_sd_core = np.std(embeds_core)
                    if len(path) > 3:
                        average_shared_core = np.mean(shareds_core)
                        shared_sd_core = np.std(shareds_core)
                    output_csv_core = open(output_path + "outputSingleSourceTargetAVDWeaker" + str(testset) +  str(timeslice_thickness) + str(args.number_of_sol_arg) + str(target_fun) + str(community_alg) + str(cutoff) + ".csv", 'a')
                    print(args.number_of_sol_arg - number_of_solutions, ", ", len(path) - 2, ", ", average_size_core, ", ", size_sd_core, ", ", average_shared_core, ", ", shared_sd_core, ", ", average_edge_den_core, ", ", edge_den_sd_core, ", ", average_embed_core, ", ", embed_sd_core, file = output_csv_core)
                    print(target_fun, args.number_of_sol_arg-number_of_solutions, " Postprocessing (if exists): ", end-start)
                    postprocessed_out = open(output_path + "outputSingleSourceTargetAVDWeaker" + str(testset) +  str(timeslice_thickness) + str(args.number_of_sol_arg) + str(target_fun) + str(community_alg) + str(cutoff) + str(args.number_of_sol_arg - number_of_solutions) + ".json", 'w')
                    print("Remaining nodes after postprocessing: ", output_graph_updated.nodes(), file=output)
                    color_map = list(output_graph_updated)
                    c = json.dumps(color_map)
                    print(c, file = postprocessed_out)

                if postprocessing_type_updated == 3:

                    start = datetime.now()
                    # initially remove authors which are not present sufficiently often (factor subject to change) !!! might split community !!!
                    remove_nodes = []
                    for node in output_graph_updated:
                        occurences = 0
                        for idx, n in enumerate(path):
                            if idx > 0 and idx < len(path) - 1:
                                if node in [*itemgetter(*itemgetter(*id_to_community[n[0] - start_year][n[1]])(authors_yearly[n[0]]))(author_lst)]:
                                    occurences += 1
                        if occurences <= timeslice_thickness:
                            remove_nodes.append(node)
                    output_graph_updated.remove_nodes_from(remove_nodes)
                    # remove authors that are not sufficiently connected to the core community
                    while True:
                        remove_nodes = []
                        for node in output_graph_updated:
                            if output_graph_updated.degree(node) < 2:
                                remove_nodes.append(node)
                        output_graph_updated.remove_nodes_from(remove_nodes)
                        if len(remove_nodes) < 1:
                            break
                    end = datetime.now()

                    # calculate fitness scores for core community
                    sizes_core = []
                    edge_dens_core = []
                    shareds_core = []
                    embeds_core = []
                    average_edge_den_core = 0
                    edge_den_sd_core = 0
                    average_size_core = 0
                    size_sd_core = 0
                    average_shared_core = 0
                    shared_sd_core = 0
                    average_embed_core = 0
                    embed_sd_core = 0
                    
                    authors_core_calc = []
                    for author_str in list(output_graph_updated):
                        authors_core_calc.append(author_lst.index(author_str))
                    for idx, n in enumerate(path):
                        if idx > 0:
                            author_indices = []
                            authors_current = []
                            for author in authors_core_calc:
                                if author in authors_yearly[n[0]]:
                                    if authors_yearly[n[0]].index(author) in id_to_community[n[0] - start_year][n[1]]:
                                        author_indices.append(authors_yearly[n[0]].index(author))
                                        authors_current.append(author)
                            if len(author_indices) > 0:
                                core_comm_clustering = NodeClustering([author_indices], collab_graphs[n[0]], "core_community")
                                edge_dens_core.append(evaluation.internal_edge_density(core_comm_clustering.graph, core_comm_clustering, summary=False)[0])
                                sizes_core.append(len(author_indices))
                                embeds_core.append(evaluation.avg_embeddedness(core_comm_clustering.graph, core_comm_clustering, summary=False)[0])
                                if idx > 1:
                                    shareds_core.append(compare_communities(authors_current, authors_previous))
                            authors_previous = authors_current.copy()
                        if idx >= len(path) - 2:
                            break
                    average_edge_den_core = np.mean(edge_dens_core)
                    edge_den_sd_core = np.std(edge_dens_core)
                    average_size_core = np.mean(sizes_core)
                    size_sd_core = np.std(sizes_core)
                    average_embed_core = np.mean(embeds_core)
                    embed_sd_core = np.std(embeds_core)
                    if len(path) > 3:
                        average_shared_core = np.mean(shareds_core)
                        shared_sd_core = np.std(shareds_core)
                    output_csv_core = open(output_path + "outputSingleSourceTargetLight" + str(testset) +  str(timeslice_thickness) + str(args.number_of_sol_arg) + str(target_fun) + str(community_alg) + str(cutoff) + ".csv", 'a')
                    print(args.number_of_sol_arg - number_of_solutions, ", ", len(path) - 2, ", ", average_size_core, ", ", size_sd_core, ", ", average_shared_core, ", ", shared_sd_core, ", ", average_edge_den_core, ", ", edge_den_sd_core, ", ", average_embed_core, ", ", embed_sd_core, file = output_csv_core)
                    print(target_fun, args.number_of_sol_arg-number_of_solutions, " Postprocessing (if exists): ", end-start)
                    postprocessed_out = open(output_path + "outputSingleSourceTargetLight" + str(testset) +  str(timeslice_thickness) + str(args.number_of_sol_arg) + str(target_fun) + str(community_alg) + str(cutoff) + str(args.number_of_sol_arg - number_of_solutions) + ".json", 'w')
                    print("Remaining nodes after postprocessing: ", output_graph_updated.nodes(), file=output)
                    color_map = list(output_graph_updated)
                    c = json.dumps(color_map)
                    print(c, file = postprocessed_out)
            
                if args.postprocessing_type < 10:
                    break
                postprocessing_type_updated += 1

            json_out = open(output_path + "outputSingleSourceTarget" + str(testset) +  str(timeslice_thickness) + str(args.number_of_sol_arg) + str(target_fun) + str(community_alg) + str(cutoff) + str(args.number_of_sol_arg - number_of_solutions) + ".json", 'w')
            data = json_graph.adjacency_data(output_graph)
            s = json.dumps(data)
            print(s, file = json_out)

            communities_out = open(output_path + "outputSingleSourceTargetCommunities" + str(testset) +  str(timeslice_thickness) + str(args.number_of_sol_arg) + str(target_fun) + str(community_alg) + str(cutoff) + str(args.number_of_sol_arg - number_of_solutions) + ".json", 'w')
            print(json.dumps(list(before_postprocessing)), file = communities_out)
            #setup for next pass
            start = datetime.now()
            number_of_solutions -= 1
            if number_of_solutions == 0:
                break
            counter = 0
            removed = 0
            if args.core_remove:
                for idx, node in enumerate(path): # check every community on path, remove all communities that contain used authors from comparison graph
                    if counter > 0:
                        comm = id_to_community[node[0] - start_year][node[1]]
                        for author in [*itemgetter(*id_to_community[node[0] - start_year][node[1]])(authors_yearly[node[0]])]: # author = single author node in community
                            for check_comm in range(len(id_to_community[node[0] - start_year])):
                                if author in [*itemgetter(*id_to_community[node[0] - start_year][check_comm])(authors_yearly[node[0]])] and (node[0], check_comm) in comparison_graph:
                                    if author_lst[author] in list(output_graph_updated.nodes):
                                        comparison_graph.remove_node((node[0], check_comm))
                                        removed += 1
                    if counter >= len(path) - 2:
                        break
                    counter = counter + 1
            else:
                for idx, node in enumerate(path): # check every community on path, remove all communities that contain used authors from comparison graph
                    if counter > 0:
                        comm = id_to_community[node[0] - start_year][node[1]]
                        for author in [*itemgetter(*id_to_community[node[0] - start_year][node[1]])(authors_yearly[node[0]])]: # author = single author node in community
                            for check_comm in range(len(id_to_community[node[0] - start_year])):
                                if author in [*itemgetter(*id_to_community[node[0] - start_year][check_comm])(authors_yearly[node[0]])] and (node[0], check_comm) in comparison_graph:
                                    comparison_graph.remove_node((node[0], check_comm))
                                    removed += 1
                    if counter >= len(path) - 2:
                        break
                    counter = counter + 1
            end = datetime.now()
            print(target_fun, args.number_of_sol_arg-number_of_solutions, " Setting up for next pass (removing used communities from comparison graph): ", end-start)
        if args.target_function < 10 or target_fun == 5: # hier anzahl zielfunktionen
            break
        else:
            target_fun += 1

def compare_communities(first, second):
    global compare_total
    compare_start = datetime.now()
    result = len(set(first) & set(second))/len(set(first) | set(second))
    compare_end = datetime.now()
    compare_total += compare_end - compare_start
    return result
if __name__ == "__main__":
    main()