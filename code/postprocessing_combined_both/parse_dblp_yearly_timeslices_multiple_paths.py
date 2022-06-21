from errno import EBADMSG
import lxml.etree as ET
import networkx as nx
import itertools as IT
import os
from cdlib import algorithms, evaluation, NodeClustering
from datetime import datetime
import numpy as np
from io import BytesIO
import sys
import argparse
from networkx.readwrite import json_graph
import json

from pyparsing import col

# CLEAN UP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('testset_arg', type=int)
    parser.add_argument('thickness_arg', type=int)
    parser.add_argument('number_of_sol_arg', type=int)
    parser.add_argument('target_function', type=int) #10 für alle
    parser.add_argument('comm_alg', type=int) #10 for all
    parser.add_argument('xml_path', type=str)
    parser.add_argument('output_path', type=str)

    args = parser.parse_args()
    if len(sys.argv) > 1:
        testset = args.testset_arg
        timeslice_thickness = args.thickness_arg
        number_of_solutions = args.number_of_sol_arg
        if args.target_function < 10:
            target_fun = args.target_function
        else:
            target_fun = 0
        community_alg = args.comm_alg
    else:
        testset = 1 # 1, 2 Atzmüller; 3, 4 Chimani
        timeslice_thickness = 3
        number_of_solutions = 1
        target_fun = 0
        community_alg = 0
    dblp_path = args.xml_path
    if not os.path.exists(dblp_path):
        print("Error finding dblp.xml, did you enter the right path?")
    output_path = args.output_path
    if not os.path.exists(output_path):
        print("Error finding output path, did you enter the right path?")
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
    for y in collab_graphs:
        for u, v, d in collab_graphs[y].edges(data=True):
            d['weight'] = 1 + d['weight']*0.1



    communities = {}
    for y in range(start_year, end_year + 1 - timeslice_thickness + 1):
        # else years with no collaborative papers mess up the comparison graph
        if len(collab_graphs[y]) > 0:
        # put different communities in
            communities[y] = []
            if community_alg == 0 or community_alg == 10:
                for i in range(0, 10):
                    communities[y].extend(algorithms.louvain(collab_graphs[y].copy(), weight = 'weight', resolution=0.25*(i + 1), randomize=i).communities)
            if community_alg == 1 or community_alg == 10:
                for i in range(0,10):
                    communities[y].extend(algorithms.chinesewhispers(collab_graphs[y].copy(), iterations=2*(i + 1), seed=i).communities)
            if community_alg == 2 or community_alg == 10:
                for i in range(0,10):
                    communities[y].extend(algorithms.scd(collab_graphs[y].copy(), iterations=3*(i+1), seed=i).communities)
            if community_alg == 3 or community_alg == 10:
                for i in range(0,10):
                    communities[y].extend(algorithms.pycombo(collab_graphs[y].copy(), modularity_resolution=0.25*(i + 1), random_seed=i).communities)            
            
            for c in communities[y][:]:
                if len(c) < 3:
                    communities[y].remove(c)


    # get reference for individual communities to store in comparison graph
    #TODO: wenn community liste von listen eine ebene tiefer knoten extrahieren
    id_to_community = {} # vllt nach Jahren aufsplitten damit funktioniert
    community_nodes = {}
    for y in range(start_year, end_year + 1 - timeslice_thickness + 1):
        id_to_community[y] = []
        community_nodes[y] = []
        for c in communities[y]:
            nodes = set()
            for node in c:
                nodes.add(authors_yearly[y][node])
            id_to_community[y].append(c)
            community_nodes[y].append(nodes)

    community_calcs = {} # add list of community, embeddedness, density
    for y in range(start_year, end_year + 1 - timeslice_thickness + 1):
        for c in communities[y]:
            community_calcs[id_to_community[y].index(c)] = []
            clustering = NodeClustering([c], graph=collab_graphs[y], method_name="egal")
            community_calcs[id_to_community[y].index(c)].append(evaluation.avg_embeddedness(collab_graphs[y], clustering, summary=False)[0])
            community_calcs[id_to_community[y].index(c)].append(evaluation.internal_edge_density(collab_graphs[y], clustering, summary=False)[0])


    while args.target_function >= 0:

        number_of_solutions = args.number_of_sol_arg
        output = open(output_path + "output" + str(testset) +  str(timeslice_thickness) + str(number_of_solutions) + str(target_fun) + str(community_alg) + ".txt", 'w')

        comparison_graph = nx.DiGraph()
        i = 0
        epsilon = 0.001
        # fill comparison graph
        # even numbers for source nodes, odd numbers for target nodes
        #TODO: WAHRSCHEINLICH HIER
        for y in range(start_year, end_year + 1 - timeslice_thickness + 1):
            if i == 0:
                comparison_graph.add_node(0) # maybe (y, start) and (y, end)
                comparison_graph.add_node(1)
                for c in communities[y]:
                    comparison_graph.add_node((y, id_to_community[y].index(c)))
                    comparison_graph.add_edge(0, (y, id_to_community[y].index(c)), weight = 1)
                    comparison_graph.add_edge((y, id_to_community[y].index(c)), 1, weight = 1)
            else:
                comparison_graph.add_node(i * 2)
                comparison_graph.add_node(i * 2 + 1)
                for c in communities[y]:
                    if (y, id_to_community[y].index(c)) in comparison_graph:
                        continue
                    comparison_graph.add_node((y, id_to_community[y].index(c)))
                    # if one huge difference between communities, they likely aren't the same one -> maybe flat values; didnt't work so try multiple paths

                    comparison_graph.add_edge(i * 2, (y, id_to_community[y].index(c)), weight = 1) #TODO: hier justieren, wie viele Knoten in kürzestem Pfad sind (kleiner = weniger)
                    comparison_graph.add_edge((y, id_to_community[y].index(c)), i * 2 + 1, weight = 1)

                    for c2 in communities[previous_year]:
                        base_weight = compare_communities(community_nodes[y][id_to_community[y].index(c)], community_nodes[previous_year][id_to_community[previous_year].index(c2)])
                        if base_weight > epsilon and target_fun == 0:
                            base_weight = 1/(base_weight * ((community_calcs[id_to_community[y].index(c)][0] + community_calcs[id_to_community[previous_year].index(c2)][0])/2) * ((community_calcs[id_to_community[y].index(c)][1] + community_calcs[id_to_community[previous_year].index(c2)][1])/2))
                        elif base_weight > epsilon and target_fun == 1:
                            base_weight = 1-(base_weight * ((community_calcs[id_to_community[y].index(c)][0] + community_calcs[id_to_community[previous_year].index(c2)][0])/2) * ((community_calcs[id_to_community[y].index(c)][1] + community_calcs[id_to_community[previous_year].index(c2)][1])/2))
                        elif base_weight > epsilon and target_fun == 2:
                            if base_weight < 0.5:
                                base_weight = 100000
                            else:
                                base_weight = 1/(base_weight * ((community_calcs[id_to_community[y].index(c)][0] + community_calcs[id_to_community[previous_year].index(c2)][0])/2) * ((community_calcs[id_to_community[y].index(c)][1] + community_calcs[id_to_community[previous_year].index(c2)][1])/2))
                        elif base_weight > epsilon and target_fun == 3:
                            if base_weight < 0.5:
                                base_weight = 100000
                            else:
                                base_weight = 1-(base_weight * ((community_calcs[id_to_community[y].index(c)][0] + community_calcs[id_to_community[previous_year].index(c2)][0])/2) * ((community_calcs[id_to_community[y].index(c)][1] + community_calcs[id_to_community[previous_year].index(c2)][1])/2))
                        elif base_weight > epsilon and target_fun == 4:
                            if base_weight < 0.5:
                                base_weight = 100000
                            else:
                                base_weight = (1-base_weight)
                        else:
                            base_weight = 100000
                        comparison_graph.add_edge((previous_year, id_to_community[previous_year].index(c2)), (y, id_to_community[y].index(c)), weight = base_weight)

            previous_year = y
            i = i + 1
            if i == len(communities):
                for c in communities[y]:
                    comparison_graph.add_edge((y, id_to_community[y].index(c)), "end", weight = 1)


        num_communities = 0
        for y in communities:
            num_communities = num_communities + len(communities[y])
        print("Total number of communities: ", num_communities, file=output)

        output_csv = open(output_path + "output" + str(testset) +  str(timeslice_thickness) + str(number_of_solutions) + str(target_fun) + str(community_alg) + ".csv", 'w')
        output_csv_core = open(output_path + "outputCore" + str(testset) +  str(timeslice_thickness) + str(number_of_solutions) + str(target_fun) + str(community_alg) + ".csv", 'w')

        while number_of_solutions > 0:
            # find shortest path in comparison graph to find most similar communities in different years
            paths = []
            for i in range(0, end_year + 1 -start_year - timeslice_thickness + 1):
                for j in range(i, end_year + 1 -start_year - timeslice_thickness + 1):
                    counter = 0
                    paths.append(nx.astar_path(comparison_graph, i * 2, j * 2 + 1))
                    # TODO: make sure this is correct
                    for n in paths[i * (end_year + 1 -start_year - timeslice_thickness - j)  + j]: # calculate something for fitness / result quality here, maybe average shared nodes or something with path length over number of communities in path ^some power between 0 and 1
                        conferences_in_comm = set()
                        if counter > 0:
                            for author in community_nodes[n[0]][n[1]]:
                                conferences_in_comm = conferences_in_comm.union(author_conferences[author])
                        counter = counter + 1
                        if counter == len(paths[i * (end_year + 1 -start_year - timeslice_thickness - j)  + j]) - 1:
                            break


            best_average_size = 0
            best_average_edge_den = 0
            best_average_edge_den_start = -1
            best_average_edge_den_end = -1
            best_output_edge_den = 0
            best_average_size = 0
            best_average_shared = 0
            best_average_embed = 0
            average_shared = 0
            path_len_out = 0
            for i in range(0, end_year + 1 -start_year - timeslice_thickness + 1):
                
                for j in range(i, end_year + 1 -start_year - timeslice_thickness + 1):
                    path_len = 0
                    edge_den_accum = 0
                    size_accum = 0
                    path_len_out = 0
                    average_shared = 0
                    average_embed = 0
                    for idx, n in enumerate(paths[i * (end_year + 1 -start_year - timeslice_thickness - j)  + j]): 
                        if path_len > 0:
                            calc_clustering = NodeClustering([id_to_community[n[0]][n[1]]], graph=collab_graphs[n[0]], method_name="egal")
                            edge_den_local = evaluation.internal_edge_density(calc_clustering.graph, calc_clustering, summary=False)[0]
                            avg_embed_local = evaluation.avg_embeddedness(calc_clustering.graph, calc_clustering, summary=False)[0]
                            edge_den_accum += edge_den_local
                            size_accum += len(calc_clustering.communities[0])
                            average_embed += avg_embed_local
                            path_len_out += comparison_graph.get_edge_data(n, paths[i * (end_year + 1 -start_year - timeslice_thickness - j)  + j][paths[i * (end_year + 1 -start_year - timeslice_thickness - j)  + j].index(n) + 1])['weight']
                            if path_len < len(paths[i * (end_year + 1 - start_year - timeslice_thickness - j) + j]) - 2:
                                next_item = paths[i * (end_year + 1 - start_year - timeslice_thickness - j) + j][idx + 1]
                                shared_local = compare_communities(community_nodes[n[0]][n[1]], community_nodes[next_item[0]][next_item[1]])
                                if shared_local < epsilon:
                                    break
                                average_shared += shared_local
                        path_len = path_len + 1
                        if path_len == len(paths[i * (end_year + 1 - start_year - timeslice_thickness - j) + j]) - 1:
                            average_size = size_accum/(path_len - 1)
                            output_edge_den = edge_den_accum/(path_len - 1)
                            average_shared /= path_len - 1
                            average_embed /= path_len - 1
                            average_edge_den = average_shared * average_size
                            if average_edge_den > best_average_edge_den:
                                best_average_edge_den = average_edge_den
                                best_average_edge_den_start = i
                                best_average_edge_den_end = j
                                best_output_edge_den = output_edge_den
                                best_average_size = average_size
                                best_average_embed = average_embed
                            break

            print("Best path: ", file = output)
            counter = 0
            i = best_average_edge_den_start
            j = best_average_edge_den_end
            enumerate_obj = paths[i * (end_year + 1 -start_year - timeslice_thickness - j)  + j]
            for idx, n in enumerate(enumerate_obj):
                print(n, file = output)
                conferences_in_comm = set()
                if counter > 0:
                    print(counter, file = output)
                    if counter == len(paths[i * (end_year + 1 -start_year - timeslice_thickness - j)  + j]) - 1:
                        break
                    print(community_nodes[n[0]][n[1]], file = output)
                    calc_clustering = NodeClustering([id_to_community[n[0]][n[1]]], graph=collab_graphs[n[0]], method_name="egal")
                    edge_den_local = evaluation.internal_edge_density(calc_clustering.graph, calc_clustering, summary=False)
                    print("edge density: ", edge_den_local, file = output)
                    for author in community_nodes[n[0]][n[1]]:
                        conferences_in_comm = conferences_in_comm.union(author_conferences[author])
                    print(conferences_in_comm, file = output)
                    if counter < len(paths[i * (end_year + 1 -start_year - timeslice_thickness - j)  + j]) - 2:
                        next_item = paths[i * (end_year + 1 - start_year - timeslice_thickness - j) + j][idx + 1]
                        average_shared = compare_communities(community_nodes[n[0]][n[1]], community_nodes[next_item[0]][next_item[1]])
                        best_average_shared += average_shared
                        print("shared nodes %: ", compare_communities(community_nodes[n[0]][n[1]], community_nodes[next_item[0]][next_item[1]]), file = output)
                    print(comparison_graph.get_edge_data(n, paths[i * (end_year + 1 -start_year - timeslice_thickness - j)  + j][paths[i * (end_year + 1 -start_year - timeslice_thickness - j)  + j].index(n) + 1]), file = output)
                counter = counter + 1
                if counter == len(paths[i * (end_year + 1 -start_year - timeslice_thickness - j)  + j]) - 1:
                    break
            if len(paths[i * (end_year + 1 -start_year - timeslice_thickness - j)  + j]) > 3:
                best_average_shared /= (len(paths[i * (end_year + 1 - start_year - timeslice_thickness - j)  + j]) - 3)
            print(args.number_of_sol_arg - number_of_solutions, ", ", len(paths[i * (end_year + 1 - start_year - timeslice_thickness - j)  + j]) - 2 , ", ", best_output_edge_den, ", ", best_average_size, ", ", best_average_embed, ", ", best_average_shared , file = output_csv)





            # create output graph
            output_graph = nx.Graph()
            counter = 0
            color_map = []
            core_community = set()
            output_graph_updated = nx.Graph()
            for idx, n in enumerate(enumerate_obj):
                if counter > 0:
                    comm = id_to_community[n[0]][n[1]]
                    add_nodes = set(comm)
                    add_size = len(add_nodes)
                    core_community = set(comm) # add nodes from community chain to "core community"
                    comms_only_subgraph = nx.subgraph(collab_graphs[n[0]], core_community)
                    for node in comms_only_subgraph.nodes():
                        output_graph_updated.add_node(authors_yearly[n[0]][node])
                    for edge in comms_only_subgraph.edges():
                        if output_graph_updated.has_edge(authors_yearly[n[0]][edge[0]], authors_yearly[n[0]][edge[1]]):
                            output_graph_updated[authors_yearly[n[0]][edge[0]]][authors_yearly[n[0]][edge[1]]]['weight'] = output_graph_updated[authors_yearly[n[0]][edge[0]]][authors_yearly[n[0]][edge[1]]]['weight'] + comms_only_subgraph[edge[0]][edge[1]]['weight'] # /timeslice_thickness
                        else:
                            output_graph_updated.add_edge(authors_yearly[n[0]][edge[0]], authors_yearly[n[0]][edge[1]], weight = comms_only_subgraph[edge[0]][edge[1]]['weight']) # /timeslice_thickness)
                    for node in comm:
                        color_map.append(authors_yearly[n[0]][node])
                        add_nodes = add_nodes.union(set(collab_graphs[n[0]].neighbors(node)))
                    subgraph = nx.subgraph(collab_graphs[n[0]], add_nodes)
                    for node in subgraph.nodes():
                        output_graph.add_node(authors_yearly[n[0]][node])
                    for edge in subgraph.edges():
                        if output_graph.has_edge(authors_yearly[n[0]][edge[0]], authors_yearly[n[0]][edge[1]]):
                            output_graph[authors_yearly[n[0]][edge[0]]][authors_yearly[n[0]][edge[1]]]['weight'] = output_graph[authors_yearly[n[0]][edge[0]]][authors_yearly[n[0]][edge[1]]]['weight'] + subgraph[edge[0]][edge[1]]['weight'] # /timeslice_thickness
                        else:
                            output_graph.add_edge(authors_yearly[n[0]][edge[0]], authors_yearly[n[0]][edge[1]], weight = subgraph[edge[0]][edge[1]]['weight']) # /timeslice_thickness)
                if counter >= len(paths[i * (end_year + 1 -start_year - timeslice_thickness - j)  + j]) - 2:
                        break
                counter = counter + 1
            # initially remove authors which are not present sufficiently often (factor subject to change) !!! might split community !!!
            remove_nodes = []
            for node in output_graph_updated:
                max_edge_weight = 0
                for edge in output_graph_updated.edges(node):
                    if output_graph_updated.get_edge_data(edge[0], edge[1])['weight'] > max_edge_weight:
                        max_edge_weight = output_graph_updated.get_edge_data(edge[0], edge[1])['weight']
                if max_edge_weight < (len(paths[i * (end_year + 1 - start_year - timeslice_thickness - j)  + j]) - 2)/5:
                    remove_nodes.append(node)
            output_graph_updated.remove_nodes_from(remove_nodes)

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


            # calculate fitness scores for core community
            average_edge_den_core = 0
            average_size_core = 0
            average_shared_core = 0
            average_embed_core = 0
            for idx, n in enumerate(paths[i * (end_year + 1 -start_year - timeslice_thickness - j)  + j]):
                if idx > 0:
                    author_indices = []
                    authors_current = []
                    for author in list(output_graph_updated):
                        if author in authors_yearly[n[0]]:
                            if authors_yearly[n[0]].index(author) in id_to_community[n[0]][n[1]]:
                                author_indices.append(authors_yearly[n[0]].index(author))
                                authors_current.append(author)
                    if len(author_indices) > 0:
                        core_comm_clustering = NodeClustering([author_indices], collab_graphs[n[0]], "core_community")
                        average_edge_den_core += evaluation.internal_edge_density(core_comm_clustering.graph, core_comm_clustering, summary=False)[0]
                        average_size_core += len(author_indices)
                        average_embed_core += evaluation.avg_embeddedness(core_comm_clustering.graph, core_comm_clustering, summary=False)[0]
                        if idx > 1:
                            average_shared_core += compare_communities(authors_current, authors_previous)
                    authors_previous = authors_current.copy()
                if idx >= len(paths[i * (end_year + 1 -start_year - timeslice_thickness - j)  + j]) - 2:
                    break
            average_edge_den_core /= len(paths[i * (end_year + 1 -start_year - timeslice_thickness - j)  + j]) - 2
            average_size_core /= len(paths[i * (end_year + 1 -start_year - timeslice_thickness - j)  + j]) - 2
            average_embed_core /= len(paths[i * (end_year + 1 -start_year - timeslice_thickness - j)  + j]) - 2
            if len(paths[i * (end_year + 1 -start_year - timeslice_thickness - j)  + j]) > 3:
                average_shared_core /= len(paths[i * (end_year + 1 -start_year - timeslice_thickness - j)  + j]) - 3
            print(args.number_of_sol_arg - number_of_solutions, ", ", len(paths[i * (end_year + 1 -start_year - timeslice_thickness - j)  + j]) - 2, ", ", average_edge_den_core, ", ", average_size_core, ", ", average_embed_core, ", ", average_shared_core, file = output_csv_core)
            json_out = open(output_path + "output" + str(testset) +  str(timeslice_thickness) + str(args.number_of_sol_arg) + str(target_fun) + str(community_alg) + str(args.number_of_sol_arg - number_of_solutions) + ".json", 'w')
            colors_out = open(output_path + "outputColors" + str(testset) +  str(timeslice_thickness) + str(args.number_of_sol_arg) + str(target_fun) + str(community_alg) + str(args.number_of_sol_arg - number_of_solutions) + ".json", 'w')
            print("Remaining nodes after postprocessing: ", output_graph_updated.nodes(), file=output)
            data = json_graph.adjacency_data(output_graph)
            s = json.dumps(data)
            print(s, file = json_out)
            color_map = list(output_graph_updated)
            c = json.dumps(color_map)
            print(c, file = colors_out)


            #setup for next pass
            number_of_solutions -= 1
            if number_of_solutions == 0:
                break
            counter = 0
            for idx, node in enumerate(enumerate_obj): # check every community on path, remove all communities that contain used authors from comparison graph
                if counter > 0:
                    comm = id_to_community[node[0]][node[1]]
                    for author in id_to_community[node[0]][node[1]]: # author = single author node in community
                        for check_comm in id_to_community[node[0]]:
                            if author in check_comm and (node[0], id_to_community[node[0]].index(check_comm)) in comparison_graph:
                                comparison_graph.remove_node((node[0], id_to_community[node[0]].index(check_comm)))

                if counter >= len(paths[i * (end_year + 1 -start_year - timeslice_thickness - j)  + j]) - 2:
                    break
                counter = counter + 1
        if args.target_function < 10 or target_fun == 4: #hier anzahl zielfunktionen
            break
        else:
            target_fun += 1


def compare_communities(first, second):
    return len(set(first) & set(second))/len(set(first) | set(second))

if __name__ == "__main__":
    main()