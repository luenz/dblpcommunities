import lxml.etree as ET
import networkx as nx
import itertools as IT
import os
from cdlib import algorithms, evaluation, NodeClustering
from datetime import datetime
import numpy as np
from io import BytesIO
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
    parser.add_argument('number_of_sol_arg', type=int)
    parser.add_argument('target_function', type=int) #10 für alle
    parser.add_argument('comm_alg', type=int) #10 for all
    parser.add_argument('cutoff', type=str)
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
        cutoff = float(args.cutoff)
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
    flag = True
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
        output = open(output_path + "outputSingleSourceTarget" + str(testset) +  str(timeslice_thickness) + str(number_of_solutions) + str(target_fun) + str(community_alg) + str(cutoff) + ".txt", 'w')

        comparison_graph = nx.DiGraph()
        i = 0
        per_year_buffer = []
        comm_counter = 0
        epsilon = 0.001
        # fill comparison graph
        for y in range(start_year, end_year + 1 - timeslice_thickness + 1):
            if i == 0:
                comparison_graph.add_node("start")
                comparison_graph.add_node("end")
                for c in communities[y]:
                    comparison_graph.add_node((y, id_to_community[y].index(c)))
                    comparison_graph.add_edge("start",  (y, id_to_community[y].index(c)), weight = (y - start_year + 1)*cutoff) #irgendwas mit y drin wahrscheinlich?
                    comm_counter = comm_counter + 1
            else:
                for c in communities[y]:
                    if (y, id_to_community[y].index(c)) in comparison_graph:
                        comm_counter = comm_counter + 1
                        continue
                    comparison_graph.add_node((y, id_to_community[y].index(c)))
                    comparison_graph.add_edge("start",  (y, id_to_community[y].index(c)), weight = (y - start_year + 1)*cutoff) #irgendwas mit y drin wahrscheinlich?
                    # if one huge difference between communities, they likely aren't the same one -> maybe flat values
                    clustering = NodeClustering([c], graph=collab_graphs[y], method_name="egal")
                    avg_embed = evaluation.avg_embeddedness(collab_graphs[y], clustering, summary=False)
                    edge_den = evaluation.internal_edge_density(collab_graphs[y], clustering, summary=False)
                    comm_counter = comm_counter + 1
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
                    comparison_graph.add_edge((y, id_to_community[y].index(c)), "end", weight = (end_year + 1 - timeslice_thickness - y)*cutoff)

        num_communities = 0
        for y in communities:
            num_communities = num_communities + len(communities[y])
        print("Total number of communities: ", num_communities, file = output)

        output_csv = open(output_path + "outputSingleSourceTarget" + str(testset) +  str(timeslice_thickness) + str(number_of_solutions) + str(target_fun) + str(community_alg) + str(cutoff) + ".csv", 'w')
        output_csv_core = open(output_path + "outputSingleSourceTargetCore" + str(testset) +  str(timeslice_thickness) + str(number_of_solutions) + str(target_fun) + str(community_alg) + str(cutoff) + ".csv", 'w')
        while number_of_solutions > 0:
            # find shortest path in comparison graph to find most similar communities in different years
            average_size = 0
            average_edge_den = 0
            average_embed = 0
            average_shared = 0
            counter = 0
            path = nx.astar_path(comparison_graph, "start", "end")
            for idx, n in enumerate(path):
                print(n, file = output)
                conferences_in_comm = set()
                if counter > 0:
                    print(community_nodes[n[0]][n[1]], file = output)
                    calc_clustering = NodeClustering([id_to_community[n[0]][n[1]]], graph=collab_graphs[n[0]], method_name="egal")
                    average_size += len(calc_clustering.communities[0])
                    edge_den_old = average_edge_den
                    average_edge_den += evaluation.internal_edge_density(calc_clustering.graph, calc_clustering, summary=False)[0]
                    average_embed += evaluation.avg_embeddedness(calc_clustering.graph, calc_clustering, summary=False)[0]
                    if counter >= len(path) - 2:
                        break
                    else:
                        next_item = path[idx + 1]
                        average_shared += compare_communities(community_nodes[n[0]][n[1]], community_nodes[next_item[0]][next_item[1]])
                        for author in community_nodes[n[0]][n[1]]:
                            conferences_in_comm = conferences_in_comm.union(author_conferences[author])
                        print(conferences_in_comm, file = output)
                        print(comparison_graph.get_edge_data(n, path[path.index(n) + 1]), file = output)
                counter = counter + 1
                if counter == len(path) - 1:
                    break
            average_size = average_size/(len(path) - 2)
            average_edge_den = average_edge_den/(len(path) - 2)
            average_embed = average_embed/(len(path) - 2)
            if len(path) > 3:
                average_shared /= (len(path) - 3) # only count year transitions
            print(args.number_of_sol_arg - number_of_solutions, ", ", len(path) - 2, ", ", average_edge_den, ", ", average_size, ", ", average_embed, ", ", average_shared, file = output_csv)
            # create output graph
            output_graph = nx.Graph()
            color_map = []
            core_community = set()
            output_graph_updated = nx.Graph()
            counter = 0
            for idx, n in enumerate(path):
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
                if counter >= len(path) - 2:
                        break
                counter = counter + 1

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

            # calculate fitness scores for core community
            average_edge_den_core = 0
            average_size_core = 0
            average_shared_core = 0
            average_embed_core = 0
            for idx, n in enumerate(path):
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
                if idx >= len(path) - 2:
                    break
            average_edge_den_core /= len(path) - 2
            average_size_core /= len(path) - 2
            average_embed_core /= len(path) - 2
            if len(path) > 3:
                average_shared_core /= len(path) - 3
            print(args.number_of_sol_arg - number_of_solutions, ", ", len(path) - 2, ", ", average_edge_den_core, ", ", average_size_core, ", ", average_embed_core, ", ", average_shared_core, file = output_csv_core)


            labels = nx.get_edge_attributes(output_graph, 'weight')
            nx.draw_networkx(output_graph) # , pos = nx.spring_layout(output_graph))
            nx.draw_networkx_edge_labels(output_graph, pos = nx.spring_layout(output_graph), edge_labels = labels)
            # matplotlib.pyplot.show(block=False)
            # matplotlib.pyplot.show()
            json_out = open(output_path + "outputSingleSourceTarget" + str(testset) +  str(timeslice_thickness) + str(args.number_of_sol_arg) + str(target_fun) + str(community_alg) + str(cutoff) + str(args.number_of_sol_arg - number_of_solutions) + ".json", 'w')
            colors_out = open(output_path + "outputSingleSourceTargetColors" + str(testset) +  str(timeslice_thickness) + str(args.number_of_sol_arg) + str(target_fun) + str(community_alg) + str(cutoff) + str(args.number_of_sol_arg - number_of_solutions) + ".json", 'w')
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
            for idx, node in enumerate(path): # check every community on path, remove all communities that contain used authors from comparison graph
                if counter > 0:
                    comm = id_to_community[node[0]][node[1]]
                    for author in id_to_community[node[0]][node[1]]: # author = single author node in community
                        for check_comm in id_to_community[node[0]]:
                            if author in check_comm and (node[0], id_to_community[node[0]].index(check_comm)) in comparison_graph:
                                comparison_graph.remove_node((node[0], id_to_community[node[0]].index(check_comm)))

                if counter >= len(path) - 2:
                    break
                counter = counter + 1
        if args.target_function < 10 or target_fun == 4: # hier anzahl zielfunktionen
            break
        else:
            target_fun += 1

def compare_communities(first, second):
    return len(set(first) & set(second))/len(set(first) | set(second))

if __name__ == "__main__":
    main()