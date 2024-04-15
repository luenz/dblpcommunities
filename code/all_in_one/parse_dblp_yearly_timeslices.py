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
from math import log
import json
from operator import itemgetter
from scan import SCAN_nx
import community_louvain
from collections import defaultdict
from io import TextIOWrapper


"""A script to read a dlbp.xml file and find time-continuous communities within the induced temporal graph.
Outputs one up to several csv, txt and json files which contain information about the communities and can
be used to visualize said communities. 
The .csv files contain solution index, lifetime in time steps, 
average size of single time step communities and their standard deviation, average Jaccard coefficient
of adjacent single time step communities and their standard deviation, average internal edge density of 
single time step communities and their standard deviation, average average embeddedness of single time
step communities and their standard deviation for all discovered time-continuous communities. 

The .txt files contain the included authors and actual year for each time step for every time-continuous community,
as well as information about which author are part of the core community if applicable as well as some numbers used for 
comparison.

The .json files contain the graph representations of the aggregated communities (or in the case of the "communities.json"
file, a list of community and in the case of the "[postprocessing].json" files the remaining authors after postprocessing)
for each individual time-continuous community that was found, indicated by the number at the end of the file name.
"""

timing = False


def main():
    """The main function which reads the xml file, finds the time-continuous communities and outputs the output files.
    Parameters
    ----------
    testset_arg: int
        The index of the used dataset
    thickness_arg: int
        The intended timestep thickness
    number_of_sol_arg: int
        The number of time-continous communities to be found
    distance_function: int
        The index of the distance function to be used for the comparison graph. 10 for all, subsequently
    comm_alg: int
        The index of the static community algorithms to be used. 1 for Louvain, 2 for Chinese Whispers, 3 for SCD,
        4 for SCAN and 10 for all
    enex_mult: str
        The multiplier for late entry/early exit-edges in the comparison graph. 0.5 is a good starting point.
    preprocessing_type: int
        The index of the desired preprocessing type. 0 for none, >0 for reduced edge weight difference
    postprocessing_type: int
        The index of the desired postprocessing scheme. 0 for standard, 1 for avg_deg, 2 for avg_deg_weaker, 3 for light.
        10 if all are to be used sequentially.
    removal_option: int
        The index of the node removal scheme for the comparison graph. 0 for remove_core, 1 for remove_all, 2 for 
        remove_3_or_more, 3 for remove_half_or_more
    xml_path: str
        The path to the dblp.xml file. Note that dblp.dtd must be in the directory from which this program is run.
    -t, --timing: bool
        Parameter used to decide if the run is to be timed."""
    parser = argparse.ArgumentParser()
    parser.add_argument('testset_arg', type=int)
    parser.add_argument('thickness_arg', type=int)
    parser.add_argument('number_of_sol_arg', type=int)
    # 10 to use all distance functions
    parser.add_argument('distance_function', type=int)
    # 10 to use all community discovery algorithms
    parser.add_argument('comm_alg', type=int)
    # multiplier for length of late entry and early exit edges in comparison graph
    parser.add_argument('enex_mult', type=str)
    # 0 for none, >0 for closer edge weights
    parser.add_argument('preprocessing_type', type=int)
    # 0 for standard, 1 for avg_degree, 2 for avg_deg_weaker, 3 for light
    parser.add_argument('postprocessing_type', type=int)
    parser.add_argument('removal_option', type=int)
    parser.add_argument('xml_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('-t', '--timing', action='store_true')
    args = parser.parse_args()
    timing = args.timing
    if args.thickness_arg == 0:
        print("No graph creation possible with timestep thickness 0")
        sys.exit()
    postprocessing_type_max = 4
    if len(sys.argv) > 1:
        testset = args.testset_arg
        timestep_thickness = args.thickness_arg
        number_of_solutions = args.number_of_sol_arg
        if args.distance_function < 10:
            distance_fun = args.distance_function
        else:
            distance_fun = 0
        community_alg = args.comm_alg
        enex_mult = float(args.enex_mult)
        if args.postprocessing_type < 10:
            postprocessing_type = args.postprocessing_type
        else:
            postprocessing_type = 0
    else:
        testset = 1  # 1, 2 Atzmüller; 3, 4 Chimani
        timestep_thickness = 3
        number_of_solutions = 1
        distance_fun = 0
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
    file_identifier = (str(testset) + str(timestep_thickness) + 
        str(number_of_solutions))
    if timing:
        start = datetime.now()

    collab_graphs, authors_yearly, author_lst, author_conferences, start_year, end_year = xml_to_timestep_graphs(
        testset, timestep_thickness, args.preprocessing_type, dblp_path)

    if timing:
        end = datetime.now()
        print("Parsing xml and building collab graphs: ", end-start)
        print("Start_year: " + str(start_year))
        print("End_year: " + str(end_year))
        start = datetime.now()

    communities = []
    communities_louvain = []
    communities_cw = []
    communities_scd = []
    communities_scan = []
    community_origins = []
    for y in range(start_year, end_year + 1 - timestep_thickness + 1):
        # put different communities in
        community_origins.append([])
        communities.append([])
        if len(collab_graphs[y]) > 0:
            if community_alg == 10:
                community_origins.append([])
            if community_alg == 0 or community_alg == 10:
                for i in range(0, 10):
                    comms_louvain = community_louvain.best_partition(
                        collab_graphs[y].copy(), weight='weight', resolution=0.25*(i + 1), random_state=i)
                    coms_to_node = defaultdict(list)
                    for n, c in comms_louvain.items():
                        coms_to_node[c].append(n)
                    if community_alg == 10:
                        communities_louvain.extend(
                            list((commu) for commu in coms_to_node.values()))

                    communities[y - start_year].extend([list(commu)
                        for commu in coms_to_node.values()])
            if community_alg == 1 or community_alg == 10:
                for i in range(0, 10):
                    communities_local = algorithms.chinesewhispers(
                        collab_graphs[y].copy(), iterations=2*(i + 1), seed=i).communities
                    if community_alg == 10:
                        communities_cw.extend(communities_local)
                    communities[y - start_year].extend(communities_local)
            if community_alg == 2 or community_alg == 10:
                for i in range(0, 10):
                    communities_local = algorithms.scd(
                        collab_graphs[y].copy(), iterations=3*(i+1), seed=i).communities
                    if community_alg == 10:
                        communities_scd.extend(communities_local)
                    communities[y - start_year].extend(communities_local)
            if community_alg == 3 or community_alg == 10:
                for i in range(0, 10):
                    scan_init = SCAN_nx(collab_graphs[y].copy(
                    ), epsilon=0.5+(i*0.05), mu=2+(i*0.2), seed=i)
                    communities_local = scan_init.execute()
                    if community_alg == 10:
                        communities_scan.extend(communities_local)
                    communities[y - start_year].extend(communities_local)

            no_small_communities = []
            # remove duplicates and very small single time step communities
            for c in communities[y - start_year]:
                if not len(c) < 3:
                    c.sort()
                    no_small_communities.append(c)
            communities[y - start_year] = no_small_communities
            communities[y - start_year].sort()
            communities[y - start_year] = list(communities[y - start_year]
                for communities[y - start_year], _ in IT.groupby(communities[y - start_year]))
            if community_alg == 10:
                for c in communities_louvain:
                    c.sort()
                for c in communities_cw:
                    c.sort()
                for c in communities_scd:
                    c.sort()
                for c in communities_scan:
                    c.sort()
                for c in communities[y - start_year]:
                    community_origin_local = []
                    if c in communities_louvain:
                        community_origin_local.append(0)
                    if c in communities_cw:
                        community_origin_local.append(1)
                    if c in communities_scd:
                        community_origin_local.append(2)
                    if c in communities_scan:
                        community_origin_local.append(3)
                    community_origins[y - start_year].append(community_origin_local)

    if timing:
        end = datetime.now()
        print("Finding communities: ", end-start)
    # get reference for individual communities to store in comparison graph
    id_to_community = [] 
    community_nodes = []
    for y in range(start_year, end_year + 1 - timestep_thickness + 1):
        id_to_community.append([])
        community_nodes.append([])
        for c in communities[y - start_year]:
            nodes = set()
            for node in c:
                nodes.add(authors_yearly[y][node])
            id_to_community[y - start_year].append(c)
            community_nodes[y - start_year].append(nodes)

    community_calcs = []  # add list of community, embeddedness, density
    for y in range(start_year, end_year + 1 - timestep_thickness + 1):
        community_calcs.append([])
        for c_index in range(len(communities[y - start_year])):
            community_calcs[y - start_year].append([])
            clustering = NodeClustering(
                [id_to_community[y - start_year][c_index]], graph=collab_graphs[y])
            community_calcs[y - start_year][c_index].append(
                evaluation.avg_embeddedness(collab_graphs[y], clustering, summary=False)[0])
            community_calcs[y - start_year][c_index].append(
                evaluation.internal_edge_density(collab_graphs[y], clustering, summary=False)[0])

    # loop through all distance functions for fewer necessary executions of the program if multiple distance functions are to be tested
    while args.distance_function >= 0:
        number_of_solutions = args.number_of_sol_arg
        output = open(output_path + "outputSingleSourceTarget" + file_identifier + str(distance_fun) + str(community_alg) + str(enex_mult) + ".txt", 'w')

        if timing:
            start = datetime.now()

        comparison_graph = build_comparison_graph(communities, start_year, end_year,
            timestep_thickness, enex_mult, id_to_community, authors_yearly, community_calcs, distance_fun)

        if timing:
            end = datetime.now()
            print("Building comparison graph for distance function ",
                distance_fun, ": ", end-start)
        num_communities = 0
        for y in range(len(communities)):
            num_communities = num_communities + len(communities[y])
        print("Total number of communities: ", num_communities, file=output)

        output_csv = open(output_path + "outputSingleSourceTarget" + file_identifier + str(distance_fun) + str(community_alg) + str(enex_mult) + ".csv", 'w')
        while number_of_solutions > 0:
            # find shortest path in comparison graph to find most similar communities in consecutive years
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
                print("No path left", file=output, flush=True)
                print("", file=output_csv, flush=True)
                break
            if timing:
                start = datetime.now()
            path = nx.astar_path(comparison_graph, "start", "end")
            if timing:
                end = datetime.now()
                print(distance_fun, args.number_of_sol_arg -
                    number_of_solutions, " Finding shortest path: ", end-start)
                start = datetime.now()
            for idx, n in enumerate(path):
                print(n, file=output, flush=True)
                conferences_in_comm = set()
                if counter > 0:
                    print([*itemgetter(*itemgetter(*id_to_community[n[0] - start_year][n[1]])
                        (authors_yearly[n[0]]))(author_lst)], file=output, flush=True)
                    calc_clustering = NodeClustering(
                        [id_to_community[n[0] - start_year][n[1]]], graph=collab_graphs[n[0]], method_name="egal")
                    sizes.append(len(calc_clustering.communities[0]))
                    edge_den_old = average_edge_den
                    print(evaluation.internal_edge_density(calc_clustering.graph, calc_clustering, summary=False)[0],
                        evaluation.avg_embeddedness(calc_clustering.graph, calc_clustering, summary=False)[0], sep=',', file=output)
                    edge_dens.append(evaluation.internal_edge_density(
                        calc_clustering.graph, calc_clustering, summary=False)[0])
                    embeds.append(evaluation.avg_embeddedness(
                        calc_clustering.graph, calc_clustering, summary=False)[0])
                    for author in id_to_community[n[0] - start_year][n[1]]:
                        conferences_in_comm = conferences_in_comm.union(
                            author_conferences[author_lst[authors_yearly[n[0]][author]]])
                    print(conferences_in_comm, file=output, flush=True)
                    if community_alg == 10:
                        print(community_origins[n[0]-start_year]
                            [n[1]], file=output, flush=True)
                    if counter >= len(path) - 2:
                        break
                    else:
                        next_item = path[idx + 1]
                        shareds.append(compare_communities([*itemgetter(*id_to_community[n[0] - start_year][n[1]])(authors_yearly[n[0]])], [
                                       *itemgetter(*id_to_community[next_item[0] - start_year][next_item[1]])(authors_yearly[next_item[0]])]))
                        print(compare_communities([*itemgetter(*id_to_community[n[0] - start_year][n[1]])(authors_yearly[n[0]])], [
                              *itemgetter(*id_to_community[next_item[0] - start_year][next_item[1]])(authors_yearly[next_item[0]])]), file=output)
                        print(comparison_graph.get_edge_data(
                            n, path[idx + 1]), file=output, flush=True)
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
                # only count year transitions
                average_shared = np.mean(shareds)
                shared_sd = np.std(shareds)
            print(args.number_of_sol_arg - number_of_solutions, ", ", len(path) - 2, ", ", average_size, ", ", size_sd, ", ", average_shared,
                ", ", shared_sd, ", ", average_edge_den, ", ", edge_den_sd, ", ", average_embed, ", ", embed_sd, file=output_csv)
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
                    # add nodes from community chain to "core community"
                    core_community = set(comm)
                    comms_only_subgraph = nx.subgraph(
                        collab_graphs[n[0]], core_community)
                    # get average degree inisde community
                    print(len(comms_only_subgraph.edges)*2 /
                        len(comms_only_subgraph.nodes), file=output)

                    for node in comms_only_subgraph.nodes():
                        output_graph_updated.add_node(
                            author_lst[authors_yearly[n[0]][node]])
                    for edge in comms_only_subgraph.edges():
                        if output_graph_updated.has_edge(author_lst[authors_yearly[n[0]][edge[0]]], author_lst[authors_yearly[n[0]][edge[1]]]):
                            output_graph_updated[author_lst[authors_yearly[n[0]][edge[0]]]][author_lst[authors_yearly[n[0]][edge[1]]]]['weight'] = output_graph_updated[author_lst[authors_yearly[n[0]]
                                [edge[0]]]][author_lst[authors_yearly[n[0]][edge[1]]]]['weight'] + comms_only_subgraph[edge[0]][edge[1]]['weight']  # /timestep_thickness
                        else:
                            output_graph_updated.add_edge(author_lst[authors_yearly[n[0]][edge[0]]], author_lst[authors_yearly[n[0]]
                                [edge[1]]], weight=comms_only_subgraph[edge[0]][edge[1]]['weight']) 
                    for node in comm:
                        color_map.append(
                            author_lst[authors_yearly[n[0]][node]])
                        add_nodes = add_nodes.union(
                            set(collab_graphs[n[0]].neighbors(node)))
                    subgraph = nx.subgraph(collab_graphs[n[0]], add_nodes)
                    for node in subgraph.nodes():
                        output_graph.add_node(
                            author_lst[authors_yearly[n[0]][node]])
                    for edge in subgraph.edges():
                        if output_graph.has_edge(author_lst[authors_yearly[n[0]][edge[0]]], author_lst[authors_yearly[n[0]][edge[1]]]):
                            output_graph[author_lst[authors_yearly[n[0]][edge[0]]]][author_lst[authors_yearly[n[0]][edge[1]]]]['weight'] = output_graph[author_lst[authors_yearly[n[0]]
                                [edge[0]]]][author_lst[authors_yearly[n[0]][edge[1]]]]['weight'] + subgraph[edge[0]][edge[1]]['weight']  # /timestep_thickness
                        else:
                            output_graph.add_edge(author_lst[authors_yearly[n[0]][edge[0]]], author_lst[authors_yearly[n[0]]
                                [edge[1]]], weight=subgraph[edge[0]][edge[1]]['weight'])
                if counter >= len(path) - 2:
                    break
                counter = counter + 1
            if timing:
                end = datetime.now()
                print(distance_fun, args.number_of_sol_arg-number_of_solutions,
                    " Creating output graph and csv content and writing communities in path to txt: ", end-start)
            before_postprocessing = output_graph_updated.copy()

            while postprocessing_type_updated < postprocessing_type_max:
                output_graph_updated = before_postprocessing.copy()
                
                handle_postprocessing(postprocessing_type_updated, timestep_thickness, output_graph_updated, path, author_lst,
                                    authors_yearly, id_to_community, collab_graphs, start_year, 
                                    file_identifier + str(distance_fun) + str(community_alg) + str(enex_mult), output_path, output, args.number_of_sol_arg, 
                                    number_of_solutions, distance_fun, community_alg, enex_mult)

                if args.postprocessing_type < 10:
                    break
                postprocessing_type_updated += 1

            json_out = open(output_path + "outputSingleSourceTarget" + file_identifier + str(distance_fun) + str(community_alg) + str(enex_mult) + str(args.number_of_sol_arg - number_of_solutions) + ".json", 'w')
            data = json_graph.adjacency_data(output_graph)
            s = json.dumps(data)
            print(s, file=json_out)

            communities_out = open(output_path + "outputSingleSourceTargetCommunities" + file_identifier + str(distance_fun) + str(community_alg) + str(enex_mult) + str(args.number_of_sol_arg - number_of_solutions) + ".json", 'w')
            print(json.dumps(list(before_postprocessing)), file=communities_out)
            # setup for next pass
            number_of_solutions -= 1
            if number_of_solutions == 0:
                break
            remove_used_from_comparison_graph(args.removal_option, path, output_graph_updated, start_year, 
                                            timestep_thickness, id_to_community, authors_yearly, author_lst, comparison_graph)
        if args.distance_function < 10 or distance_fun == 4:  # index of last target function
            break
        else:
            distance_fun += 1


def compare_communities(first, second):
    """Compares the contents of two containers and returns their Jaccard coefficient."""
    result = len(set(first) & set(second))/len(set(first) | set(second))
    return result


def xml_to_timestep_graphs(dataset: int, timestep_thickness: int, preprocessing_type: int, xml_path: str):
    """Reads the dblp.xml and creates a collaboration graph for each timestep.
    
    Parameters
    ----------
    dataset: int
        An int describing what dataset to use
    timestep_thickness: int
        How many years each timestep contains
    preprocessing_type: int
        Whether the graph should include preprocessing to reduce difference between edge weights
    xml_path: the path to dblp.xml
    Returns
    -------
    The collaboration graph for each timestep, a dictionary containing all authors per year, a list of all authors,
    a dictionary describing which authors take part in which conferences, the year the first time step starts and the 
    year the last time step starts.
    """
    file = open(xml_path, 'rb')
    xml_file = BytesIO(file.read())
    author_lst = []
    collab_graphs = {}
    authors_yearly = {}
    author_conferences = {}
    author_lst_local = []
    flag = 0
    start_year = 0
    end_year = 0
    for event, elem in ET.iterparse(xml_file, dtd_validation=True):
        if elem.tag == "inproceedings":
            if dataset == 1:
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
            if dataset == 2:
                if "conf/kdd/" in elem.attrib['key']:
                    flag = 'kdd'
                if "conf/icdm/" in elem.attrib['key']:
                    flag = 'icdm'
                if "conf/icml/" in elem.attrib['key']:
                    flag = 'icml'
            if dataset == 3 or dataset == 4:
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
            if dataset == 4:
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

            for year in range(int(elem.text) - timestep_thickness + 1, int(elem.text) + 1):
                if not year in collab_graphs:
                    collab_graphs[year] = nx.Graph()
                    authors_yearly[year] = []
                if len(author_lst_local) > 1:
                    for a in author_lst_local:
                        # choose proper graph by year
                        # save list of all authors for faster comparisons (index vs string)
                        if a not in author_lst:
                            author_lst.append(a)
                        a_index = author_lst.index(a)
                        if a_index not in authors_yearly[year]:
                            authors_yearly[year].append(a_index)
                        collab_graphs[year].add_node(
                            authors_yearly[year].index(a_index))
                    for combo in IT.combinations(author_lst_local, 2):
                        # same graph as above
                        if collab_graphs[year].has_edge(authors_yearly[year].index(author_lst.index(combo[0])), authors_yearly[year].index(author_lst.index(combo[1]))):
                            collab_graphs[year][authors_yearly[year].index(author_lst.index(combo[0]))][authors_yearly[year].index(author_lst.index(
                                combo[1]))]['weight'] = collab_graphs[year][authors_yearly[year].index(author_lst.index(combo[0]))][authors_yearly[year].index(author_lst.index(combo[1]))]['weight'] + 1
                        else:
                            collab_graphs[year].add_edge(authors_yearly[year].index(author_lst.index(
                                combo[0])), authors_yearly[year].index(author_lst.index(combo[1])), weight=1)
            author_lst_local = []
            flag = 0
        elem.clear()
    collab_graphs = {key: collab_graphs[key] for key in sorted(collab_graphs)}
    if preprocessing_type > 0:
        for y in collab_graphs:
            for u, v, d in collab_graphs[y].edges(data=True):
                d['weight'] = 1 + d['weight']*0.1
    for y in range(start_year, end_year + 1 - timestep_thickness + 1):
        try:
            x = collab_graphs[y]
        except KeyError:
            collab_graphs[y] = nx.Graph()
    return collab_graphs, authors_yearly, author_lst, author_conferences, start_year, end_year


def build_comparison_graph(communities: list, start_year: int, end_year: int, timestep_thickness: int, enex_mult: float,
                        id_to_community: list, authors_yearly: dict, community_calcs: list, distance_fun: int):
    """Builds the comparison graph.
    
    Parameters
    ----------
    communities: list
        A list of all communities to be turned into nodes
    start_year: int
        The year the first timestep starts
    end_year: int
        The year the last timestep starts
    timestep_thickness: int
        How many years each timestep contains
    enex_mult: float
        A multiplier for the length of late entry/early exit-edges
    id_to_community: list
        A list connecting communities to their global ids
    authors_yearly: dict
        A dictionary containing the authors for each individual timestep
    community_calcs: list
        A nested list containing stats for each community
    distance_fun: int
        The index for the used distance function
    Reruns
    ------
    The finished comparison graph"""
    comparison_graph = nx.DiGraph()
    i = 0
    comm_counter = 0
    epsilon = 0.001  # make sure we never divide by 0
    # fill comparison graph
    for y in range(start_year, end_year + 1 - timestep_thickness + 1):
        if i == 0:
            comparison_graph.add_node("start")
            comparison_graph.add_node("end")

            for c_index in range(len(communities[y - start_year])):
                comparison_graph.add_node((y, c_index))
                comparison_graph.add_edge(
                    "start",  (y, c_index), weight=(y - start_year)*enex_mult)
                comparison_graph.add_edge((y, c_index), "end", weight=(
                    end_year + 1 - timestep_thickness - y)*enex_mult)
                comm_counter = comm_counter + 1
        else:
            for c_index in range(len(communities[y - start_year])):
                if (y, c_index) in comparison_graph:
                    comm_counter = comm_counter + 1
                    continue
                comparison_graph.add_node((y, c_index))
                comparison_graph.add_edge(
                    "start",  (y, c_index), weight=(y - start_year)*enex_mult)
                comparison_graph.add_edge((y, c_index), "end", weight=(
                    end_year + 1 - timestep_thickness - y)*enex_mult)
                comm_counter = comm_counter + 1
                for c2_index in range(len(communities[previous_year - start_year])):
                    base_weight = compare_communities([*itemgetter(*id_to_community[y - start_year][c_index])(authors_yearly[y])], [
                                                      *itemgetter(*id_to_community[previous_year - start_year][c2_index])(authors_yearly[previous_year])])
                    if base_weight > epsilon and distance_fun == 0:
                        if base_weight < 0.5:
                            base_weight = 100000
                        else:
                            base_weight = 1/((log(len(id_to_community[y - start_year][c_index])) + log((len(id_to_community[previous_year - start_year][c2_index]))))/2*20*(base_weight * (
                                (community_calcs[y - start_year][c_index][0] + community_calcs[previous_year - start_year][c2_index][0])/2) * ((community_calcs[y - start_year][c_index][1] + community_calcs[previous_year - start_year][c2_index][1])/2)))
                    elif base_weight > epsilon and distance_fun == 1:
                        if base_weight < 0.5:
                            base_weight = 100000
                        else:
                            base_weight = 1-(base_weight * ((community_calcs[y - start_year][c_index][0] + community_calcs[previous_year - start_year][c2_index][0])/2) * (
                                (community_calcs[y - start_year][c_index][1] + community_calcs[previous_year - start_year][c2_index][1])/2))
                    elif base_weight > epsilon and distance_fun == 2:
                        if base_weight < 0.5:
                            base_weight = 100000
                        else:
                            base_weight = 1/(((log(len(id_to_community[y - start_year][c_index])) + log(
                                (len(id_to_community[previous_year - start_year][c2_index]))))/2)**2*base_weight)
                    elif base_weight > epsilon and distance_fun == 3:
                        if base_weight < 0.5:
                            base_weight = 100000
                        else:
                            base_weight = (1-base_weight)
                    elif base_weight > epsilon and distance_fun == 4:
                        if base_weight < 0.5:
                            base_weight = 100000
                        else:
                            base_weight = 1/(((log(len(id_to_community[y - start_year][c_index])) + log(
                                (len(id_to_community[previous_year - start_year][c2_index]))))/2)**2*base_weight*2)
                    else:
                        base_weight = 100000
                    if base_weight < (end_year + 1 - start_year)*enex_mult:
                        comparison_graph.add_edge(
                            (previous_year, c2_index), (y, c_index), weight=base_weight)
        previous_year = y
        i = i + 1
    return comparison_graph

def handle_postprocessing(postprocessing_type_updated: int, timestep_thickness: int, output_graph_updated: nx.Graph, path: dict, author_lst: list, authors_yearly: dict,
                        id_to_community: list, collab_graphs: dict, start_year: int, file_identifier: str, output_path: str,
                        output_file: TextIOWrapper, total_sol_num: int, current_sol_num: int, distance_fun: int,
                        community_alg: int, enex_mult: float):
    """A function applying the desired postprocessing to the found community path (time-continuous community)
    Parameters
    ----------
    postprocessing_type_updated: int
        The index for the desired postprocessing type
    timestep_thickness: int
        The number of years contained in each timestep
    output_graph_updated: nx.Graph
        A copy of the output graph to be used in postprocessing
    path: dict
        The solution path (time-continuous community)
    id_to_community: list
        A list connecting communities to their global ids
    start_year: int
        The year the first timestep starts
    file_identifier: str
        The identifier used for the output file names
    output_path: str
        The name of the output folder
    output_file: TextIOWrapper
        The output text file to write information about the community after postprocessing into
    total_sol_num: int 
        The total number of time-continuous communities to be found
    current_sol_num: int
        A number representing which of the found time-continuous communities is currently being processed
    distance_fun: int
        The index of the used distance function
    community_alg: int
        The index of the used community detection algorithm(s)
    enex_mult: float
        The used late entry/early exit edge weight multiplier
    Returns
    -------
    None"""
    
    if postprocessing_type_updated == 0:
        # initially remove authors which are not present sufficiently often (factor subject to change) !!! might split community !!!
        remove_nodes = []
        for node in output_graph_updated:
            max_edge_weight = 0
            for edge in output_graph_updated.edges(node):
                if output_graph_updated.get_edge_data(edge[0], edge[1])['weight'] > max_edge_weight:
                    max_edge_weight = output_graph_updated.get_edge_data(edge[0], edge[1])[
                        'weight']
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
                            author_indices.append(
                                authors_yearly[n[0]].index(author))
                            authors_current.append(author)
                if len(author_indices) > 0:
                    core_comm_clustering = NodeClustering(
                        [author_indices], collab_graphs[n[0]], "core_community")
                    edge_dens_core.append(evaluation.internal_edge_density(
                        core_comm_clustering.graph, core_comm_clustering, summary=False)[0])
                    sizes_core.append(len(author_indices))
                    embeds_core.append(evaluation.avg_embeddedness(
                        core_comm_clustering.graph, core_comm_clustering, summary=False)[0])
                    if idx > 1:
                        shareds_core.append(compare_communities(
                            authors_current, authors_previous))
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
        output_csv_core = open(output_path + "outputSingleSourceTargetSTD" + file_identifier + str(distance_fun) + str(community_alg) + str(enex_mult) + ".csv", 'a')
        print(total_sol_num - current_sol_num, ", ", len(path) - 2, ", ", average_size_core, ", ", size_sd_core, ", ", average_shared_core,
            ", ", shared_sd_core, ", ", average_edge_den_core, ", ", edge_den_sd_core, ", ", average_embed_core, ", ", embed_sd_core, file=output_csv_core)
        postprocessed_out = open(output_path + "outputSingleSourceTargetSTD" + file_identifier + str(distance_fun) + str(community_alg) + str(enex_mult) + str(total_sol_num - current_sol_num) + ".json", 'w')
        print("Remaining nodes after postprocessing: ",
            output_graph_updated.nodes(), file=output_file)
        color_map = list(output_graph_updated)
        c = json.dumps(color_map)
        print(c, file=postprocessed_out)

    if postprocessing_type_updated == 1:

        # initially remove authors which are not present sufficiently often (factor subject to change) !!! might split community !!!
        remove_nodes = []
        average_degree = 0  # average_degree = 2*|edges|/|vertices|, not very time intensive so kept like this for the time being
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
                            author_indices.append(
                                authors_yearly[n[0]].index(author))
                            authors_current.append(author)
                if len(author_indices) > 0:
                    core_comm_clustering = NodeClustering(
                        [author_indices], collab_graphs[n[0]], "core_community")
                    edge_dens_core.append(evaluation.internal_edge_density(
                        core_comm_clustering.graph, core_comm_clustering, summary=False)[0])
                    sizes_core.append(len(author_indices))
                    embeds_core.append(evaluation.avg_embeddedness(
                        core_comm_clustering.graph, core_comm_clustering, summary=False)[0])
                    if idx > 1:
                        shareds_core.append(compare_communities(
                            authors_current, authors_previous))
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
        output_csv_core = open(output_path + "outputSingleSourceTargetAVD" + file_identifier + str(distance_fun) + str(community_alg) + str(enex_mult) + ".csv", 'a')
        print(total_sol_num - current_sol_num, ", ", len(path) - 2, ", ", average_size_core, ", ", size_sd_core, ", ", average_shared_core,
            ", ", shared_sd_core, ", ", average_edge_den_core, ", ", edge_den_sd_core, ", ", average_embed_core, ", ", embed_sd_core, file=output_csv_core)
        postprocessed_out = open(output_path + "outputSingleSourceTargetAVD" + file_identifier + str(distance_fun) + str(community_alg) + str(enex_mult) + str(total_sol_num - current_sol_num) + ".json", 'w')
        print("Remaining nodes after postprocessing: ",
            output_graph_updated.nodes(), file=output_file)
        color_map = list(output_graph_updated)
        c = json.dumps(color_map)
        print(c, file=postprocessed_out)

    if postprocessing_type_updated == 2:

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
                            author_indices.append(
                                authors_yearly[n[0]].index(author))
                            authors_current.append(author)
                if len(author_indices) > 0:
                    core_comm_clustering = NodeClustering(
                        [author_indices], collab_graphs[n[0]], "core_community")
                    edge_dens_core.append(evaluation.internal_edge_density(
                        core_comm_clustering.graph, core_comm_clustering, summary=False)[0])
                    sizes_core.append(len(author_indices))
                    embeds_core.append(evaluation.avg_embeddedness(
                        core_comm_clustering.graph, core_comm_clustering, summary=False)[0])
                    if idx > 1:
                        shareds_core.append(compare_communities(
                            authors_current, authors_previous))
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
        output_csv_core = open(output_path + "outputSingleSourceTargetAVDWeaker" + file_identifier + str(distance_fun) + str(community_alg) + str(enex_mult) + ".csv", 'a')
        print(total_sol_num - current_sol_num, ", ", len(path) - 2, ", ", average_size_core, ", ", size_sd_core, ", ", average_shared_core,
            ", ", shared_sd_core, ", ", average_edge_den_core, ", ", edge_den_sd_core, ", ", average_embed_core, ", ", embed_sd_core, file=output_csv_core)
        postprocessed_out = open(output_path + "outputSingleSourceTargetAVDWeaker" + file_identifier + str(distance_fun) + str(community_alg) + str(enex_mult) + str(total_sol_num - current_sol_num) + ".json", 'w')
        print("Remaining nodes after postprocessing: ",
            output_graph_updated.nodes(), file=output_file)
        color_map = list(output_graph_updated)
        c = json.dumps(color_map)
        print(c, file=postprocessed_out)

    if postprocessing_type_updated == 3:

        # initially remove authors which are not present sufficiently often (factor subject to change) !!! might split community !!!
        remove_nodes = []
        for node in output_graph_updated:
            occurences = 0
            for idx, n in enumerate(path):
                if idx > 0 and idx < len(path) - 1:
                    if node in [*itemgetter(*itemgetter(*id_to_community[n[0] - start_year][n[1]])(authors_yearly[n[0]]))(author_lst)]:
                        occurences += 1
            if occurences <= timestep_thickness:
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
                            author_indices.append(
                                authors_yearly[n[0]].index(author))
                            authors_current.append(author)
                if len(author_indices) > 0:
                    core_comm_clustering = NodeClustering(
                        [author_indices], collab_graphs[n[0]], "core_community")
                    edge_dens_core.append(evaluation.internal_edge_density(
                        core_comm_clustering.graph, core_comm_clustering, summary=False)[0])
                    sizes_core.append(len(author_indices))
                    embeds_core.append(evaluation.avg_embeddedness(
                        core_comm_clustering.graph, core_comm_clustering, summary=False)[0])
                    if idx > 1:
                        shareds_core.append(compare_communities(
                            authors_current, authors_previous))
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
        output_csv_core = open(output_path + "outputSingleSourceTargetLight" + file_identifier + str(distance_fun) + str(community_alg) + str(enex_mult) + ".csv", 'a')
        print(total_sol_num - current_sol_num, ", ", len(path) - 2, ", ", average_size_core, ", ", size_sd_core, ", ", average_shared_core,
            ", ", shared_sd_core, ", ", average_edge_den_core, ", ", edge_den_sd_core, ", ", average_embed_core, ", ", embed_sd_core, file=output_csv_core)
        postprocessed_out = open(output_path + "outputSingleSourceTargetLight" + file_identifier + str(distance_fun) + str(community_alg) + str(enex_mult) + str(total_sol_num - current_sol_num) + ".json", 'w')
        print("Remaining nodes after postprocessing: ", output_graph_updated.nodes(), file=output_file)
        color_map = list(output_graph_updated)
        c = json.dumps(color_map)
        print(c, file=postprocessed_out)
        
def remove_used_from_comparison_graph(removal_option: int, path: dict, output_graph_updated: nx.Graph, start_year: int,
                                    timestep_thickness: int, id_to_community: list, authors_yearly: dict, author_lst: list,
                                    comparison_graph: nx.Graph):
    """A function for removing (single time step) communities similar to the used ones from the comparison graph.
    Parameters
    ----------
    removal_option: int
        Which removal scheme is to be used
    path: dict
        The found solution path (time-continuous community)
    output_graph_updated: nx.Graph
        A graph representation of the postprocessing results
    start_year: int
        The year the first time step begins
    timestep_thickness: int
        The number of years contained in each time step
    id_to_community: list
        A list relating (single time step) communities to their global ids
    authors_yearly: dict
        A dictionary containing information on which authors were active in what timesteps
    author_lst: list
        A list relating authors to their global ids
    comparison_graph: nx.Graph
        The comparison graph to remove nodes from
    Returns
    -------
    None"""
    counter = 0
    core_too_small = False
    if removal_option == 1:
        if len(list(output_graph_updated.nodes)) > timestep_thickness:
            # check every community on path, remove all communities that contain used authors from comparison graph
            for idx, node in enumerate(path):
                if counter > 0:
                    # author = single author node in community
                    for author in [*itemgetter(*id_to_community[node[0] - start_year][node[1]])(authors_yearly[node[0]])]:
                        for check_comm in range(len(id_to_community[node[0] - start_year])):
                            if author in [*itemgetter(*id_to_community[node[0] - start_year][check_comm])(authors_yearly[node[0]])] and (node[0], check_comm) in comparison_graph:
                                if author_lst[author] in list(output_graph_updated.nodes):
                                    comparison_graph.remove_node(
                                        (node[0], check_comm))
                if counter >= len(path) - 2:
                    break
                counter = counter + 1
        else:
            core_too_small = True
    if removal_option == 0 or core_too_small:
        # check every community on path, remove all communities that contain used authors from comparison graph
        for idx, node in enumerate(path):
            if counter > 0:
                # author = single author node in community
                for author in [*itemgetter(*id_to_community[node[0] - start_year][node[1]])(authors_yearly[node[0]])]:
                    for check_comm in range(len(id_to_community[node[0] - start_year])):
                        if author in [*itemgetter(*id_to_community[node[0] - start_year][check_comm])(authors_yearly[node[0]])] and (node[0], check_comm) in comparison_graph:
                            comparison_graph.remove_node(
                                (node[0], check_comm))
            if counter >= len(path) - 2:
                break
            counter = counter + 1
    if removal_option == 2:
        # check every community on path, remove all communities that contain used authors from comparison graph
        for idx, node in enumerate(path):
            if counter > 0:
                for check_comm in range(len(id_to_community[node[0] - start_year])):
                    overlap = 0
                    # author = single author node in community
                    for author in [*itemgetter(*id_to_community[node[0] - start_year][node[1]])(authors_yearly[node[0]])]:
                        if author in [*itemgetter(*id_to_community[node[0] - start_year][check_comm])(authors_yearly[node[0]])]:
                            overlap = overlap + 1
                            if overlap > 2 and (node[0], check_comm) in comparison_graph:
                                comparison_graph.remove_node(
                                    (node[0], check_comm))
            if counter >= len(path) - 2:
                break
            counter = counter + 1
    if removal_option == 3:
        # check every community on path, remove all communities that contain used authors from comparison graph
        for idx, node in enumerate(path):
            if counter > 0:
                for check_comm in range(len(id_to_community[node[0] - start_year])):
                    overlap = 0
                    # author = single author node in community
                    for author in [*itemgetter(*id_to_community[node[0] - start_year][node[1]])(authors_yearly[node[0]])]:
                        if author in [*itemgetter(*id_to_community[node[0] - start_year][check_comm])(authors_yearly[node[0]])]:
                            overlap = overlap + 1
                            if overlap > len([*itemgetter(*id_to_community[node[0] - start_year][check_comm])(authors_yearly[node[0]])])/2 and (node[0], check_comm) in comparison_graph:
                                comparison_graph.remove_node(
                                    (node[0], check_comm))
            if counter >= len(path) - 2:
                break
            counter = counter + 1


if __name__ == "__main__":
    main()
