import argparse
import sys
from datetime import datetime
import lxml.etree as ET
import itertools as IT
import os
import networkx as nx
from io import BytesIO
import json
from networkx.readwrite import json_graph
import matplotlib
import copy

import matplotlib.pyplot


def main():
    """Displays all single time steps as graphs using networkx and matplotlib. Neighbors óf the single time step communities
    are also included to see the connection to the entire graph.
    
    Parameters
    ----------
    testset_arg: int
        Index of the used data set
    thickness_arg: int
        Number of years per time step
    number_of_sol_arg: int
        Total number of solutions in txt file
    preprocessing_type: int
        Index of used preprocessing type
    xml_path: str
        Path to dblp.xml. Note dblp.dtd has to be present in the directory the program is ran from
    graph_path: str
        Path to .json file of graph desired to be displayed (outputSingleSourceTargetxyz.json)
    no_postprocessing_path : str
        Path to .json file of communities desired to be displayed (outputSingleSourceTargetCommunitiesxyz.json)
    postprocessing_path: str
        Path to .json file of core community if applicable (outputSingleSourceTarget[Postprocessing]xyz.json)
    txt_path: str
        Path to .txt file containing information about single time step communities
    sol_num: int
        Index of the desired time-continuous community"""
    parser = argparse.ArgumentParser()
    parser.add_argument('testset_arg', type=int)
    parser.add_argument('thickness_arg', type=int)
    parser.add_argument('number_of_sol_arg', type=int)
    parser.add_argument('preprocessing_type', type=int) # 0 for none, >0 for closer edge weights
    parser.add_argument('xml_path', type=str) # dblp.xml
    parser.add_argument('graph_path', type=str) # nx.Graph as json, "outputSingleSourceTargetxyz.json"
    parser.add_argument('no_postprocessing_path', type=str) # nx.Graph as json, "outputSingleSourceTargetCommunitiesxyz.json"
    parser.add_argument('postprocessing_path', type=str) # nx.Graph as json, "outputSingleSourceTarget[Postprocessing]xyz.json"
    parser.add_argument('txt_path', type=str) # information about solution path as txt
    parser.add_argument('sol_num', type=int) # which of the solutions in txt is used
    args = parser.parse_args()
    if len(sys.argv) > 1:
        testset = args.testset_arg
        timeslice_thickness = args.thickness_arg
    else:
        testset = 1 # 1, 2 Atzmüller; 3, 4 Chimani
        timeslice_thickness = 3
    dblp_path = args.xml_path
    if not os.path.exists(dblp_path):
        print("Error finding dblp.xml, did you enter the right path?")
        sys.exit()


    file = open(dblp_path, 'rb')
    xml_file = BytesIO(file.read())
    collab_graphs, authors_yearly, author_lst, author_conferences, start_year, end_year = xml_to_timestep_graphs(testset,
                                                                                                                timeslice_thickness,
                                                                                                                args.preprocessing_type,
                                                                                                                dblp_path)


    no_postprocessing = None
    postprocessing = None
    infile = open(args.graph_path)
    if os.path.isfile(args.no_postprocessing_path):
        no_postprocessing = open(args.no_postprocessing_path)
    if os.path.isfile(args.postprocessing_path):
        postprocessing = open(args.postprocessing_path)
    s = json.load(infile)
    print(infile)
    graph = json_graph.adjacency_graph(s)
    labels = nx.get_edge_attributes(graph, 'weight')
    print(labels)
    for key in labels.keys():
        labels[key] = round(labels[key], 2)
    pos = nx.spring_layout(graph, seed=0)
    print(graph.nodes())

    txt_file = open(args.txt_path, 'r')
    print(txt_file)
    
    data = txt_file.readlines()
    current_year = 0
    path_number = -1
    single_path_statistics = list()
    next = 0
    buffer = 0
    first_year = 0
    countdown = 0
    #extract relevant information from txt file
    for line in data:
        line = line.strip()
        if line == 'start':
            path_number += 1
            first_year = 0
            next = 'year'
            continue
        if next == 'year':
            current_year = int(line[1:5])
            if first_year == 0:
                first_year = current_year
            single_path_statistics.append(dict())
            single_path_statistics[path_number][current_year] = dict()
            next = 'names'
            continue
        if next == 'names':
            single_path_statistics[path_number][current_year]['authors'] = set([a.strip()[1:len(a.strip())-1] for a in line[1:len(line)-1].split(',')])
            next = 'iedembed'
            continue
        if next == 'iedembed':
            values = line.split(',')
            single_path_statistics[path_number][current_year]['ied'] = float(values[0])
            single_path_statistics[path_number][current_year]['embed'] = float(values[1])
            next = 'conf_identifiers'
            continue
        if next == 'conf_identifiers':
            single_path_statistics[path_number][current_year]['identifiers'] = line[1:len(line)-1].split(',')
            next = 'algorithms'
            continue
        if next == 'algorithms':
                single_path_statistics[path_number][current_year]['algorithms'] = set(int(a) for a in line[1:len(line)-1].split(','))
                next = 'overlap_with_following'
                continue
        if next == 'overlap_with_following':
            buffer = float(line)
            next = 'weight'
            continue
        if next == 'weight':
            if 'weight' in line:
                single_path_statistics[path_number][current_year]['overlap'] = buffer
                next = 'year'
            else:
                single_path_statistics[path_number][first_year]['avd'] = buffer
                single_path_statistics[path_number][first_year + 1]['avd'] = float(line)
                countdown = len(single_path_statistics[path_number]) - 3
                next = 'avd'
            continue
        if next == 'avd':
            if countdown >= 0:
                single_path_statistics[path_number][first_year + len(single_path_statistics[path_number]) - countdown - 1]['avd'] = float(line)
                next = 'avd'
                countdown -= 1
            else:
                next = None
            continue
        if line == 'start':
            next = 'year'
            continue
    if no_postprocessing is not None:
        colors_in = json.load(no_postprocessing)

    if postprocessing is not None:
        colors_in_post = json.load(postprocessing)
    nx.draw_networkx(graph, pos, with_labels=True)
    matplotlib.pyplot.show()
    flag = 0
    for year in single_path_statistics[args.sol_num]:
        draw_nodes = set(single_path_statistics[args.sol_num][year]['authors'])

        draw_copy = copy.copy(draw_nodes)
        for author in draw_copy:
            draw_nodes = draw_nodes.union(set(collab_graphs[year].neighbors(author_lst[author_lst.index(author)])))
            print("Neighbors: ", set(collab_graphs[year].neighbors(author)))
        if no_postprocessing is not None:
            color_map = list(set(colors_in) & set(authors_yearly[year]))
            print("bruh")
            print(color_map)
            print(year)
            print("Draw_nodes_2: ", draw_nodes)
            nx.draw_networkx_nodes(graph, pos, nodelist=draw_nodes) # , pos = nx.spring_layout(output_graph))
            nx.draw_networkx_nodes(graph, pos, nodelist=list(set(single_path_statistics[args.sol_num][year]['authors'])), node_color='r')
            nx.draw_networkx_edges(graph, pos, edgelist=nx.subgraph(collab_graphs[year], draw_nodes).edges)
            node_labels = {n: n for n in draw_nodes}
        else:
            nx.draw_networkx_nodes(graph, pos, nodelist=single_path_statistics[args.sol_num][year]['authors']) # , pos = nx.spring_layout(output_graph))
            nx.draw_networkx_edges(graph, pos, nodelist=color_map)
        if postprocessing is not None:
            print("Postprocessing is not none")
            print(colors_in_post)
            color_map = list(set(colors_in_post) & set(single_path_statistics[args.sol_num][year]['authors']))
            print(color_map)
            print(year)
            nx.draw_networkx_nodes(graph, pos, nodelist=color_map, node_color='g')
        else:
            nx.draw_networkx(graph, pos, nodelist=single_path_statistics[args.sol_num][year]['authors']) # , pos = nx.spring_layout(output_graph))
            #nx.draw_networkx_edges(graph, pos, nodelist=color_map)
        if flag != 0:
            color_map = list(set(single_path_statistics[args.sol_num][year]['authors']) & set(single_path_statistics[args.sol_num][year - 1]['authors']))
            nx.draw_networkx_nodes(graph, pos, nodelist=color_map, node_color='y')
        else:
            flag = 1
        nx.draw_networkx_labels(graph, pos, node_labels)
        edge_labels = nx.get_edge_attributes(nx.subgraph(collab_graphs[year], draw_nodes), 'weight')

        nx.draw_networkx_edge_labels(graph, pos, edge_labels = edge_labels)
        matplotlib.pyplot.show()





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
                        if collab_graphs[year].has_edge(combo[0], combo[1]):
                            collab_graphs[year][combo[0]][combo[1]]['weight'] = collab_graphs[year][combo[0]][combo[1]]['weight'] + 1
                        else:
                            collab_graphs[year].add_edge(combo[0], combo[1], weight = 1)
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



if __name__=="__main__":
    main()
