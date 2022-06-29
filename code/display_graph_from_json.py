from networkx.readwrite import json_graph
import json
import sys
import argparse
import networkx as nx
import matplotlib
import os.path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('no_postprocessing_path', type=str)
    parser.add_argument('postprocessing_path', type=str)
    args = parser.parse_args()
    no_postprocessing = None
    postprocessing = None
    infile = open(args.path)
    if os.path.isfile(args.no_postprocessing_path):
        no_postprocessing = open(args.no_postprocessing_path)
    if os.path.isfile(args.postprocessing_path):
        postprocessing = open(args.postprocessing_path)
    s = json.load(infile)
    graph = json_graph.adjacency_graph(s)
    labels = nx.get_edge_attributes(graph, 'weight')
    print(labels)
    for key in labels.keys():
        labels[key] = round(labels[key], 2)
    pos = nx.spring_layout(graph)
    print(graph.nodes())
    if no_postprocessing is not None:
        color_map = json.load(no_postprocessing)
        print(color_map)
        nx.draw_networkx(graph, pos) # , pos = nx.spring_layout(output_graph))
        nx.draw_networkx_nodes(graph, pos, nodelist=color_map, node_color='r')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels = labels)
    else:
        nx.draw_networkx(graph, pos) # , pos = nx.spring_layout(output_graph))
        nx.draw_networkx_edge_labels(graph, pos, edge_labels = labels)
    if postprocessing is not None:
        color_map = json.load(postprocessing)
        print(color_map)
        nx.draw_networkx_nodes(graph, pos, nodelist=color_map, node_color='g')
    else:
        nx.draw_networkx(graph, pos) # , pos = nx.spring_layout(output_graph))
        nx.draw_networkx_edge_labels(graph, pos, edge_labels = labels)
    matplotlib.pyplot.show()

if __name__ == "__main__":
    main()