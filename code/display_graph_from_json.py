from networkx.readwrite import json_graph
import json
import sys
import argparse
import networkx as nx
import matplotlib
import os.path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_end', type=str)
    parser.add_argument('-s', '--singlesource', dest='singlesource', action='store_true')
    args = parser.parse_args()
    colors = None
    if args.singlesource:
        infile = open("../resources/outputSingleSourceTarget" + args.file_end + ".json")
        if os.path.isfile("../resources/outputSingleSourceTargetColors" + args.file_end + ".json"):
            colors = open("../resources/outputSingleSourceTargetColors" + args.file_end + ".json")
    else:
        infile = open("../resources/output" + args.file_end + ".json")
        if os.path.isfile("../resources/outputColors" + args.file_end + ".json"):
            colors = open("../resources/outputColors" + args.file_end + ".json")
    s = json.load(infile)
    graph = json_graph.adjacency_graph(s)
    labels = nx.get_edge_attributes(graph, 'weight')
    pos = nx.spring_layout(graph)
    print(graph.nodes())
    if colors is not None:
        color_map = json.load(colors)
        print(color_map)
        nx.draw_networkx(graph, pos) # , pos = nx.spring_layout(output_graph))
        nx.draw_networkx_nodes(graph, pos, nodelist=color_map, node_color='r')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels = labels)
    else:
        nx.draw_networkx(graph, pos) # , pos = nx.spring_layout(output_graph))
        nx.draw_networkx_edge_labels(graph, pos, edge_labels = labels)
    matplotlib.pyplot.show()

if __name__ == "__main__":
    main()