from pyspark import SparkContext, SparkConf
import os
import sys
import itertools
import time
from copy import deepcopy


def output_text(true_pair_list, output_path):
    with open(output_path, 'w+') as o_file:
        for item in true_pair_list:
            o_file.writelines(str(item)[1:-1] + "\n")
        o_file.close()


def build_tree(top_node, node_list, edge_dict):
    node_list2 = list(node_list)
    layer_list = [[top_node]]
    visited_set = set([top_node])
    valid_edge_list = []
    node_list2.remove(top_node)
    while len(node_list2) != 0:
        current_node = list()
        current_edge = list()
        for prev_node in layer_list[-1]:
            current_node += [node for node in edge_dict[prev_node] if node not in visited_set]
            current_edge += [sorted([prev_node, node]) for node in edge_dict[prev_node] if node in current_node]
        #print('length of node list: ', len(node_list2))
        #print('length of visited node:', len(visited_set))
        #print('current node list length', len(current_node))

        if len(current_node) == 0:
            break
        else:
            layer_list.append(list(set(current_node)))
            valid_edge_list.append(current_edge)
            visited_set = visited_set | set(current_node)
            for node in list(set(current_node)):
                node_list2.remove(node)
    return layer_list, valid_edge_list


def cal_weight_of_edges(layer_list, valid_edge_list):
    edge_weight_dict = dict()
    while len(valid_edge_list) != 0:
        current_edges = valid_edge_list.pop()
        current_nodes = layer_list.pop()
        node_short_path_dict = dict()
        if len(valid_edge_list) == 0:
            for node in layer_list[-1]:
                node_short_path_dict[node] = 1
        else:
            for node in layer_list[-1]:
                node_short_path_dict[node] = len([edge for edge in valid_edge_list[-1] if node in edge])
        # calculate accumulate from previous edge weight for every current node, store in a dict
        node_weight_dict = dict()
        for node in current_nodes:
            node_weight_dict.setdefault(node, 1)
            node_weight_dict[node] += sum([v for k, v in edge_weight_dict.items() if node in k])
        # calculate current edge weight
        for node, weight in node_weight_dict.items():
            relate_edge = [edge for edge in current_edges if node in edge]
            relate_upper_nodes = []
            for edge in relate_edge:
                if edge[0] == node:
                    relate_upper_nodes.append(edge[1])
                else:
                    relate_upper_nodes.append(edge[0])
            total_weight = 0
            for up_node in relate_upper_nodes:
                total_weight += node_short_path_dict[up_node]
            for e in relate_upper_nodes:
                edge_weight_dict[tuple(sorted([node, e]))] = node_weight_dict[node]/total_weight*node_short_path_dict[e]
    return edge_weight_dict


def cal_betweenness(node_list, edge_dict):
    # iter over all node, let each node as top node to construct a tree
    between_dict = dict()
    for top_node in node_list:
        # bulid a tree given a top node
        # return: layer_list, valid_edge_list
        # layer_list: [[top node], [node in layer 1], [nodes in layer 2], ...]
        # valid_edege_list: [[valid edge pairs in layer 1], [valid edge pairs in layer2], ...]
        layer_list, valid_edge_list = build_tree(top_node, node_list, edge_dict)

        # given layer_list and valid_edge_list, calculate weight of each edges, store in a dictionary
        # return: {(uid1, uid2):weight}
        edge_weight_dict = cal_weight_of_edges(layer_list, valid_edge_list)
        for edge, weight in edge_weight_dict.items():
            between_dict.setdefault(edge, 0)
            between_dict[edge] += weight
    between_list = [[edge, weight/2] for edge, weight in between_dict.items()]
    between_list.sort(key=lambda row: [row[1], row[0]], reverse=True)
    return between_list


# split the graph by remove on edge (split_edge)
# return a list of sub-graph
def split_graph(edge_dict, split_edge):
    current_graph = deepcopy(edge_dict)

    # Remove the split edge
    current_graph[split_edge[0]].remove(split_edge[1])
    current_graph[split_edge[1]].remove(split_edge[0])
    community_list = []
    remain_nodes = list(set(current_graph.keys()))
    # after removing the split edge, pick a node and start search its neighbors
    # define as a community when disconnected with other nodes
    # iterate until all nodes are searched
    while len(remain_nodes) != 0:
        # start from a node
        start_node = remain_nodes[0]
        visited_nodes = set()
        # remove the start node from remain nodes
        remain_nodes.remove(start_node)
        visited_nodes.add(start_node)
        child_nodes = current_graph[start_node]
        while len(child_nodes) != 0:
            temp_node_list = []
            for child_node in child_nodes:
                if child_node not in visited_nodes:
                    visited_nodes.add(child_node)
                    remain_nodes.remove(child_node)
                    temp_node_list += [node for node in current_graph[child_node] if node not in visited_nodes
                                       and node not in child_nodes]
            child_nodes = list(set(temp_node_list))
        community_list.append(sorted(list(visited_nodes)))
    return community_list, current_graph


def cal_modularity(community_list, edge_dict, m_2):

    moduality = 0
    for community in community_list:
        for i in community:
            for j in community:
                moduality += (j in edge_dict[i]) - (len(edge_dict[i])*len(edge_dict[j]))/m_2
    return moduality/m_2


def find_best_community(node_list, edge_dict):
    between_list = cal_betweenness(node_list, edge_dict)
    current_graph = deepcopy(edge_dict)
    max_modularity = -1
    best_community = None
    # stop condition: no more edge in current graph
    while sum([len(v) for k, v in current_graph.items()]) != 0:
        # the edge that should be removed: with highest betweenness
        removed_edge = between_list[0][0]

        # check community after remove the edge
        # return community_list, current_graph
        # community_list: [[nodes in community 1], [nodes in community 2], ..]
        # current_graph: update current_graph by removing the target edge
        community_list, current_graph = split_graph(current_graph, removed_edge)

        # calculate Modularity
        # not using current_graph, use the origional graph
        m_2 = sum([len(v) for k, v in edge_dict.items()])
        modularity = cal_modularity(community_list, edge_dict, m_2)
        print('Current modularity:', modularity)
        print('Best modularity:', max_modularity)
        print('current number of community:', len(community_list))
        if modularity > max_modularity:
            max_modularity = modularity
            best_community = community_list

        # calculate betweenness for current graph
        between_list = cal_betweenness(node_list, current_graph)
    best_community.sort(key=lambda row: [len(row), row[0]])
    return best_community



def task2(filter_threshold, input_file_path, betweenness_output_file_path, community_output_file_path):
    conf = SparkConf().setMaster('local[3]').setAppName('HW4_task1')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    input_text = sc.textFile(input_file_path)
    row_name = input_text.first()
    '''
    user_index = input_text.filter(lambda row: row != row_name).map(lambda row: row.split(',')[0]).distinct().\
        zipWithIndex().collectAsMap()
    inv_user_index = {v:k for k, v in user_index.items()}
    business_index = input_text.filter(lambda row: row != row_name).map(lambda row: row.split(',')[1]).distinct().\
        zipWithIndex().collectAsMap()
    inv_business_index = {v:k for k, v in business_index.items()}
    '''
    input_data_dict = input_text.filter(lambda row: row != row_name).\
        map(lambda row: [row.split(',')[0], row.split(',')[1]]).groupByKey().\
        map(lambda row: [row[0], list(row[1])]).collectAsMap()
    node_list = []
    edge_list = []
    for pair in itertools.combinations(input_data_dict.keys(), 2):
        if len(set(input_data_dict[pair[0]])&set(input_data_dict[pair[1]])) >= int(filter_threshold):
            node_list.append(pair[0])
            node_list.append(pair[1])
            edge_list.append([pair[0], pair[1]])
            edge_list.append([pair[1], pair[0]])
    # list of nodes
    node_list = list(set(node_list))
    print('total number of nodes:', len(node_list))
    # edge_dict: {node1:[n2, n3, n4]}
    edge_dict = sc.parallelize(edge_list).groupByKey().map(lambda row: [row[0], list(row[1])]).collectAsMap()
    # task 2.1
    between_list = cal_betweenness(node_list, edge_dict)
    output_text(between_list, betweenness_output_file_path)

    # task 2.2
    best_community_list = find_best_community(node_list, edge_dict)
    output_text(best_community_list, community_output_file_path)


if __name__ == "__main__":
    #'''
    filter_threshold = 7
    input_file_path = "resource/asnlib/publicdata/ub_sample_data.csv"
    betweenness_output_file_path = 'task2b'
    community_output_file_path= "task2c"
    #'''
    #filter_threshold, input_file_path, betweenness_output_file_path, community_output_file_path = sys.argv[1:5]
    start = time.time()
    task2(filter_threshold, input_file_path, betweenness_output_file_path, community_output_file_path)
    print("Duriation: %d s" % (time.time() - start))