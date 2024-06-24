import math
import time
from graph_operation import graph_depth
from copy import deepcopy
from collections import defaultdict


class SampleSubgraph:
    def __init__(self, query, data_graph):
        # data information contains:
        # 0: id 1: label 2: degree 3: edge_info 4: edge_label 5: vertex neighbor 6: label_dict
        self.query = query
        self.data_graph = data_graph

    def find_subgraph(self, start_query_vertex, candidates):
        output_vertices = list()
        output_v_label = list()
        output_degree = list()
        output_edges = list()
        output_edge_label = list()
        output_v_neigh = list()
        depth = graph_depth(self.query, start_query_vertex)
        candidate_u = candidates[start_query_vertex]
        all_candidates = list()
        for i in range(len(candidates)):
            for j in range(len(candidates[i])):
                all_candidates.append(candidates[i][j])
        all_candidates = list(set(all_candidates))
        # print(all_candidates)
        data_label = self.data_graph[1]
        data_neigh = self.data_graph[5]
        # two possible ways: 1. start from all candidates and perform BFS search
        # 2. when the candidate is visited, we don't do the search starting from that node.
        all_need_visited = deepcopy(candidate_u)
        while len(all_need_visited) > 0:
            search_depth = 0
            queue = list()
            depth_queue = list()
            new_graph_vertices = list()
            new_graph_v_label = dict()
            new_graph_v_degree = defaultdict(lambda : 0)
            new_e_u = list()
            new_e_v = list()
            new_edge_label = list()
            new_graph_v_neigh = defaultdict(list)
            start_data_vertex = all_need_visited.pop(0)
            queue.append(start_data_vertex)
            depth_queue.append(search_depth)
            new_graph_vertices.append(start_data_vertex)
            new_graph_v_label[start_data_vertex] = data_label[start_data_vertex]
            while len(queue)>0:
                current_data_vertex = queue.pop(0)
                search_depth = depth_queue.pop(0)
                if search_depth > depth:
                    break
                for v in data_neigh[current_data_vertex]:
                    if v in all_need_visited:
                        all_need_visited.remove(v)
                    if v not in new_graph_vertices and v in all_candidates:
                        new_graph_vertices.append(v)
                        new_graph_v_label[v] = data_label[v]
                        queue.append(v)
                        depth_queue.append(search_depth+1)
                        for neigh_v in data_neigh[v]:
                            if neigh_v in new_graph_vertices:
                                # two way (undirected) edges
                                new_e_u.append(v)
                                new_e_v.append(neigh_v)
                                new_e_u.append(neigh_v)
                                new_e_v.append(v)
                                new_graph_v_degree[v] += 1
                                new_graph_v_degree[neigh_v] += 1

                                # neighbor should be added only once.
                                # new_graph_v_neigh[v].append(neigh_v)
                                new_graph_v_neigh[neigh_v].append(v)

                                new_edge_label.append(1)
                                new_edge_label.append(1)
            new_graph_edges = [new_e_u, new_e_v]
            output_vertices.append(new_graph_vertices)
            output_v_label.append(deepcopy(new_graph_v_label))
            output_degree.append(deepcopy(new_graph_v_degree))
            output_edges.append(new_graph_edges)
            output_edge_label.append(new_edge_label)
            output_v_neigh.append(deepcopy(new_graph_v_neigh))
        return output_vertices, output_v_label, output_degree, output_edges, output_edge_label, output_v_neigh

    def find_subgraph_induced(self, candidates):
        t_0 = time.time()
        all_candidates = list()
        for i in range(len(candidates)):
            for j in range(len(candidates[i])):
                all_candidates.append(candidates[i][j])
        all_candidates = list(set(all_candidates))
        all_need_visited = deepcopy(all_candidates)
        queue = list()
        depth_queue = list()
        new_graph_vertices = list()
        new_graph_v_label = dict()
        new_graph_v_degree = defaultdict(lambda : 0)
        new_e_u = list()
        new_e_v = list()
        new_edge_label = list()
        new_graph_v_neigh = defaultdict(list)

        # get data graph information
        data_label = self.data_graph[1]
        data_edge = self.data_graph[3]
        data_neigh = self.data_graph[5]
        
        new_graph_vertices = deepcopy(all_candidates)
        for v in new_graph_vertices:
            new_graph_v_label[v] = data_label[v]
        t_1 = time.time()
        print('sample satage 1: {}s'.format(t_1-t_0))
        # for i in range(len(data_edge[0])):
        #     # if two nodes are both in candidate set, the edge is included for new graph
        #     u = data_edge[0][i]
        #     v = data_edge[1][i]
        #     if u in all_candidates and v in all_candidates:
        #         new_e_u.append(u)
        #         new_e_v.append(v)
        #         # only add once, since the edge will appear twice.
        #         new_graph_v_degree[u] += 1
        #         new_graph_v_neigh[u].append(v)
        #         new_edge_label.append(1)

        for vertex in new_graph_vertices:
            # if two nodes are both in candidate set, the edge is included for new graph
            neigh_of_v = data_neigh[vertex]
            for u in neigh_of_v:
                if u in all_candidates:
                    new_e_u.append(u)
                    new_e_v.append(vertex)
                    # only add once, since the edge will appear twice.
                    new_graph_v_degree[vertex] += 1
                    new_graph_v_neigh[vertex].append(u)
                    new_edge_label.append(1)

        t_2 = time.time()
        print('sample stage 2: {}s'.format(t_2-t_1))
        new_edges = [deepcopy(new_e_u), deepcopy(new_e_v)]
        new_vertices = new_graph_vertices
        new_v_label = new_graph_v_label
        new_degree = deepcopy(new_graph_v_degree)
        new_edge_label = deepcopy(new_edge_label)
        new_v_neigh = new_graph_v_neigh

        check_info = [new_vertices, new_v_label, new_degree, new_edges, new_edge_label, new_v_neigh]
        output_vertices, output_v_label, output_degree, output_edges, output_edge_label, output_v_neigh = self._split_graph(check_info)
        t_3 = time.time()
        print('sample stage 3: {}s'.format(t_3-t_2))
        # output_graph_info = [output_vertices, output_v_label, output_degree, output_edges, output_edge_label, output_v_neigh]

        return output_vertices, output_v_label, output_degree, output_edges, output_edge_label, output_v_neigh

    def load_induced_subgraph(self, candidates, induced_subgraph_list, neighbor_offset):
        queue = list()
        depth_queue = list()
        new_graph_vertices = list()
        new_graph_v_label = dict()
        new_graph_v_degree = defaultdict(lambda : 0)
        new_e_u = list()
        new_e_v = list()
        new_edge_label = list()
        new_graph_v_neigh = defaultdict(list)

        # get data graph information
        data_label = self.data_graph[1]
        data_edge = self.data_graph[3]
        data_neigh = self.data_graph[5]

        new_graph_vertices = deepcopy(candidates)
        for v in new_graph_vertices:
            new_graph_v_label[v] = data_label[v]

        for i in range(len(candidates)):
            vertex = candidates[i]
            strat_index = neighbor_offset[i]
            end_index = neighbor_offset[i+1]
            for j in range(strat_index, end_index):
                u = induced_subgraph_list[j]
                new_e_u.append(u)
                new_e_v.append(vertex)
                # only add once, since the edge will appear twice.
                new_graph_v_degree[vertex] += 1
                new_graph_v_neigh[vertex].append(u)
                new_edge_label.append(1)
        
        new_edges = [deepcopy(new_e_u), deepcopy(new_e_v)]
        new_vertices = new_graph_vertices
        new_v_label = new_graph_v_label
        new_degree = deepcopy(new_graph_v_degree)
        new_edge_label = deepcopy(new_edge_label)
        new_v_neigh = new_graph_v_neigh

        check_info = [new_vertices, new_v_label, new_degree, new_edges, new_edge_label, new_v_neigh]
        output_vertices, output_v_label, output_degree, output_edges, output_edge_label, output_v_neigh = self._split_graph(check_info)

        return output_vertices, output_v_label, output_degree, output_edges, output_edge_label, output_v_neigh

    def _split_graph(self, graph_info):
        vertices_id = graph_info[0]
        vertices_label = graph_info[1]
        vertices_neighbor = graph_info[5]
        num_vertices = len(vertices_id)
        to_be_visited = deepcopy(vertices_id)
        

        # initialize the output lists
        output_vertices = list()
        output_v_label = list()
        output_v_degree = list()
        output_edges = list()
        output_e_label = list()
        output_v_neigh = list()

        while len(to_be_visited) > 0:
            # initialize the temp containers.
            out_temp_vertices = list()
            out_temp_v_label = dict()
            out_temp_v_degree = defaultdict(lambda: 0)
            out_temp_e_u = list()
            out_temp_e_v = list()
            out_temp_e_label = list()
            out_temp_v_neigh = defaultdict(list)

            start_node = to_be_visited[0]
            # to_be_visited.remove(start_node)
            queue = list()
            queue.append(start_node)

            while len(queue) > 0:
                current_node = queue.pop(0)
                current_neighbors = vertices_neighbor[current_node]
                try:
                    to_be_visited.remove(current_node)
                except ValueError:
                    # print('node {} has been removed'.format(current_node))
                    continue   # if there is no this node in the to be visited set, we donot need to compute it again. will lead to bugs.  
                out_temp_vertices.append(current_node)        
                out_temp_v_label[current_node] = vertices_label[current_node]         
                for v in current_neighbors:
                    # a BFS, do we need to check whether it is in the to be visited set? (in Queue it should!)
                    out_temp_e_u.append(current_node)
                    out_temp_e_v.append(v)       # add a one-way edge, it will be added again.
                    out_temp_e_label.append(1)   # edge label is always 1.
                    out_temp_v_degree[current_node] += 1
                    out_temp_v_neigh[current_node].append(v)
                    if v in to_be_visited:
                        queue.append(v)

            output_vertices.append(deepcopy(out_temp_vertices))
            output_v_label.append(deepcopy(out_temp_v_label))
            output_v_degree.append(deepcopy(out_temp_v_degree))
            output_edges.append([deepcopy(out_temp_e_u), deepcopy(out_temp_e_v)])
            output_e_label.append(deepcopy(out_temp_e_label))
            output_v_neigh.append(deepcopy(out_temp_v_neigh))
        
        return output_vertices, output_v_label, output_v_degree, output_edges, output_e_label, output_v_neigh

    def update_query(self, query):
        self.query = query


def _all_train_and_test(training_percent, name_list):
    example_name = name_list[0]
    train_name_list = list()
    test_name_list = list()
    potential_names_4 = list()
    potential_names_8 = list()
    potential_names_12 = list()
    potential_names_16 = list()
    if 'youtube' in example_name or 'eu2005' in example_name or 'patent' in example_name:
        for i in range(len(name_list)):
            if '_4_' in name_list[i]:
                potential_names_4.append(name_list[i])
            elif '_8_' in name_list[i]:
                potential_names_8.append(name_list[i])
        train_name_list.extend(potential_names_4[:math.floor(len(potential_names_4) * training_percent)])
        train_name_list.extend(potential_names_8[:math.floor(len(potential_names_8) * training_percent)])
        test_name_list.extend(potential_names_4[math.floor(len(potential_names_4) * training_percent):])
        test_name_list.extend(potential_names_8[math.floor(len(potential_names_8) * training_percent):])
        return train_name_list, test_name_list
    else:
        for i in range(len(name_list)):
            if '_4_' in name_list[i]:
                potential_names_4.append(name_list[i])
            elif '_8_' in name_list[i]:
                potential_names_8.append(name_list[i])
            elif '_16_' in name_list[i]:
                potential_names_16.append(name_list[i])
        # print(len(potential_names_4))
        train_name_list.extend(potential_names_4[:math.floor(len(potential_names_4) * training_percent)])
        train_name_list.extend(potential_names_8[:math.floor(len(potential_names_8) * training_percent)])
        train_name_list.extend(potential_names_16[:math.floor(len(potential_names_16) * training_percent)])
        test_name_list.extend(potential_names_4[math.floor(len(potential_names_4) * training_percent):])
        test_name_list.extend(potential_names_8[math.floor(len(potential_names_8) * training_percent):])
        test_name_list.extend(potential_names_16[math.floor(len(potential_names_16) * training_percent):])
        return train_name_list, test_name_list




def train_and_test(query_vertices_num, training_percent, name_list):
    train_name_list = list()
    test_name_list = list()
    if query_vertices_num == '4':
        target_string = 'dense_4_'
    elif query_vertices_num == '8':
        target_string = '_8_'
    elif query_vertices_num == '12':
        target_string = '_12_'
    elif query_vertices_num == '16':
        target_string = '_16_'
    elif query_vertices_num == '24':
        target_string = '_24_'
    elif query_vertices_num == '32':
        target_string = '_32_'
    elif query_vertices_num == 'all':
        return _all_train_and_test(training_percent, name_list)
    else:
        raise NotImplementedError('The query vertex number input is not supported')
    potential_names = list()
    for i in range(len(name_list)):
        if target_string in name_list[i]:
            potential_names.append(name_list[i])
    total_num = len(potential_names)
    train_num = math.floor(total_num*training_percent)
    test_num = total_num - train_num
    for i in range(train_num):
        train_name_list.append(potential_names[i])
    for i in range(test_num):
        test_name_list.append(potential_names[train_num+i])

    return train_name_list, test_name_list
