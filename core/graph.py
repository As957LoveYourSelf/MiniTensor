"""
Begin Date: 2021.11.15
Author: ChuQi Zhang
Last Change Date: 2022.3.6

=======================================
Need to do:
1. Add backward(auto gradient) function
=======================================
"""


class Graph:
    def __init__(self):
        super(Graph, self).__init__()
        self.__forward_graph = {}
        self.root_node = None

    def graph_backward(self):
        pass

    def show_graph(self):
        print(f"forward graph: {[(k, [i.node_name for i in v]) for (k, v) in self.__forward_graph.items()]}")
        print(f"root node: {self.root_node if self.root_node is None else self.root_node.node_name}")

    def add_nodes(self, *args, **kwargs):
        key = kwargs.get('key')
        value = kwargs.get('value')
        _type = kwargs.get('type')
        assert isinstance(value, list)
        assert isinstance(key, str)
        if str.lower(_type) == 'forward':
            try:
                self.__forward_graph[key] += value
            except KeyError:
                self.__forward_graph[key] = value
        elif str.lower(_type) == 'root_node':
            self.root_node = value
        else:
            raise TypeError(f"you should use 'forward' or 'backward', instead {_type}")


global graph
global nodes_num
global nodes_name

nodes_num = 0
nodes_name = []
graph = Graph()
