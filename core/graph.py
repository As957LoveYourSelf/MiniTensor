"""
Begin Date: 2021.11.15
Author: ChuQi Zhang
Last Change Date: 2021.11.18
"""


class Graph:
    def __init__(self):
        super(Graph, self).__init__()
        self.__forward_graph = {}
        self.__backward_graph = {}

    def show_graph(self):
        print(f"forward graph: {[(k,[i.node_name for i in v]) for (k,v) in self.__forward_graph.items()]}")
        print(f"backward graph: {[(k,[i.node_name for i in v]) for (k,v) in self.__backward_graph.items()]}")

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
        elif str.lower(_type) == 'backward':
            try:
                self.__backward_graph[key] += value
            except KeyError:
                self.__backward_graph[key] = value
        else:
            raise TypeError(f"you should use 'forward' or 'backward', instead {_type}")


global graph
global nodes_num
global nodes_name

nodes_num = 0
nodes_name = []
graph = Graph()



