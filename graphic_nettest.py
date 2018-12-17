import graph_nets as gn
import sonnet as snt

input_graph = get_graphs

graph_net_module = gn.modules.GraphNetwork(
    edge_model_fn = lambda : snt.nets.MLP([32, 32]),
    node_model_fn = lambda : snt.nets.MLP([32, 32]),
    global_model_fn = lambda : snt.nets.MLP([32, 32]),
)

output_graphs = graph_net_module(input_graph)