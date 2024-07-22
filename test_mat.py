import numpy as np

import paibox as pb
from paibox.backend.graphs import (
    PAIGraph,
    convert2routing_groups,
    get_node_degrees,
    get_succ_cb_by_node,
    toposort,
)
diag_matrix = np.diag([1,2,3,4]).astype(np.int8)

# 将两个5x5的对角矩阵水平拼接成一个5x10的数组
weight = np.hstack((diag_matrix, diag_matrix)).astype(np.int8)

# print(weight)

class NetForMAT(pb.Network):

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(input=None, shape_out=(3, 4), name="inp1")
        self.n0 = pb.IF((3, 4), threshold=1, reset_v=0, name = "n0")
        self.n1 = pb.TonicSpiking((3, 8), 3, name="n1")
        self.n2 = pb.TonicSpiking((3, 8), 3, name="n2")
        self.s0 = pb.FullConn(self.inp1, self.n0, conn_type=pb.SynConnType.One2One, name="s0")
        self.s1 = pb.MatMul2d(self.n0, self.n1, weight, "s1")
        self.s2 = pb.FullConn(self.n1, self.n2, conn_type=pb.SynConnType.All2All, weights=1, name="s2")

net = NetForMAT()
# net.s1.income_displs  = [4, 4, 4]
# net.s1.outcome_displs = [8, 8, 8]
# print(net.s1.comm.connectivity)
mapper = pb.Mapper()
mapper.clear()
mapper.build(net)
graph_info = mapper.compile(weight_bit_optimization=False, grouping_optim_target="both", no_twisted_branch=True)
# print ("Nodes:")
# for node in graph._raw_nodes.values():
#     print(f"\t{node.name}")
# print ("Edges:")
# for edge in graph._raw_edges.values():
#     print(f"\t{edge.name}: {edge.source.name} -> {edge.dest.name}")
#     print(edge.connectivity.astype(int))
# graph._pre_optim()
# print ("Nodes:")
# for node in graph._raw_nodes.values():
#     print(f"\t{node.name}")
# print ("Edges:")
# for edge in graph._raw_edges.values():
#     print(f"\t{edge.name}: {edge.source.name} -> {edge.dest.name}")
#     print(edge.connectivity.astype(int))