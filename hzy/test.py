import numpy as np

import paibox as pb
from paicorelib import (
    LCM,
    LDM,
    LIM,
    NTM,
    RM,
    SIM,
    TM,
    MaxPoolingEnable,
    SpikeWidthFormat,
    SNNModeEnable,
)
#relu = pb.neuron.ReLu([2,2],bit_trunc=8)
#x=np.array([[5,-10, 3, -5],[-2, 7, 8, 9],[1,-2,10,-5],[5,9 ,5, 6]])
x=np.array([[5, 10, 3, -5]])
# inp = pb.InputProj(input=x, shape_out=(4, 4), keep_shape=True, name='inp1')
# s1 = pb.synapses.NoDecay(
#             inp,
#             relu,
#             weights=1,
#             conn_type=pb.synapses.ConnType.All2All,
#         )
# print(s1.weights.shape)
weight1=np.array([[1,0],[1,0],[0,1],[0,1]])

class fcnet(pb.Network):
    def __init__(self, weight1):
        super().__init__()

        self.i1 = pb.InputProj(input=x, shape_out=(4,), keep_shape=True, name='inp1')
        self.relu = pb.neuron.ReLu(4,bit_trunc=8)
        self.s1 = pb.synapses.NoDecay(
            self.i1,
            self.relu,
            weights=np.array([[1,-2, 3, -4],[1,-2, 3, -4],[1,-2, 3, -4],[1,-2, 3, -4]]),
            conn_type=pb.synapses.SynConnType.All2All,
        )
        self.maxpool = pb.neuron.MaxPooling(shape=3, kernel_size=2, stride=1)
        self.s2 = pb.synapses.NoDecay(
            self.relu,
            self.maxpool,
            conn_type=pb.synapses.SynConnType.ToMaxPooling,
        )
        self.probe1 = pb.simulator.Probe(target=self.maxpool, attr="voltage")

fcnet = fcnet(1)
sim = pb.Simulator(fcnet)
sim.run(2, reset=True)
print("maxpool_v:",sim.data[fcnet.probe1])



