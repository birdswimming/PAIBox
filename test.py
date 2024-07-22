import numpy as np

import paibox as pb


class NetForTest3(pb.Network):

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(input=None, shape_out=(400,), name="inp1")
        # self.inp2 = pb.projection.InputProj(input=None, shape_out=(300,), name="inp2")
        self.n0 = pb.TonicSpiking(400, 3, name="n0")
        # self.n0_copy = pb.neuron.TonicSpiking(400, 3, name="n0_copy")
        self.n1 = pb.TonicSpiking(400, 3, name="n1")
        self.n2 = pb.TonicSpiking(800, 3, name="n2")
        self.n3 = pb.TonicSpiking(400, 4, name="n3")
        self.n4 = pb.TonicSpiking(300, 4, name="n4")
        self.s0 = pb.FullConn(
            self.inp1, self.n0, conn_type=pb.SynConnType.One2One, name="s0"
        )
        # self.s0_copy = pb.FullConn(
        #     self.inp1, self.n0_copy, conn_type=pb.SynConnType.One2One, name="s0_copy"
        # )
        self.s1 = pb.FullConn(
            self.n0, self.n1, conn_type=pb.SynConnType.One2One, name="s1"
        )
        self.s2 = pb.FullConn(
            self.n1, self.n2, conn_type=pb.SynConnType.All2All, name="s2"
        )
        self.s3 = pb.FullConn(
            self.n2, self.n3, conn_type=pb.SynConnType.All2All, name="s3"
        )
        self.s4 = pb.FullConn(
            self.n0, self.n4, conn_type=pb.SynConnType.All2All, name="s4"
        )
        self.s5 = pb.FullConn(
            self.n4, self.n2, conn_type=pb.SynConnType.All2All, name="s5"
        )
        # self.s6 = pb.FullConn(
        #     self.inp2, self.n4, conn_type=pb.SynConnType.One2One, name="s6"
        # )



class NetForTest(pb.Network):

    def __init__(self):
        super().__init__()
        self.inp1 = pb.InputProj(input=None, shape_out=(400,), name="inp1")
        self.inp2 = pb.InputProj(input=None, shape_out=(400,), name="inp2")
        self.n0 = pb.TonicSpiking(400, 3, name="n0")
        self.n1 = pb.TonicSpiking(400, 3, name="n1")
        self.n2 = pb.TonicSpiking(800, 3, name="n2")
        self.n3 = pb.TonicSpiking(400, 4, name="n3")
        self.n4 = pb.TonicSpiking(300, 4, name="n4")
        self.s0 = pb.FullConn(
            self.inp1, self.n0, conn_type=pb.SynConnType.One2One, name="s0"
        )
        self.s1 = pb.FullConn(
            self.inp1, self.n1, conn_type=pb.SynConnType.One2One, name="s1"
        )
        self.s2 = pb.FullConn(
            self.n0, self.n2, conn_type=pb.SynConnType.All2All, name="s2"
        )
        self.s3 = pb.FullConn(
            self.n1, self.n2, conn_type=pb.SynConnType.All2All, name="s3"
        )
        self.s4 = pb.FullConn(
            self.n2, self.n3, conn_type=pb.SynConnType.All2All, name="s4"
        )
        self.s5 = pb.FullConn(
            self.n2, self.n4, conn_type=pb.SynConnType.All2All, name="s5"
        )
        self.s6 = pb.FullConn(
            self.inp2, self.n4, conn_type=pb.SynConnType.All2All, name="s6"
        )
        self.s7 = pb.FullConn(
            self.n0, self.n3, conn_type=pb.SynConnType.All2All, name="s7"
        )

class fcnet_3(pb.Network):
    def __init__(self):
        super().__init__()
        pe = pb.simulator.PoissonEncoder()
        self.i1 = pb.InputProj(input=pe, shape_out=(2, 5, 5))
        self.n1 = pb.IF((1, 7), threshold=1, reset_v=0, name="n_1")
        self.n2 = pb.IF((1, 5, 5), threshold=1, reset_v=0, name="n_2")
        self.n3 = pb.IF((1, 5, 5), threshold=1, reset_v=0, name="n_3")
        self.n4 = pb.IF((1, 5), threshold=1, reset_v=0, name="n_4")
        self.n5 = pb.IF((1, 3), threshold=1, reset_v=0, name="n_5")
        self.n6 = pb.IF((1, 3), threshold=1, reset_v=0, name="n_6")
        self.n7 = pb.IF((1, 3), threshold=1, reset_v=0, name="n_7")
        
        self.s0 = pb.FullConn(
        self.i1,
        self.n1,
        weights=1,
        conn_type=pb.SynConnType.All2All,
        )
        
        self.s1 = pb.FullConn(
        self.n1,
        self.n2,
        weights=1,
        conn_type=pb.SynConnType.All2All,
        )
        
        self.s2 = pb.FullConn(
        self.n2,
        self.n3,
        weights=1,
        conn_type=pb.SynConnType.All2All,
        )
        
        self.s3 = pb.FullConn(
        self.n1,
        self.n3,
        weights=1,
        conn_type=pb.SynConnType.All2All,
        )
        
        self.s4 = pb.FullConn(
        self.n3,
        self.n4,
        weights=1,
        conn_type=pb.SynConnType.All2All,
        )
        
        self.s5 = pb.FullConn(
        self.n4,
        self.n5,
        weights=1,
        conn_type=pb.SynConnType.All2All,
        )
        
        self.s6 = pb.FullConn(
        self.n3,
        self.n5,
        weights=1,
        conn_type=pb.SynConnType.All2All,
        )
        
        self.s7 = pb.FullConn(
        self.n5,
        self.n6,
        weights=1,
        conn_type=pb.SynConnType.All2All,
        )
        self.s8 = pb.FullConn(
        self.n6,
        self.n7,
        weights=1,
        conn_type=pb.SynConnType.All2All,
        )
        self.s9 = pb.FullConn(
        self.n5,
        self.n7,
        weights=1,
        conn_type=pb.SynConnType.All2All,
        )
net = fcnet_3()
mapper = pb.Mapper()
mapper.clear()
mapper.build(net)
graph_info = mapper.compile(weight_bit_optimization=False, grouping_optim_target="both", no_twisted_branch=True)
# print("OK")
print (pb.__version__)
# self.inp = (400)
# self.x = (1000)
# self.y = (500)
# self.z = (200)
# self.s0 = cnn(400, 1000)
# self.s1 = Mat(x,y)
# full_conn
# s1.bool = True
# s1.List = {250, 250, 250, 250}

# self.s2 = (y,z)

# s0 = cnn(400, 1000) -> s0_0, s0_1, s0_2, s0_3 .... s0_n




# self.x1 = (250)
# self.x2 = (250)
# self.x3 = (250)
# self.x4 = (250)
# self.y1 = (125)
# self.y2 = (125)
# self.y3 = (125)
# self.y4 = (125)
# self.z  = (200)
# self.s1_1 = (x1,y1)
# self.s1_2 = (x2,y2)
# self.s1_3 = (x3,y3)
# self.s1_4 = (x4,y4)
# self.s2_1 = (y1,z)
# self.s2_2 = (y2,z)
# self.s2_3 = (y3,z)
# self.s2_4 = (y4,z)

# self.x(0-249)   (routing_coord_y1, 0-249)
# self.x(250-499) (routing_coord_y2, 0-249)

# CoreBlock routing_coord
# {
#     self.y1 = (250)
#     self.y2 = (250)
#     self.y3
#     self.y4
# }


# Mat(1000, 400)

# -> full(1000, 400){bool=true, List = {250, 250, 250, 250}}

# (Mat, conn, full ... neuron) -> (full) -> (core_config)