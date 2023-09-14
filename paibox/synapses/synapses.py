from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from paibox.base import NeuDyn, DynamicSys
from paibox.projection import InputProj

from .connector import All2All, IndexConn, MatConn, One2One, TwoEndConnector
from .transforms import AllToAll, MaskedLinear, OneToOne

__all__ = ["Synapses", "SynSys", "NoDecay"]


class Synapses:
    """A map connected between neurons of the previous `Node`, and axons of the following `Node`.

    User can use connectivity matrix or COO to represent the connectivity of synapses.
    """

    def __init__(
        self,
        source: Union[NeuDyn, InputProj],
        dest: NeuDyn,
        conn: Union[
            TwoEndConnector, np.ndarray, Dict[str, Union[List[int], np.ndarray]]
        ],
    ) -> None:
        """
        Arguments:
            - source: the source group of neurons.
            - dest: the destination group of neurons.
            - conn: the connectivity representation.
            - name: the name of the synapses. Optional.
        """
        self.source = source
        self.dest = dest
        # TODO conn, what for?
        self.conn = self._init_conn(conn)

    def _init_conn(
        self,
        conn: Union[
            TwoEndConnector, np.ndarray, Dict[str, Union[List[int], np.ndarray]]
        ],
    ) -> Union[TwoEndConnector, MatConn, IndexConn]:
        """Build a connector given the arrays or dictionary."""
        if isinstance(conn, TwoEndConnector):
            return conn(self.num_in, self.num_out)

        if isinstance(conn, np.ndarray):
            conn = MatConn(conn_mat=conn)
        elif isinstance(conn, Dict):
            if not ("i" in conn and "j" in conn):
                raise ValueError("The keys of the dictionary must include 'i' and 'j'.")
            conn = IndexConn(source_ids=conn["i"], dest_ids=conn["j"])
        else:
            raise TypeError(f"Unsupported type: {type(self.conn)}.")

        return conn

    @property
    def shape_in(self) -> Tuple[int, ...]:
        return self.source.shape_out

    @property
    def shape_out(self) -> Tuple[int, ...]:
        return self.dest.shape_in

    @property
    def num_in(self) -> int:
        return self.source.num_out

    @property
    def num_out(self) -> int:
        return self.dest.num_in

    @property
    def num_axon(self) -> int:
        return self.num_in
    
    @property
    def num_dentrite(self) -> int:
        return self.num_out


class SynSys(Synapses, DynamicSys):
    @property
    def connectivity(self) -> np.ndarray:
        raise NotImplementedError


class NoDecay(SynSys):
    """Synapses model with no decay.

    Attributes:
        - comm: the connectivity representation.

    Properties:
        - connectivity: the `np.ndarray` format of comm.
    """

    def __init__(
        self,
        source: Union[NeuDyn, InputProj],
        dest: NeuDyn,
        conn: Union[
            TwoEndConnector, np.ndarray, Dict[str, Union[List[int], np.ndarray]]
        ],
        weights: Union[int, np.integer, np.ndarray] = 1,
        *,
        name: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            - source: source neuron(s).
            - dest: destination neuron(s).
            - conn: connectivity representation.
            - weights: weights of the synapses. It can be an integer or `np.ndarray`.
            - name: name of this synapses. Optional.
        """
        super().__init__(source, dest, conn)
        super(Synapses, self).__init__(name)

        if isinstance(conn, All2All):
            self.comm = AllToAll(self.num_in, self.num_out, weights)
        elif isinstance(conn, One2One):
            self.comm = OneToOne(self.num_in, weights)
        elif isinstance(conn, MatConn):
            self.comm = MaskedLinear(conn, weights)
        else:
            # TODO Error description
            raise ValueError

        self.reset()

        # Register `self` for the destination NeuDyn.
        dest.register_master(f"{self.name}.output", self)

    def __call__(self, spike: Optional[np.ndarray] = None, **kwargs):
        return self.update(spike, **kwargs)

    def update(self, spike: Optional[np.ndarray] = None, **kwargs):
        if spike is None:
            synin = self.source.output
        else:
            synin = spike

        self._synout = self.comm(synin)

        # TODO Keep the return for a while.
        return self._synout

    def reset(self, init_val=0) -> None:
        # TODO Add initialization methods
        # Only constant initialization right now.
        self._synout = np.zeros((self.num_in, self.num_out), dtype=np.int8)

    @property
    def output(self) -> np.ndarray:
        return self._synout

    @property
    def state(self) -> np.ndarray:
        return self._synout

    @property
    def connectivity(self) -> np.ndarray:
        return self.comm.connectivity

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return (
            f"{name}(name={self.name}, \n"
            f'{" " * len(name)} source={self.source}, \n'
            f'{" " * len(name)} dest={self.dest})'
        )
