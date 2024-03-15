from typing import ClassVar, Literal, Optional, Tuple, Union

import numpy as np
from paicorelib import HwConfig
from paicorelib import WeightPrecision as WP

from paibox.base import NeuDyn, SynSys
from paibox.exceptions import ShapeError
from paibox.neuron import Neuron
from paibox.projection import InputProj
from paibox.types import DataArrayType, SynOutType, WeightType

from .conv_utils import _fm_ndim2_check
from .transforms import AllToAll, Conv2dForward
from .transforms import GeneralConnType as GConnType
from .transforms import Identity, MaskedLinear, OneToOne, Transform

RIGISTER_MASTER_KEY_FORMAT = "{0}.output"


def _check_equal(num_in: int, num_out: int) -> int:
    if num_in != num_out:
        raise ShapeError(
            f"The number of source & destination neurons must "
            f"be equal, but {num_in} != {num_out}."
        )

    return num_in


class Synapses:
    def __init__(
        self,
        source: Union[NeuDyn, InputProj],
        dest: NeuDyn,
    ) -> None:
        self.source = source
        self.dest = dest

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


class FullConnectedSyn(Synapses, SynSys):
    def __init__(
        self,
        source: Union[NeuDyn, InputProj],
        dest: NeuDyn,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(source, dest)
        super(Synapses, self).__init__(name)

        self.set_memory("_synout", np.zeros((self.num_out,), dtype=np.int32))

        # Register `self` for the destination `NeuDyn`.
        dest.register_master(RIGISTER_MASTER_KEY_FORMAT.format(self.name), self)

    def __call__(self, *args, **kwargs) -> SynOutType:
        return self.update(*args, **kwargs)

    def update(self, spike: Optional[np.ndarray] = None, *args, **kwargs) -> SynOutType:
        # Retrieve the spike at index `timestamp` of the dest neurons
        if self.dest.is_working:
            if isinstance(self.source, InputProj):
                synin = self.source.output.copy() if spike is None else spike
            else:
                idx = self.dest.timestamp % HwConfig.N_TIMESLOT_MAX
                synin = self.source.output[idx].copy() if spike is None else spike
        else:
            # Retrieve 0 to the dest neurons if it is not working
            synin = np.zeros_like(self.source.spike, dtype=np.bool_)

        self._synout = self.comm(synin).astype(np.int32)
        return self._synout

    def reset_state(self, *args, **kwargs) -> None:
        # TODO Add other initialization methods in the future.
        self.reset_memory()  # Call reset of `StatusMemory`.

    def _set_comm(self, comm: Transform) -> None:
        self.comm = comm

    @property
    def output(self) -> SynOutType:
        return self._synout

    @property
    def weights(self) -> WeightType:
        return self.comm.weights

    @property
    def weight_precision(self) -> WP:
        return self.comm._get_wp(self.CFLAG_ENABLE_WP_OPTIMIZATION)

    @property
    def connectivity(self):
        """The connectivity matrix in `np.ndarray` format."""
        return self.comm.connectivity


class FullConnSyn(FullConnectedSyn):
    def __init__(
        self,
        source: Union[NeuDyn, InputProj],
        dest: NeuDyn,
        weights: DataArrayType,
        conn_type: GConnType,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(source, dest, name)

        if conn_type is GConnType.One2One:
            comm = OneToOne(_check_equal(self.num_in, self.num_out), weights)
        elif conn_type is GConnType.Identity:
            if not isinstance(weights, (int, np.bool_, np.integer)):
                raise TypeError(
                    f"Expected type int, np.bool_, np.integer, but got type {type(weights)}"
                )
            comm = Identity(_check_equal(self.num_in, self.num_out), weights)
        elif conn_type is GConnType.All2All:
            comm = AllToAll((self.num_in, self.num_out), weights)
        else:  # MatConn
            if not isinstance(weights, np.ndarray):
                raise TypeError(
                    f"Expected type np.ndarray, but got type {type(weights)}"
                )
            comm = MaskedLinear((self.num_in, self.num_out), weights)

        self._set_comm(comm)


class Conv2dSyn(FullConnectedSyn):
    _spatial_ndim: ClassVar[int] = 2

    def __init__(
        self,
        source: Union[Neuron, InputProj],
        dest: Neuron,
        kernel: np.ndarray,
        stride: Tuple[int, int],
        # padding: Tuple[int, int],
        fm_order: Literal["CHW", "HWC"],
        order: Literal["OIHW", "IOHW"],
        name: Optional[str] = None,
    ) -> None:
        super().__init__(source, dest, name)

        if kernel.ndim != self._spatial_ndim + 2:
            raise ShapeError(
                f"The convolution kernel dimension must be {self._spatial_ndim + 2}, but got {kernel.ndim}."
            )

        if order == "IOHW":
            _kernel = kernel.transpose(1, 0, 2, 3)
        else:
            _kernel = kernel.copy()

        # O,I,H,W
        out_channels, in_channels, kernel_h, kernel_w = _kernel.shape

        # C,H,W
        in_ch, in_h, in_w = _fm_ndim2_check(source._shape, fm_order)
        out_ch, out_h, out_w = _fm_ndim2_check(dest._shape, fm_order)

        if in_ch != in_channels:
            raise ShapeError(f"Input channels mismatch: {in_ch} != {in_channels}")

        if out_ch != out_channels:
            raise ShapeError(f"Output channels mismatch: {out_ch} != {out_channels}")

        # If padding is considered, the implementation of convolution unrolling
        # is extremely complex, so fix it.
        padding = (0, 0)

        assert (in_h + 2 * padding[0] - kernel_h) // stride[0] + 1 == out_h
        assert (in_w + 2 * padding[1] - kernel_w) // stride[1] + 1 == out_w

        comm = Conv2dForward(
            (in_h, in_w), (out_h, out_w), _kernel, stride, padding, fm_order
        )

        self._set_comm(comm)
