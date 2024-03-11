from typing import Optional

from paicorelib import LCM, LDM, LIM, NTM, RM, SIM,MaxPoolingEnable, SpikeWidthFormat, SNNModeEnable

from paibox.types import Shape

from .base import Neuron


class ReLu(Neuron):
    """ReLu neuron"""

    def __init__(
        self,
        shape: Shape,
        threshold: int = 0,
        bias: int = 0,
        bit_trunc = 8,
        *,
        delay: int = 1,
        tick_wait_start: int = 1,
        tick_wait_end: int = 0,
        unrolling_factor: int = 1,
        keep_shape: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """

        """
        _bias = bias
        _sim = SIM.MODE_DETERMINISTIC
        _lim = LIM.MODE_DETERMINISTIC
        _ld = LDM.MODE_FORWARD
        _lc = LCM.LEAK_BEFORE_COMP
        _pos_thres = threshold
        _reset_v = 0
        _ntm = NTM.MODE_SATURATION
        _reset_mode = RM.MODE_NONRESET

        super().__init__(
            shape,
            _reset_mode,
            _reset_v,
            _lc,
            0,
            _ntm,
            0,
            _pos_thres,
            _ld,
            _lim,
            _bias,
            _sim,
            bit_trunc,
            SpikeWidthFormat.WIDTH_8BIT,
            MaxPoolingEnable.DISABLE,
            SNNModeEnable.DISABLE,
            delay=delay,
            tick_wait_start=tick_wait_start,
            tick_wait_end=tick_wait_end,
            unrolling_factor=unrolling_factor,
            keep_shape=keep_shape,
            name=name,
        )


class MaxPooling(Neuron):

    def __init__(
            self,
            shape: Shape,
            kernel_size: int = 1,
            stride: int = 1,
            threshold: int = 0,
            *,
            delay: int = 1,
            tick_wait_start: int = 1,
            tick_wait_end: int = 0,
            unrolling_factor: int = 1,
            keep_shape: bool = False,
            name: Optional[str] = None,
    ) -> None:
        """

        """
        self.kernel_size = kernel_size
        self.stride = stride
        _bias = 0
        _sim = SIM.MODE_DETERMINISTIC
        _lim = LIM.MODE_DETERMINISTIC
        _ld = LDM.MODE_FORWARD
        _lc = LCM.LEAK_BEFORE_COMP
        _pos_thres = threshold
        _reset_v = 0
        _ntm = NTM.MODE_SATURATION
        _reset_mode = RM.MODE_NONRESET

        super().__init__(
            shape,
            _reset_mode,
            _reset_v,
            _lc,
            0,
            _ntm,
            0,
            _pos_thres,
            _ld,
            _lim,
            0,
            _sim,
            0,
            SpikeWidthFormat.WIDTH_8BIT,
            MaxPoolingEnable.ENABLE,
            SNNModeEnable.DISABLE,
            delay=delay,
            tick_wait_start=tick_wait_start,
            tick_wait_end=tick_wait_end,
            unrolling_factor=unrolling_factor,
            keep_shape=keep_shape,
            name=name,
        )