<<<<<<< HEAD
from .base import (
    Conv2dSemiFoldedSyn,
    FullConnectedSyn,
    FullConnSyn,
    MaxPool2dSemiFoldedSyn,
)
=======
from .base import Conv2dHalfRollSyn, FullConnectedSyn, FullConnSyn, MaxPool2dSemiMapSyn, EdgeSlice
>>>>>>> implement of edge_group with bug
from .transforms import ConnType
from .synapses import MatMul2d
