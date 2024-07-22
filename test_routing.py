from paibox.backend.routing import RoutingRoot
from paibox.backend.mapper  import Mapper
from paicorelib import Coord, HwConfig
from paicorelib.routing_defs import ROUTING_DIRECTIONS_IDX, RoutingCoord, RoutingCost
from paicorelib.routing_defs import RoutingDirection as Direction
from paicorelib.routing_defs import RoutingLevel as Level
from paicorelib.routing_defs import RoutingStatus as Status
from paicorelib.routing_defs import get_routing_consumption
from typing import List
from paibox.backend.context import _BACKEND_CONTEXT
from paibox.backend.mapper import _calculate_core_consumption, reorder_routing_groups
class RoutingGroup(int):
    """Core blocks located within a routing group are routable.

    NOTE: Axon groups within a routing group are the same.
    """

    def __init__(self, num_core) -> None:
        self.n_core_required = num_core
        self.assigned_coords: List[Coord] = []
        """Assigned core coordinates in the routing group"""
        self.wasted_coords: List[Coord] = []
        """Wasted core coordinates in routing group"""
        
    @property
    def routing_cost(self) -> RoutingCost:
        return get_routing_consumption(self.n_core_required)

    @property
    def routing_level(self) -> Level:
        return self.routing_cost.get_routing_level()
    
    def assign(self, assigned: List[Coord], wasted: List[Coord], chip_coord: Coord) -> None:
        return

_BACKEND_CONTEXT["target_chip_addr"] = [Coord(0, 0), Coord(0,1)]
mapper = Mapper()
routing_tree = mapper.routing_tree
required_cores = [1] * 1009
# required_cores = [1, 16, 64]
routing_groups = []
for core_num in required_cores:
    routing_groups.append(RoutingGroup(core_num))

# graph = {
#     routing_groups[0]: [],
#     routing_groups[1]: [routing_groups[0]],
#     routing_groups[2]: [routing_groups[1]],
#     routing_groups[3]: [routing_groups[1]],
#     routing_groups[4]: [routing_groups[2], routing_groups[3]],
# }
# routing_groups = reorder_routing_groups(graph)
# n_core_required = _calculate_core_consumption(routing_groups)
# print(f"Total core required: {n_core_required}")
i = 0
for rg in routing_groups:
    print("Routing Group[{}] with {} cores".format(i, rg.n_core_required))
    routing_tree.place_routing_group(rg)
    i += 1
    print()
print("Used L2 Coords:")
for index, routing_coord_list in enumerate(routing_tree.used_L2_coords):
    print("Chip[{}]:".format(index))
    for routing_coord in routing_coord_list:
        print(routing_coord)