from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class Request:
    cpu: float
    ram: float
    disk: float
    load: float
    arrival_time: float
    departure_time: float
    latency: int
    serving_node: int = None
    service_type: int = None  # currently: 0==SVE 1==SDP 2==APP 3==LAF


def greedy_qos_policy(action_mask: npt.NDArray, lat_val: npt.NDArray) -> int:
    """Returns the index of the feasible node with the lowest latency.
    This policy does not take into account QoS requirements."""
    feasible_nodes = np.argwhere(action_mask[:-1] == True).flatten()
    if len(feasible_nodes) == 0:
        return len(action_mask)
    return feasible_nodes[np.argmin(lat_val[feasible_nodes])]


def greedy_lb_qos_policy(obs: npt.NDArray, action_mask: npt.NDArray, lat_val: npt.NDArray, service_lat: float) -> int:
    """Returns the index of the feasible node with the lowest average weighted load
    between cpu, ram, disk and bandwidth, and for which the latency is lower than the
    QoS service requirement."""
    node_mask = np.logical_and(action_mask[:-1], lat_val <= service_lat)
    feasible_nodes = np.argwhere(node_mask == True).flatten()
    # cpu, ram, disk and bandwidth are the first 4 columns of the observation matrix
    mean_load = np.mean(obs[feasible_nodes, :4], axis=1).flatten()
    if len(feasible_nodes) == 0:
        return len(action_mask)
    return feasible_nodes[np.argmin(mean_load)]


def greedy_lb_policy(obs: npt.NDArray, action_mask: npt.NDArray) -> int:
    """Returns the index of the feasible node with the lowest average weighted load
    between cpu, ram, disk and bandwidth. It does not consider QoS latency requirements."""
    feasible_nodes = np.argwhere(action_mask[:-1] == True).flatten()
    # cpu, ram, disk and bandwidth are the first 4 columns of the observation matrix
    mean_load = np.mean(obs[feasible_nodes, :4], axis=1).flatten()
    if len(feasible_nodes) == 0:
        return len(action_mask)
    return feasible_nodes[np.argmin(mean_load)]
