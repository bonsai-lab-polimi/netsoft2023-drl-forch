import heapq
import json

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
from gym.utils import seeding

from envs.utils import Request


class FogOrchestrationEnv(gym.Env):
    def __init__(
        self,
        n_nodes: int,
        arrival_rate_r: float,
        call_duration_r: float,
        episode_length: int = 100,
        seed: int = 42,
    ) -> None:

        # used to build an observation, defined as a matrix having as rows the nodes and columns their associated metrics
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_nodes + 1, 12), dtype=np.float32)

        # deploy the service on the 1,2,..., n node or drop it
        self.action_space = spaces.Discrete(n_nodes + 1)
        self.episode_length = episode_length

        self.n_nodes = n_nodes
        self.arrival_rate_r = arrival_rate_r
        self.call_duration_r = call_duration_r
        self.running_requests: list[Request] = []

        self.np_random, seed = seeding.np_random(seed)

        # creating the array that specifies if a node is fixed (0) or is not (1)
        self.node_type = self.np_random.randint(low=0, high=2, size=n_nodes)

        # setting the initial scenario
        self.idx_fixed = self.node_type == 0
        self.cpu = self.np_random.uniform(low=0.0, high=0.2, size=n_nodes)
        self.cpu[self.idx_fixed] = 0.0
        self.ram = self.np_random.uniform(low=0.0, high=0.2, size=n_nodes)
        self.ram[self.idx_fixed] = 0.0
        self.disk = self.np_random.uniform(low=0.0, high=0.2, size=n_nodes)
        self.disk[self.idx_fixed] = 0.0
        self.load = self.np_random.uniform(low=0.0, high=0.2, size=n_nodes)
        self.load[self.idx_fixed] = 0.0
        self.lat_val = self.np_random.randint(low=4, high=10, size=n_nodes)
        self.lat_val[self.idx_fixed] = self.np_random.randint(low=0, high=5, size=self.lat_val[self.idx_fixed].shape)

        # creating the array containing the type of services dewployable in the nodes
        self.serv_type = self.np_random.randint(low=0, high=4, size=n_nodes)

        # filling the service list randomly and loading it
        # self.service_list_generator()
        # with open("./envs/services.json") as f:
        #    self.service_list = json.load(f)

        self.current_time = 0
        self.t_ = 1

        self.accepted_requests = 0
        self.offered_requests = 0
        self.ep_accepted_requests = 0

        self.next_request()

    def step(self, action) -> tuple:
        self.offered_requests += 1
        # resetting the variable for blocking a request
        self.penalty = False
        lat_penalty = np.nan
        if action < self.n_nodes:
            if self.node_is_full(action):
                self.penalty = True
                print(f"Blocking: node {action} is full")
                raise ValueError("Action mask is not working properly. Full nodes should be always masked.")
            else:
                self.accepted_requests += 1
                self.ep_accepted_requests += 1
                self.request.serving_node = action
                self.cpu[action] += self.request.cpu
                self.ram[action] += self.request.ram
                self.disk[action] += self.request.disk
                self.load[action] += self.request.load
                self.enqueue_request(self.request)
                lat_penalty = max(0, self.lat_val[action] - self.request.latency)
        else:
            self.penalty = True

        info = {
            # "cpu": self.cpu,
            # "ram": self.ram,
            # "disk": self.disk,
            # "load": self.load,
            # "idx_fixed": self.idx_fixed,
            "mean_cpu_fixed": np.mean(self.cpu[self.idx_fixed]),
            "std_cpu_fixed": np.std(self.cpu[self.idx_fixed], ddof=1),
            "mean_ram_fixed": np.mean(self.ram[self.idx_fixed]),
            "std_ram_fixed": np.std(self.ram[self.idx_fixed], ddof=1),
            "mean_disk_fixed": np.mean(self.disk[self.idx_fixed]),
            "std_disk_fixed": np.std(self.disk[self.idx_fixed], ddof=1),
            "mean_load_fixed": np.mean(self.load[self.idx_fixed]),
            "std_load_fixed": np.std(self.load[self.idx_fixed], ddof=1),
            "mean_cpu_mobile": np.mean(self.cpu[~self.idx_fixed]),
            "std_cpu_mobile": np.std(self.cpu[~self.idx_fixed], ddof=1),
            "mean_ram_mobile": np.mean(self.ram[~self.idx_fixed]),
            "std_ram_mobile": np.std(self.ram[~self.idx_fixed], ddof=1),
            "mean_disk_mobile": np.mean(self.disk[~self.idx_fixed]),
            "std_disk_mobile": np.std(self.disk[~self.idx_fixed], ddof=1),
            "mean_load_mobile": np.mean(self.load[~self.idx_fixed]),
            "std_load_mobile": np.std(self.load[~self.idx_fixed], ddof=1),
            "mean_lat_fixed": np.mean(self.lat_val[self.idx_fixed]),
            "std_lat_fixed": np.std(self.lat_val[self.idx_fixed], ddof=1),
            "mean_lat_mobile": np.mean(self.lat_val[~self.idx_fixed]),
            "std_lat_mobile": np.std(self.lat_val[~self.idx_fixed], ddof=1),
            "request_cpu": self.request.cpu,
            "request_ram": self.request.ram,
            "request_disk": self.request.disk,
            "request_load": self.request.load,
            "request_latency": self.request.latency,
            "requests_serving": len(self.running_requests),
            "action": action,
            "block_prob": 1 - (self.accepted_requests / self.offered_requests),
            "ep_block_prob": 1 - (self.ep_accepted_requests / self.t_),
            "lat_penalty": lat_penalty,
        }

        self.next_request()

        observation = self.observation()

        # reward function
        if self.penalty:
            if not self.eve_full():
                reward = -1.0
            else:
                reward = -0.5
        else:
            if self.request.latency >= self.lat_val[action]:
                reward = 1
            else:
                reward = (self.lat_val[action] - self.request.latency) / self.lat_val[action]

        if self.t_ == self.episode_length:
            done = True
        else:
            done = False

        self.upt_latency()

        self.t_ += 1

        return observation, reward / self.episode_length, done, info

    def reset(self) -> npt.NDArray:
        self.ep_accepted_requests = 0
        self.t_ = 1
        return self.observation()

    def observation(self) -> npt.NDArray:
        cloud_node = np.full((1, 6), -1)
        observation = np.stack([self.cpu, self.ram, self.disk, self.load, self.node_type, self.lat_val], axis=1)
        # Condition the elements in the set with the current node request
        node_demand = np.tile(
            np.array(
                [self.request.cpu, self.request.ram, self.request.disk, self.request.load, self.request.latency, self.dt]
            ),
            (self.n_nodes + 1, 1),
        )
        observation = np.concatenate([observation, cloud_node], axis=0)
        observation = np.concatenate([observation, node_demand], axis=1)
        return observation

    def enqueue_request(self, request: Request) -> None:
        heapq.heappush(self.running_requests, (request.departure_time, request))

    def action_masks(self):
        valid_actions = np.ones(self.n_nodes + 1, dtype=bool)
        for i in range(self.n_nodes):
            if self.serv_type[i] == 0 or self.serv_type[i] == self.request.service_type:
                if self.node_is_full(i):
                    valid_actions[i] = False
                else:
                    valid_actions[i] = True
            else:
                valid_actions[i] = False
        valid_actions[self.n_nodes] = True
        return valid_actions

    def dequeue_request(self):
        _, request = heapq.heappop(self.running_requests)
        self.cpu[request.serving_node] -= request.cpu
        self.ram[request.serving_node] -= request.ram
        self.disk[request.serving_node] -= request.disk
        self.load[request.serving_node] -= request.load

    def is_deployable_on(self, action) -> bool:
        if self.serv_type[action] == 0 or self.serv_type[action] == self.request.service_type:
            return True
        return False

    def upt_latency(self):
        self.lat_val[~self.idx_fixed] = self.np_random.randint(low=4, high=10, size=self.lat_val[~self.idx_fixed].shape)
        self.lat_val[self.idx_fixed] = self.np_random.randint(low=0, high=5, size=self.lat_val[self.idx_fixed].shape)

    def node_is_full(self, action) -> bool:
        if self.node_type[action] == 1:
            if (
                self.cpu[action] + self.request.cpu > 0.5
                or self.ram[action] + self.request.ram > 0.5
                or self.disk[action] + self.request.disk > 0.5
                or self.load[action] + self.request.load > 0.5
            ):
                return True
        else:
            if (
                self.cpu[action] + self.request.cpu > 0.95
                or self.ram[action] + self.request.ram > 0.95
                or self.disk[action] + self.request.disk > 0.95
                or self.load[action] + self.request.load > 0.95
            ):
                return True
        return False

    def eve_full(self) -> bool:
        is_full = [self.node_is_full(i) for i in range(self.n_nodes)]
        return np.all(is_full)

    def service_list_generator(self, n_services: int = 1000) -> None:
        d = {"services": []}
        for _ in range(n_services):
            d["services"].append(self.service_generator())
        json_object = json.dumps(d, indent=1)

        with open("./envs/services.json", "w") as f:
            f.write(json_object)

    def service_generator(self) -> dict:
        n = self.np_random.uniform(low=0, high=1)
        d = {"id": None, "cpu": None, "ram": None, "disk": None, "load": None, "latency": None, "base": ""}
        if n < 0.1:
            d["id"] = "FVE" + str(self.np_random.randint(low=1, high=101))
            d["cpu"] = self.np_random.uniform(low=0.15, high=0.3)
            d["ram"] = self.np_random.uniform(low=0.15, high=0.3)
            d["disk"] = self.np_random.uniform(low=0.15, high=0.3)
            d["load"] = self.np_random.uniform(low=0.15, high=0.3)
            d["latency"] = self.np_random.randint(low=3, high=8)
        elif n < 0.3:
            d["id"] = "SDP" + str(self.np_random.randint(low=1, high=101))
            d["cpu"] = self.np_random.uniform(low=0.1, high=0.2)
            d["ram"] = self.np_random.uniform(low=0.1, high=0.2)
            d["disk"] = self.np_random.uniform(low=0.1, high=0.2)
            d["load"] = self.np_random.uniform(low=0.1, high=0.2)
            d["latency"] = self.np_random.randint(low=2, high=6)
            d["base"] = "FVE000"
        elif n < 0.6:
            d["id"] = "APP" + str(self.np_random.randint(low=1, high=101))
            d["cpu"] = self.np_random.uniform(low=0.01, high=0.1)
            d["ram"] = self.np_random.uniform(low=0.01, high=0.1)
            d["disk"] = self.np_random.uniform(low=0.01, high=0.1)
            d["load"] = self.np_random.uniform(low=0.01, high=0.1)
            d["latency"] = self.np_random.randint(low=0, high=3)
            d["base"] = "FVE000"
        else:
            d["id"] = "LAF" + str(self.np_random.randint(low=1, high=101))
            d["cpu"] = self.np_random.uniform(low=0.01, high=0.02)
            d["ram"] = self.np_random.uniform(low=0.01, high=0.02)
            d["disk"] = self.np_random.uniform(low=0.01, high=0.02)
            d["load"] = self.np_random.uniform(low=0.01, high=0.02)
            d["latency"] = self.np_random.randint(low=0, high=3)
            d["base"] = "FVE000"
        return d

    def next_request(self) -> None:
        arrival_time = self.current_time + self.np_random.exponential(scale=1 / self.arrival_rate_r)
        departure_time = arrival_time + self.np_random.exponential(scale=self.call_duration_r)
        self.dt = departure_time - arrival_time
        self.current_time = arrival_time

        while True:
            if self.running_requests:
                next_departure_time, _ = self.running_requests[0]
                if next_departure_time < arrival_time:
                    self.dequeue_request()
                    continue
            break

        new_service = self.service_generator()

        if "FVE" in new_service["id"]:
            self.request = Request(
                cpu=new_service["cpu"] + self.np_random.uniform(low=-0.05, high=0.05),
                ram=new_service["ram"] + self.np_random.uniform(low=-0.05, high=0.05),
                disk=new_service["disk"] + self.np_random.uniform(low=-0.05, high=0.05),
                load=new_service["load"] + self.np_random.uniform(low=-0.05, high=0.05),
                arrival_time=arrival_time,
                departure_time=departure_time,
                latency=new_service["latency"],
                service_type=0,
            )

        elif "SDP" in new_service["id"]:
            self.request = Request(
                cpu=new_service["cpu"] + self.np_random.uniform(low=-0.025, high=0.025),
                ram=new_service["ram"] + self.np_random.uniform(low=-0.025, high=0.025),
                disk=new_service["disk"] + self.np_random.uniform(low=-0.025, high=0.025),
                load=new_service["load"] + self.np_random.uniform(low=-0.025, high=0.025),
                arrival_time=arrival_time,
                departure_time=departure_time,
                latency=new_service["latency"],
                service_type=1,
            )

        elif "APP" in new_service["id"]:
            self.request = Request(
                cpu=new_service["cpu"] + self.np_random.uniform(low=-0.005, high=0.005),
                ram=new_service["ram"] + self.np_random.uniform(low=-0.005, high=0.005),
                disk=new_service["disk"] + self.np_random.uniform(low=-0.005, high=0.005),
                load=new_service["load"] + self.np_random.uniform(low=-0.005, high=0.005),
                arrival_time=arrival_time,
                departure_time=departure_time,
                latency=new_service["latency"],
                service_type=2,
            )

        elif "LAF" in new_service["id"]:
            self.request = Request(
                cpu=new_service["cpu"] + self.np_random.uniform(low=-0.0025, high=0.0025),
                ram=new_service["ram"] + self.np_random.uniform(low=-0.0025, high=0.0025),
                disk=new_service["disk"] + self.np_random.uniform(low=-0.0025, high=0.0025),
                load=new_service["load"] + self.np_random.uniform(low=-0.0025, high=0.0025),
                arrival_time=arrival_time,
                departure_time=departure_time,
                latency=new_service["latency"],
                service_type=3,
            )
