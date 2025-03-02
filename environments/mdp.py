from typing import Tuple
import networkx as nx
import numpy as np
from dataclasses import dataclass

from scipy.stats.distributions import rv_frozen


ROOT_NAME = "0"


@dataclass
class Node:
    name: str
    depth: int
    is_leaf: bool
    reward: float

    def __hash__(self) -> int:
        return hash(self.name)


def sample_mdp(
    depth: int,
    branching_factor: int,
    terminal_reward_dist: rv_frozen,
    seed: int = 1,
) -> Tuple[nx.DiGraph, Node, float]:
    np.random.seed(seed)

    tree = nx.DiGraph()
    root = Node(name=ROOT_NAME, depth=0, is_leaf=False, reward=0)
    tree.add_node(root)

    def build_tree(curr: Node) -> None:
        if curr.is_leaf:
            return

        if len(list(tree.successors(curr))) == 0:
            # Expand
            for i in range(branching_factor):
                child_name = f"{curr.name}*{i}"
                child_depth = curr.depth + 1
                child_is_leaf = child_depth == depth
                child_reward = float(terminal_reward_dist.rvs()) if child_is_leaf else 0

                child = Node(
                    name=child_name,
                    depth=child_depth,
                    is_leaf=child_is_leaf,
                    reward=child_reward,
                )

                tree.add_node(child)
                tree.add_edge(curr, child)

                # DFS
                build_tree(child)

    build_tree(root)
    best_reward = max(node.reward for node in tree.nodes)

    return tree, root, best_reward
