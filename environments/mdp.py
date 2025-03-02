from typing import Optional, Tuple
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
    intermediate_reward_dist: Optional[rv_frozen] = None,
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

                if child_is_leaf:
                    child_reward = float(terminal_reward_dist.rvs())
                elif intermediate_reward_dist is not None:
                    child_reward = float(intermediate_reward_dist.rvs())
                else:
                    child_reward = 0

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
    # best_reward = max(node.reward for node in tree.nodes)
    best_reward = best_total_reward(tree, root)

    return tree, root, best_reward


def best_total_reward(tree: nx.DiGraph, curr: Node) -> float:
    if curr.is_leaf:
        return curr.reward

    return curr.reward + max(
        best_total_reward(tree, child) for child in tree.successors(curr)
    )
