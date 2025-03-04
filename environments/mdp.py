from typing import Optional, Tuple
import networkx as nx
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
) -> Tuple[nx.DiGraph, Node, float]:
    """Sample a tree from the Markov decision process.

    Args:
        depth: The depth of the tree.
        branching_factor: The branching factor of the tree.
        terminal_reward_dist: The distribution of rewards at the leaf nodes.
        intermediate_reward_dist: The distribution of rewards at internal, non-leaf nodes.

    Returns:
        tree: A `networkx` tree where each node is an object of `Node`. The reward
            at each node is sampled from the specified reward distribution(s).
        root: A `Node` object which is the root of `tree`.
        best_reward: The maximum total reward of the tree. I.e., the reward associated with
            the optimal path in the tree.
    """
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
    best_reward = best_total_reward(tree, root)

    return tree, root, best_reward


def best_total_reward(tree: nx.DiGraph, curr: Node) -> float:
    """Find the best total reward (over paths) in the tree.
    This is done through an exhaustive search!

    Args:
        tree: The tree search space.
        curr: The starting point of the search (e.g. root).

    Returns:
        best_reward: The optimal reward.
    """
    if curr.is_leaf:
        return curr.reward

    return curr.reward + max(
        best_total_reward(tree, child) for child in tree.successors(curr)
    )
