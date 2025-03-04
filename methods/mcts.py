from dataclasses import dataclass
import math
from typing import List, Tuple
import networkx as nx
import scipy.stats as st
import numpy as np

from environments.mdp import sample_mdp, Node

np.random.seed(123)


@dataclass
class MCTSNode:
    real_node: Node
    mean_val: float = 0.0
    n_visits: int = 1

    def __hash__(self) -> int:
        return hash(self.real_node.name)


def mcts(
    tree: nx.DiGraph, root: Node, n_iters: int = 50, gamma: float = math.sqrt(2)
) -> Tuple[List[Node], float]:
    curr: Node = root
    paths: List[List[Node]] = []
    rewards: List[float] = []

    mcts_tree = nx.DiGraph()
    mcts_root = MCTSNode(real_node=root)
    mcts_tree.add_node(mcts_root)

    def select(node: MCTSNode) -> MCTSNode:
        if mcts_tree.out_degree(node) == 0:
            # Frontier node
            node.n_visits += 1
            return node

        best_node = node
        best_uct = -np.inf

        for child in mcts_tree.successors(node):
            uct = child.mean_val + gamma * math.sqrt(
                math.log(node.n_visits) / child.n_visits
            )

            if uct > best_uct:
                best_uct = uct
                best_node = child

        return select(best_node)

    def expand(node: MCTSNode):
        children = tree.successors(node.real_node)

        for child in children:
            mcts_child = MCTSNode(child)
            mcts_tree.add_node(mcts_child)
            mcts_tree.add_edge(node, mcts_child)

    def rollout(mcts_node: MCTSNode) -> float:
        return 0

    def backup(node: MCTSNode, reward: float):
        pass

    for _ in range(n_iters):
        node = select(mcts_root)
        expand(node)
        # reward = rollout(node)
        # backup(node, reward)

    best_idx = np.argmax(rewards)

    return paths[best_idx], rewards[best_idx]


# Sample an MDP
BRANCHING_FACTOR = 10
DEPTH = 5
REWARD_DIST = st.norm(loc=0, scale=2)

# The MDP has intermediate rewards
tree, root, best_reward = sample_mdp(
    DEPTH,
    BRANCHING_FACTOR,
    terminal_reward_dist=REWARD_DIST,
)
print(f"Tree: {tree}; Best reward: {best_reward:.3f}")

# Run the search
res_path, res_reward = mcts(tree, root)

print(f"[MCTS] Path length: {len(res_path)}, Reward: {res_reward:.3f}")
