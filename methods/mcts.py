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
    total_val: float = 0.0
    n_visits: int = 0

    def __hash__(self) -> int:
        return hash(self.real_node.name)


def mcts(
    tree: nx.DiGraph,
    root: Node,
    n_iters: int = 5000,
    gamma: float = math.sqrt(2),
    max_children: int = 5,
) -> Tuple[List[Node], float]:
    assert tree.out_degree(root) >= max_children, (  # pyright: ignore
        "Max children must be <= branching factor of the real tree"
    )

    mcts_tree = nx.DiGraph()
    mcts_root = MCTSNode(real_node=root)
    mcts_tree.add_node(mcts_root)

    def select(node: MCTSNode) -> MCTSNode:
        n_children = len(list(mcts_tree.successors(node)))

        # Limit exploration so that the search progresses depth-wise.
        # Otherwise, if the branching factor is large, it will stuck
        # at a particular level.
        if n_children < max_children:
            # Not fully expanded => select this
            return node

        # Otherwise, it's fully expanded and pick a child with best UCT
        best_node = node
        best_uct = -np.inf

        for child in mcts_tree.successors(node):
            # Always visit an unvisited node at least once
            if child.n_visits == 0:
                uct = np.inf
            else:
                uct = (child.total_val / child.n_visits) + gamma * math.sqrt(
                    math.log(node.n_visits) / child.n_visits
                )

            if uct > best_uct:
                best_uct = uct
                best_node = child

        return select(best_node)

    def expand(node: MCTSNode) -> MCTSNode:
        children = list(tree.successors(node.real_node))
        expanded_children = {child.real_node for child in mcts_tree.successors(node)}

        for child in children:
            if child not in expanded_children:
                mcts_child = MCTSNode(child)
                mcts_tree.add_node(mcts_child)
                mcts_tree.add_edge(node, mcts_child)
                return mcts_child

        return node

    def rollout(node: MCTSNode) -> float:
        real_node = node.real_node

        while not real_node.is_leaf:
            real_node = np.random.choice(list(tree.successors(real_node)))

        return real_node.reward

    def backup(node: MCTSNode, reward: float):
        node.total_val += reward
        node.n_visits += 1

        if mcts_tree.in_degree(node) != 0:
            backup(list(mcts_tree.predecessors(node))[0], reward)

    # Do search
    for _ in range(n_iters):
        node = select(mcts_root)
        node = expand(node)
        reward = rollout(node)
        backup(node, reward)

    # Get best path
    curr: MCTSNode = mcts_root
    path: List[Node] = [curr.real_node]

    while True:
        children = list(mcts_tree.successors(curr))

        if len(children) == 0:
            break

        curr = max(children, key=lambda child: child.total_val / child.n_visits)
        path.append(curr.real_node)

    return path, path[-1].reward


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
n_iters = 10000
res_path, res_reward = mcts(
    tree,
    root,
    n_iters=n_iters,
    max_children=5,
)
print(
    f"[MCTS n_iters={n_iters}] Path length: {len(res_path)}, Reward: {res_reward:.3f}"
)
