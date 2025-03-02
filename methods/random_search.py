from typing import List
import networkx as nx
import scipy.stats as st
import numpy as np

from environments.mdp import sample_mdp, Node

np.random.seed(1)


def random_search(tree: nx.DiGraph, root: Node) -> List[Node]:
    curr: Node = root
    result: List[Node] = [curr]

    while not curr.is_leaf:
        children = list(tree.successors(curr))
        next_idx = np.random.randint(0, len(children))
        curr = children[next_idx]
        result.append(curr)

    return result


# Sample an MDP
BRANCHING_FACTOR = 5
DEPTH = 5
TERMINAL_REWARD_DIST = st.norm(loc=0, scale=2)

tree, root, best_reward = sample_mdp(
    DEPTH, BRANCHING_FACTOR, TERMINAL_REWARD_DIST, seed=123
)
print(f"Tree: {tree}; Best reward: {best_reward:.3f}")

# Run the search
N = 10
res_path: List[Node] = []
res_reward: float = -np.inf

for _ in range(N):
    path = random_search(tree, root)

    if path[-1].reward > res_reward:
        res_reward = path[-1].reward
        res_path = path

print(
    f"[Best-of-{N} Random Search] Path length: {len(res_path)}, Reward: {res_reward:.3f}"
)
