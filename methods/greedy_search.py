from typing import List
import networkx as nx
import scipy.stats as st
import numpy as np

from environments.mdp import sample_mdp, Node

np.random.seed(1)


def greedy_search(tree: nx.DiGraph, root: Node) -> List[Node]:
    curr: Node = root
    result: List[Node] = [curr]

    while not curr.is_leaf:
        children = list(tree.successors(curr))
        next_idx = np.argmax([child.reward for child in children])
        curr = children[next_idx]
        result.append(curr)

    return result


# Sample an MDP
BRANCHING_FACTOR = 10
DEPTH = 5
REWARD_DIST = st.norm(loc=0, scale=2)

# The MDP has intermediate rewards
tree, root, best_reward = sample_mdp(
    DEPTH,
    BRANCHING_FACTOR,
    terminal_reward_dist=REWARD_DIST,
    intermediate_reward_dist=REWARD_DIST,
    seed=123,
)
print(f"Tree: {tree}; Best reward: {best_reward:.3f}")

# Run the search
res_path = greedy_search(tree, root)
res_reward = sum(node.reward for node in res_path)

print(f"[Greedy Search] Path length: {len(res_path)}, Reward: {res_reward:.3f}")
