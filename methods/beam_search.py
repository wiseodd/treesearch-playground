from typing import List, Tuple
import networkx as nx
import scipy.stats as st
import numpy as np
import torch

from environments.mdp import sample_mdp, Node


def beam_search(
    tree: nx.DiGraph, root: Node, k: int
) -> Tuple[List[List[Node]], List[float]]:
    currs: List[List[Node]] = [[root]]
    total_rewards: List[float] = [0 for _ in range(k)]
    stop: bool = False

    while True:
        children = [list(tree.successors(curr[-1])) for curr in currs]
        children_flat = []
        for c in children:
            children_flat += c

        # All leaves --- done (assuming leaves on the same depth)
        if len(children_flat) == 0:
            break

        # Pick top-k children
        rewards, flat_idxs = torch.topk(
            torch.tensor([child.reward for child in children_flat]), k=k
        )

        # Construct top-k paths and their rewards
        new_currs, new_total_rewards = [], []
        for r, i in zip(rewards, flat_idxs):
            # Assuming the same branching factors
            parent_idx = i // len(children[0])
            new_currs.append(currs[parent_idx] + [children_flat[i]])
            new_total_rewards.append(total_rewards[parent_idx] + r)

        currs = new_currs
        total_rewards = new_total_rewards

    return currs, total_rewards


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
for k in [1, 2, 3]:
    res_paths, res_rewards = beam_search(tree, root, k=k)

    best_idx = np.argmax(res_rewards)
    res_path = res_paths[best_idx]
    res_reward = res_rewards[best_idx]

    print(f"[Beam Search k={k}] Path length: {len(res_path)}, Reward: {res_reward:.3f}")
