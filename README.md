# Treesearch Examples

Tree-search algorithms examples on toy Markov Decision Process (MDP) environments.

## Setup

- `uv sync`
- `uv run python methods/{ALG_NAME}.py`

## List of algorithms

- [Best-of-N Random Search](https://github.com/wiseodd/treesearch-examples/blob/main/methods/random_search.py)
- [Greedy Search](https://github.com/wiseodd/treesearch-examples/blob/main/methods/greedy_search.py)
- [Beam Search](https://github.com/wiseodd/treesearch-examples/blob/main/methods/beam_search.py)

## Search spaces

All algorithms run on toy tree search spaces, which can be controlled in terms of their sizes
(depth, branching factor) and in terms of their reward distributions.

```python
from environments.mdp import sample_mdp, Node

tree, root, best_reward = sample_mdp(
    depth=5,
    branching_factor=10,
    terminal_reward_dist=scipy.stats.norm(0, 2),
    intermediate_reward_dist=scipy.stats.norm(0, 0.5),
)

# Print "Tree: DiGraph with 111111 nodes and 111110 edges; Best reward: 20.952"
print(f"Tree: {tree}; Best reward: {best_reward:.3f}")
```

The tree is a `networkx` directed graph, where each node is specified via the following dataclass:

```python
@dataclass
class Node:
    name: str
    depth: int
    is_leaf: bool
    reward: float

    def __hash__(self) -> int:
        return hash(self.name)
```
