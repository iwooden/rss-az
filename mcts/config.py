from dataclasses import dataclass


@dataclass
class MCTSConfig:
    """Hyperparameters for MCTS search."""

    num_simulations: int = 800
    c_puct: float = 2.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    num_players: int = 3
    action_dim: int = 246
