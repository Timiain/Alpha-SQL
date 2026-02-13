from pydantic import BaseModel
from typing import Dict, Any, Optional

class MCTSConfig(BaseModel):
    """
    Configuration for the MCTS Runner.
    """
    tasks_file_path: str
    subset_file_path: Optional[str]
    db_root_dir: str
    n_processes: int
    max_rollout_steps: int
    max_depth: int
    exploration_constant: float
    save_root_dir: str
    mcts_model_kwargs: Dict[str, Any]
    reward_model_kwargs: Optional[Dict[str, Any]] = None
    reward_model_type: Optional[str] = "majority_vote"
    meta_action_prior: Optional[Dict[str, float]] = None
    multi_objective_weights: Optional[Dict[str, float]] = None
    adaptive_mcts_kwargs: Optional[Dict[str, float]] = None
    random_seed: Optional[int] = 42
