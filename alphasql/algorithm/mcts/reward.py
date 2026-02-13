from alphasql.algorithm.mcts.mcts_node import MCTSNode, MCTSNodeType
from alphasql.algorithm.mcts.mcts_action import SQLGenerationAction, SQLRevisionAction
from alphasql.database.sql_execution import cached_execute_sql_with_timeout, is_valid_execution_result
from typing import Dict, Any, List, Optional
from pathlib import Path
import math
import hashlib


class RewardModel:
    def __init__(self, **kwargs):
        pass

    def get_reward(self, end_node: MCTSNode) -> float:
        raise NotImplementedError()


class MajorityVoteRewardModel(RewardModel):
    def __init__(self, llm_kwargs: Optional[Dict[str, Any]] = None):
        self.llm_kwargs = llm_kwargs or {}

    def get_reward(self, end_node: MCTSNode) -> float:
        assert end_node.node_type == MCTSNodeType.END
        parent_node = end_node.parent_node
        assert parent_node.node_type == MCTSNodeType.SQL_REVISION or parent_node.node_type == MCTSNodeType.SQL_GENERATION
        action = parent_node.parent_action
        assert isinstance(action, SQLGenerationAction) or isinstance(action, SQLRevisionAction)
        assert parent_node.parent_node is not None
        return end_node.consistency_score if end_node.consistency_score is not None else 0.0


class ExecutionTrajectoryRewardModel(RewardModel):
    """
    ETER-style reward: embeds SQL execution traces into a deterministic vector and
    uses cosine similarity against a reference trajectory if available.
    """

    def __init__(self, reference_sql_map: Optional[Dict[str, str]] = None, embedding_dim: int = 128, mix_consistency_alpha: float = 0.3):
        self.reference_sql_map = reference_sql_map or {}
        self.embedding_dim = embedding_dim
        self.mix_consistency_alpha = mix_consistency_alpha

    def _result_to_tokens(self, execution_result) -> List[str]:
        if not is_valid_execution_result(execution_result):
            return ["<invalid>"]
        tokens: List[str] = []
        if execution_result.result_cols:
            tokens.extend([f"col:{col}" for col in execution_result.result_cols])
        for row in execution_result.result[:50]:
            for value in row:
                tokens.append(f"val:{str(value)}")
        return tokens or ["<empty>"]

    def _embed_tokens(self, tokens: List[str]) -> List[float]:
        vec = [0.0] * self.embedding_dim
        for token in tokens:
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.embedding_dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]

    def _cosine(self, x: List[float], y: List[float]) -> float:
        return sum(a * b for a, b in zip(x, y))

    def _get_reference_embedding(self, end_node: MCTSNode) -> Optional[List[float]]:
        question_key = str(end_node.path_nodes[0].original_question)
        reference_sql = self.reference_sql_map.get(question_key)
        if not reference_sql:
            return None
        db_path = Path(end_node.db_root_dir) / end_node.db_id / f"{end_node.db_id}.sqlite"
        ref_result = cached_execute_sql_with_timeout(str(db_path), reference_sql)
        return self._embed_tokens(self._result_to_tokens(ref_result))

    def get_reward(self, end_node: MCTSNode) -> float:
        assert end_node.node_type == MCTSNodeType.END
        sql_query = end_node.final_sql_query
        if not sql_query:
            end_node.eter_reward = 0.0
            return 0.0
        db_path = Path(end_node.db_root_dir) / end_node.db_id / f"{end_node.db_id}.sqlite"
        execution_result = cached_execute_sql_with_timeout(str(db_path), sql_query)
        if not is_valid_execution_result(execution_result):
            end_node.eter_reward = 0.0
            return 0.0

        candidate_embedding = self._embed_tokens(self._result_to_tokens(execution_result))
        reference_embedding = self._get_reference_embedding(end_node)
        trajectory_reward = self._cosine(candidate_embedding, reference_embedding) if reference_embedding is not None else 0.5
        consistency_reward = end_node.consistency_score if end_node.consistency_score is not None else 0.0
        reward = (1 - self.mix_consistency_alpha) * trajectory_reward + self.mix_consistency_alpha * consistency_reward
        end_node.eter_reward = trajectory_reward
        return reward
