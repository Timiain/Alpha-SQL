from alphasql.algorithm.mcts.mcts_node import *
from alphasql.algorithm.mcts.mcts_action import *
from alphasql.algorithm.mcts.reward import *
from alphasql.runner.task import Task
import math
import random
from pathlib import Path
from typing import Dict, Any, List
import pickle


class MCTSSolver:
    def __init__(self,
                 db_root_dir: str,
                 task: Task,
                 max_rollout_steps: int,
                 max_depth: int,
                 exploration_constant: float,
                 save_root_dir: str,
                 llm_kwargs: Dict[str, Any],
                 reward_model: RewardModel,
                 meta_action_prior: Dict[str, float] = None,
                 multi_objective_weights: Dict[str, float] = None,
                 adaptive_mcts_kwargs: Dict[str, float] = None):
        self.llm_kwargs = llm_kwargs
        self.reward_model = reward_model
        self.task = task
        self.db_root_dir = db_root_dir
        self.max_rollout_steps = max_rollout_steps
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.save_root_dir = save_root_dir
        self.meta_action_prior = meta_action_prior or {}

        self.multi_objective_weights = {"lambda": 0.5, "mu": 0.3, "prior": 0.2}
        if multi_objective_weights:
            self.multi_objective_weights.update(multi_objective_weights)

        self.adaptive_mcts_kwargs = {
            "tau_low": 0.05,
            "tau_high": 0.35,
            "adaptive_alpha": 2.0,
            "max_adaptive_iters": 4,
            "max_children_per_expansion": 12,
        }
        if adaptive_mcts_kwargs:
            self.adaptive_mcts_kwargs.update(adaptive_mcts_kwargs)

    def _ucb_score(self, child: MCTSNode, parent_visits: int) -> float:
        if child.N == 0:
            return float("inf")
        exploitation = child.Q / child.N
        exploration = self.exploration_constant * math.sqrt(math.log(max(parent_visits, 1)) / child.N)
        info_gain_term = self.multi_objective_weights["lambda"] * child.info_gain
        eter_term = self.multi_objective_weights["mu"] * (child.eter_reward if child.eter_reward is not None else 0.0)
        prior_term = self.multi_objective_weights["prior"] * child.action_prior
        return exploitation + exploration + info_gain_term + eter_term + prior_term

    def select(self, node: MCTSNode) -> MCTSNode:
        current = node
        while current.children and not current.is_terminal():
            eligible_children = [child for child in current.children if child.info_gain >= self.adaptive_mcts_kwargs["tau_low"] or child.N == 0]
            if not eligible_children:
                return current
            if not all(child.N > 0 for child in eligible_children):
                return next(child for child in eligible_children if child.N == 0)
            current = max(eligible_children, key=lambda child: self._ucb_score(child, current.N))
        return current

    def expand(self, node: MCTSNode) -> List[MCTSNode]:
        assert node.children == [], f"Children nodes of node {node.node_type} before expansion is not empty"
        valid_action_space = get_valid_action_space_for_node(node)
        for action in valid_action_space:
            action_nodes = action.create_children_nodes(node, self.llm_kwargs)
            prior = self.meta_action_prior.get(action.__class__.__name__, 0.0)
            for action_node in action_nodes:
                action_node.action_prior = prior
            node.children.extend(action_nodes)

        node.children.sort(key=lambda child: child.action_prior, reverse=True)
        max_children = int(self.adaptive_mcts_kwargs["max_children_per_expansion"])
        if max_children > 0:
            node.children = node.children[:max_children]

        random.shuffle(node.children)
        return node.children

    def simulate(self, node: MCTSNode) -> MCTSNode:
        assert node.children == [], "Node before simulation have non-empty children"
        current = node
        while not current.is_terminal() and current.depth < self.max_depth:
            self.expand(current)
            if not current.children:
                break
            current = random.choice(current.children)
        return current

    def _adaptive_iterations_for_node(self, node: MCTSNode) -> int:
        base = 1 + int(self.adaptive_mcts_kwargs["adaptive_alpha"] * max(node.info_gain, 0.0))
        capped = min(base, int(self.adaptive_mcts_kwargs["max_adaptive_iters"]))
        return max(1, capped)

    def backpropagate(self, node: MCTSNode):
        print("Backpropagate, Final SQL Query: ", node.final_sql_query)
        current = node
        if current.N == 0:
            reward = self.reward_model.get_reward(current)
        else:
            reward = current.Q / current.N
        while current is not None:
            prev_mean = (current.Q / current.N) if current.N > 0 else 0.0
            current.N += 1
            current.Q += reward
            new_mean = current.Q / current.N
            current.info_gain = abs(new_mean - prev_mean)
            current = current.parent_node

    def find_all_end_nodes(self, node: MCTSNode) -> List[MCTSNode]:
        if node.node_type == MCTSNodeType.END:
            return [node]
        else:
            end_nodes = []
            for child in node.children:
                end_nodes.extend(self.find_all_end_nodes(child))
            return end_nodes

    def find_all_valid_reasoning_paths(self, node: MCTSNode) -> List[List[MCTSNode]]:
        end_nodes = self.find_all_end_nodes(node)
        reasoning_paths = []
        for end_node in end_nodes:
            reasoning_paths.append(end_node.path_nodes)
        return reasoning_paths

    def solve(self):
        schema_context = "\n".join([build_table_ddl_statement(
            self.task.table_schema_dict[table_name].to_dict(),
            add_value_description=True,
            add_column_description=True,
            add_value_examples=True,
            add_expanded_column_name=True
        ) for table_name in self.task.table_schema_dict])
        root_node = MCTSNode(MCTSNodeType.ROOT,
                             parent_node=None,
                             parent_action=None,
                             depth=0,
                             db_id=self.task.db_id,
                             db_root_dir=self.db_root_dir,
                             original_question=self.task.question,
                             hint=self.task.evidence,
                             schema_context=schema_context,
                             table_schema_dict=self.task.table_schema_dict)
        root_node.path_nodes = [root_node]

        rollout_step = 0
        while rollout_step < self.max_rollout_steps:
            print(f"Question ID: {self.task.question_id}, Rollout step {rollout_step + 1} / {self.max_rollout_steps}")
            leaf_node = self.select(root_node)
            adaptive_iters = self._adaptive_iterations_for_node(leaf_node)

            for _ in range(adaptive_iters):
                if rollout_step >= self.max_rollout_steps:
                    break
                if leaf_node.is_terminal():
                    self.backpropagate(leaf_node)
                    rollout_step += 1
                    continue

                if not leaf_node.children:
                    self.expand(leaf_node)
                if not leaf_node.children:
                    rollout_step += 1
                    continue

                simulate_from = random.choice(leaf_node.children)
                end_node = self.simulate(simulate_from)
                self.backpropagate(end_node)
                rollout_step += 1

        all_valid_reasoning_paths = self.find_all_valid_reasoning_paths(root_node)
        save_path = Path(self.save_root_dir) / f"{self.task.question_id}.pkl"
        print(f"Question ID: {self.task.question_id} done, Number of valid reasoning paths: {len(all_valid_reasoning_paths)}")
        with open(save_path, "wb") as f:
            pickle.dump(all_valid_reasoning_paths, f)
