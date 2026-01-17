import numpy as np
import pickle
import os
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger("crisisconnect.rl_agent")

class RLAlertAgent:
    """
    Reinforcement Learning Agent using Q-Learning to optimize alert thresholds.
    
    State Space: Risk Score (0-100) discretized into 10 bins.
    Action Space: 0 (No Alert), 1 (Low Alert), 2 (High Alert).
    """
    
    def __init__(self, model_path: str = "data/rl_agent_qtable.pkl"):
        self.model_path = model_path
        self.actions = [0, 1, 2]  # 0: None, 1: Low, 2: High
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate
        
        # Initialize Q-table: 10 states (risk bins) x 3 actions
        # State 0: Risk 0-10, State 1: Risk 10-20, ..., State 9: Risk 90-100
        self.q_table = np.zeros((10, 3))
        
        self.load_model()

    def _get_state(self, risk_score: float) -> int:
        """Convert continuous risk score (0-100) to discrete state (0-9)."""
        # Clip to ensure 0-100 range
        score = max(0.0, min(100.0, risk_score))
        # Discretize: 99.9 -> 9, 100 -> 9
        state = min(int(score / 10), 9)
        return state

    def choose_action(self, risk_score: float, explore: bool = True) -> int:
        """
        Choose an action based on the risk score using Epsilon-Greedy strategy.
        
        Returns:
            int: 0 (No Alert), 1 (Low Alert), 2 (High Alert)
        """
        state = self._get_state(risk_score)
        
        if explore and np.random.uniform(0, 1) < self.epsilon:
            # Explore: Choose random action
            action = np.random.choice(self.actions)
        else:
            # Exploit: Choose best action from Q-table
            action = np.argmax(self.q_table[state, :])
            
        return int(action)

    def update(self, risk_score: float, action: int, reward: float):
        """
        Update Q-table based on feedback (Q-Learning algorithm).
        
        Args:
            risk_score: The risk score that led to the action.
            action: The action taken (0, 1, 2).
            reward: The reward received (+/-).
        """
        state = self._get_state(risk_score)
        
        # For this simple stateless problem (contextual bandit style), 
        # next state doesn't strictly depend on action, but we use standard Q-learning form.
        # We assume next state is effectively random or terminal, so max_future_q is 0 for single-step episodes.
        # However, if we view "next day" as next state, we could use it. 
        # For simplicity in this alert system, we treat each decision as an episode.
        
        current_q = self.q_table[state, action]
        
        # Q(s,a) = Q(s,a) + alpha * (R + gamma * max(Q(s',a')) - Q(s,a))
        # Since episodes are independent (one alert decision), we can simplify to:
        # Q(s,a) = Q(s,a) + alpha * (R - Q(s,a))
        
        new_q = current_q + self.learning_rate * (reward - current_q)
        self.q_table[state, action] = new_q
        
        self.save_model()

    def save_model(self):
        """Save Q-table to disk."""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.q_table, f)
        except Exception as e:
            logger.error(f"Failed to save RL model: {e}")

    def load_model(self):
        """Load Q-table from disk if exists."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.q_table = pickle.load(f)
                logger.info("Loaded existing RL agent model")
            except Exception as e:
                logger.error(f"Failed to load RL model: {e}")
                logger.info("Initialized new RL agent model")
        else:
            logger.info("Initialized new RL agent model")
            # Initialize with some common sense priors to avoid "tabula rasa" disasters
            # High risk (States 7,8,9) should prefer High Alert (Action 2)
            self.q_table[7:, 2] = 1.0 
            self.q_table[7:, 0] = -1.0
            # Low risk (States 0,1,2,3) should prefer No Alert (Action 0)
            self.q_table[:4, 0] = 1.0
            self.q_table[:4, 2] = -1.0

    def get_policy(self) -> Dict[str, str]:
        """Return the current learned policy in human-readable format."""
        policy = {}
        actions_map = {0: "NO_ALERT", 1: "LOW_ALERT", 2: "HIGH_ALERT"}
        for i in range(10):
            best_action = np.argmax(self.q_table[i, :])
            range_str = f"{i*10}-{i*10+10}"
            policy[range_str] = actions_map[best_action]
        return policy
