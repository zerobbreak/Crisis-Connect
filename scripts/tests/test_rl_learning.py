import sys
import os
import numpy as np
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.rl_agent import RLAlertAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("RL_Simulation")

def simulate_environment(days=1000):
    """
    Simulate a weather environment to train the RL agent.
    """
    agent = RLAlertAgent()
    logger.info("🤖 Starting RL Agent Simulation...")
    
    total_reward = 0
    rewards_history = []
    
    for day in range(days):
        # 1. Generate random risk score (0-100)
        # Bias towards lower risks (common) but occasional high risks (disasters)
        if np.random.random() < 0.8:
            risk_score = np.random.normal(20, 10) # Normal days
        else:
            risk_score = np.random.normal(80, 10) # Stormy days
        risk_score = max(0, min(100, risk_score))
        
        # 2. Agent chooses action
        action = agent.choose_action(risk_score)
        
        # 3. Determine Ground Truth (Did flood happen?)
        # Probability of flood increases with risk score
        flood_prob = 1 / (1 + np.exp(-(risk_score - 60) / 10)) # Sigmoid centered at 60
        flood_happened = np.random.random() < flood_prob
        
        # 4. Calculate Reward
        reward = 0
        if action == 2: # High Alert
            if flood_happened:
                reward = 10 # Saved lives!
            else:
                reward = -5 # Crying wolf
        elif action == 1: # Low Alert
            if flood_happened:
                reward = 5 # Better than nothing
            else:
                reward = -1 # Minor annoyance
        elif action == 0: # No Alert
            if flood_happened:
                reward = -100 # Disaster! Missed flood
            else:
                reward = 1 # Good silence
                
        # 5. Update Agent
        agent.update(risk_score, action, reward)
        
        total_reward += reward
        rewards_history.append(total_reward)
        
        if day % 100 == 0:
            logger.info(f"Day {day}: Risk={risk_score:.1f}, Action={action}, Flood={flood_happened}, Reward={reward}")

    logger.info("✅ Simulation Complete!")
    logger.info(f"Total Reward: {total_reward}")
    
    # Print Learned Policy
    logger.info("\n🧠 Learned Policy:")
    policy = agent.get_policy()
    for state, action in policy.items():
        logger.info(f"Risk {state}: {action}")

if __name__ == "__main__":
    simulate_environment()
