import torch
import random
import ctypes
import numpy as np
import time
import os
import sys
import importlib.util

# 1. ABSOLUTE SEED LOCKING
def set_master_seed(seed=42):
    """Enforce mathematical reproducibility across the entire architecture."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_master_seed(42)

# 2. DYNAMIC PATH RESOLUTION
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")
LOG_DIR = os.path.join(REPO_ROOT, "logs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 3. TOP-LEVEL CONFIGURATION
CONFIG = {
    "total_timesteps": 500_000,
    "batch_size": 128,
    "learning_rate_vit": 1e-4,   
    "learning_rate_ppo": 3e-4,   
    "gamma": 0.99,               
    "ppo_clip": 0.2,             
    "target_altitude": 50.0      
}

from sheyo_gym_atmosphere import SheyoSurfaceEnv
from sheyo_ppo import PPOAgent
try:
    from sheyo_gym_atmosphere import SheyoSurfaceEnv
except ImportError:
    from sheyo_gym_surface import SheyoSurfaceEnv

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.imus = []
        self.imgs = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.imus[:]
        del self.imgs[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

MAX_EPISODES = 5000
MAX_STEPS = 400       
UPDATE_TIMESTEP = 2000 
PRINT_INTERVAL = 10
SAVE_INTERVAL = 50

# Remove the old MAX_EPISODES, MAX_STEPS, etc. from the middle of the file.
# Move them into your CONFIG dictionary at the top instead!

def train():
    print("[SHEYO] CURRICULUM TRAINING INITIATED")
    print("-------------------------------------")
    print(" PHASE 1: Earth Gravity (-9.81)")
    print(" PHASE 2: Hyper Gravity (-20.00) @ Ep 100")
    print(" TARGET:  50.0 Meters")
    print("-------------------------------------")
    
    # Start easy
    env = SheyoSurfaceEnv(gravity=-9.81)
    
    agent = PPOAgent(state_dim=4, action_dim=3, device="cuda" if torch.cuda.is_available() else "cpu")
    buffer = RolloutBuffer()
    
    time_step = 0
    running_reward = 0
    hyper_gravity_active = False

    # Using the CONFIG dictionary from the top of the file
    max_episodes = 5000
    max_steps = 400       
    update_timestep = 2000 
    print_interval = 10
    save_interval = 50

    for episode in range(1, max_episodes + 1):
        
        # --- CURRICULUM SWITCH ---
        if episode >= 100 and not hyper_gravity_active:
            print("\n!!! WARNING: ENGAGING HYPER-GRAVITY (-20.0 m/s) !!!\n")
            env.set_difficulty(-20.0)
            hyper_gravity_active = True

        state, _ = env.reset()
        current_ep_reward = 0
        
        for t in range(max_steps):
            time_step += 1
            
            dummy_img = torch.zeros((3, 64, 64)) 
            action, log_prob, _ = agent.select_action(dummy_img, state)
            
            next_state, reward, done, truncated, _ = env.step(action)
            
            buffer.imus.append(torch.FloatTensor(state))
            buffer.imgs.append(dummy_img)
            buffer.actions.append(torch.FloatTensor(action))
            buffer.logprobs.append(torch.FloatTensor([log_prob]))
            buffer.rewards.append(reward)
            buffer.is_terminals.append(done)
            
            if time_step % update_timestep == 0:
                print(f"   >>> UPDATING POLICY at Step {time_step}...")
                agent.update(buffer)
                buffer.clear()

            current_ep_reward += reward
            state = next_state
            
            if done or truncated:
                break

        running_reward += current_ep_reward
        
        if episode % print_interval == 0:
            avg_reward = running_reward / print_interval
            print(f"Ep {episode} | Avg Reward: {avg_reward:.2f} | Steps: {t} | Gravity: {env.gravity_val}")
            running_reward = 0
            
        if episode % save_interval == 0:
            # DYNAMIC PATH: Saving to the CHECKPOINT_DIR defined at the top
            path = os.path.join(CHECKPOINT_DIR, f"sheyo_ppo_ep{episode}.pth")
            
            # ACADEMIC CHECKPOINT: Saving the policy AND the step/episode
            torch.save({
                'episode': episode,
                'policy_state_dict': agent.policy.state_dict(),
                # Add optimizer state here if agent exposes it: 'optimizer': agent.optimizer.state_dict()
            }, path)
            print(f"   [SAVED] Mathematical Checkpoint locked: {path}")

if __name__ == '__main__':
    train()