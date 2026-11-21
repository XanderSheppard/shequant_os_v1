import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from sheyo_vit import VisionTransformer

# --- HYPERPARAMETERS ---
LR_ACTOR = 3e-4       
LR_CRITIC = 1e-3      
GAMMA = 0.99          
K_EPOCHS = 4          
EPS_CLIP = 0.2        
HIDDEN_SIZE = 256

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, device="cpu"):
        super(ActorCritic, self).__init__()
        self.device = device
        
        # 1. THE EYES (Vision Transformer)
        self.vit = VisionTransformer(img_size=64, in_channels=3).to(device)
        
        # 2. THE FUSION LAYER
        self.fusion = nn.Linear(64 + state_dim, HIDDEN_SIZE)
        
        # 3. ACTOR (Controls)
        self.actor_mean = nn.Linear(HIDDEN_SIZE, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim)) 
        
        # 4. CRITIC (Value)
        self.critic = nn.Linear(HIDDEN_SIZE, 1)
        
    def forward(self):
        raise NotImplementedError

    def act(self, img, imu):
        """Training Mode: Samples with noise"""
        with torch.no_grad():
            v_feat = self.vit(img)
            
        x = torch.cat([v_feat, imu], dim=1)
        x = F.relu(self.fusion(x))
        
        action_mean = torch.tanh(self.actor_mean(x))
        action_std = self.actor_logstd.exp()
        
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=1)
        
        state_val = self.critic(x)
        
        return action.detach(), action_logprob.detach(), state_val.detach()

    def get_deterministic_action(self, img, imu):
        """Flight Mode: Pure precision, no noise"""
        with torch.no_grad():
            v_feat = self.vit(img)
            
        x = torch.cat([v_feat, imu], dim=1)
        x = F.relu(self.fusion(x))
        
        # Direct output of the mean (Best Guess)
        action = torch.tanh(self.actor_mean(x))
        return action.detach()

    def evaluate(self, img, imu, action):
        v_feat = self.vit(img) 
        x = torch.cat([v_feat, imu], dim=1)
        x = F.relu(self.fusion(x))
        
        action_mean = torch.tanh(self.actor_mean(x))
        action_std = self.actor_logstd.exp()
        
        dist = Normal(action_mean, action_std)
        
        action_logprobs = dist.log_prob(action).sum(dim=1)
        dist_entropy = dist.entropy().sum(dim=1)
        state_values = self.critic(x)
        
        return action_logprobs, state_values, dist_entropy

class PPOAgent:
    def __init__(self, state_dim, action_dim, device="cpu"):
        self.device = device
        
        self.policy = ActorCritic(state_dim, action_dim, device).to(device)
        self.optimizer = optim.Adam([
            {'params': self.policy.vit.parameters(), 'lr': LR_ACTOR}, 
            {'params': self.policy.actor_mean.parameters(), 'lr': LR_ACTOR},
            {'params': self.policy.actor_logstd, 'lr': LR_ACTOR},
            {'params': self.policy.fusion.parameters(), 'lr': LR_ACTOR},
            {'params': self.policy.critic.parameters(), 'lr': LR_CRITIC}
        ])
        
        self.policy_old = ActorCritic(state_dim, action_dim, device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, img, imu, deterministic=False):
        img = torch.FloatTensor(img).unsqueeze(0).to(self.device)
        imu = torch.FloatTensor(imu).unsqueeze(0).to(self.device)
        
        if deterministic:
            # FLIGHT MODE
            action = self.policy_old.get_deterministic_action(img, imu)
            return action.cpu().numpy()[0], None, None
        else:
            # TRAINING MODE
            action, log_prob, state_val = self.policy_old.act(img, imu)
            return action.cpu().numpy()[0], log_prob, state_val

    def update(self, buffer):
        # 1. Monte Carlo Estimate
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(buffer.rewards), reversed(buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 2. Stack Tensors
        old_imgs = torch.stack(buffer.imgs).to(self.device).squeeze(1)
        old_imus = torch.stack(buffer.imus).to(self.device).squeeze(1)
        old_actions = torch.stack(buffer.actions).to(self.device).squeeze(1)
        
        if isinstance(buffer.logprobs[0], torch.Tensor):
             old_logprobs = torch.stack(buffer.logprobs).to(self.device).squeeze(1)
        else:
             old_logprobs = torch.tensor(buffer.logprobs, dtype=torch.float32).to(self.device)

        # 3. PPO Update
        for _ in range(K_EPOCHS):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_imgs, old_imus, old_actions)
            state_values = state_values.squeeze()
            
            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        buffer.clear()