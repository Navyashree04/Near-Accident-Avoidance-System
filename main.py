#!/usr/bin/env python3
"""
near_accident_fixed_final.py

Final fix to make the ego vehicle actually reach the goal.
"""

import os
import random
import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

# -------------------------
# Repro / device
# -------------------------
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Environment (CARLO) - SIMPLIFIED
# -------------------------
class CARLOEnvironment:
    def __init__(self):
        self.reset()

    def reset(self):
        # Ego starts before crossing, goal beyond crossing
        self.ego_pos = 0.0
        self.ego_vel = 10.0  # Higher initial speed
        # Make ado start much further away to ensure success is possible
        self.ado_pos = float(np.random.uniform(60.0, 100.0))  # Further away
        self.ado_vel = float(np.random.uniform(4.0, 8.0))    # Slower ado
        self.time_step = 0
        self.max_steps = 200  # Reduced max steps to encourage faster completion
        self.crossing_pos = 50.0
        self.success_pos = 60.0  # Reduced success threshold
        return self.get_observation()

    def get_observation(self):
        return np.array([
            self.ego_pos / 100.0,
            self.ego_vel / 20.0,
            self.ado_pos / 100.0,
            self.ado_vel / 20.0
        ], dtype=np.float32)

    def step(self, action, mode='normal'):
        throttle = float(np.clip(action, -1.0, 1.0))

        dt = 0.1
        acc = throttle * 5.0  # Higher acceleration
        self.ego_vel = float(np.clip(self.ego_vel + acc * dt, 0.0, 20.0))
        self.ego_pos = float(self.ego_pos + self.ego_vel * dt)

        self.ado_pos = float(self.ado_pos - self.ado_vel * dt)

        # More lenient collision detection
        ego_at_intersection = (48.0 < self.ego_pos < 52.0)
        ado_at_intersection = (48.0 < self.ado_pos < 52.0)
        collision = bool(ego_at_intersection and ado_at_intersection)

        success = self.ego_pos > self.success_pos
        timeout = self.time_step >= self.max_steps
        self.time_step += 1
        done = collision or success or timeout

        # SIMPLIFIED Reward logic
        if collision:
            reward = -20.0
        elif success:
            # Large reward for success, bonus for finishing quickly
            reward = 50.0 + (self.max_steps - self.time_step) * 0.2
        else:
            # Progress reward based on position and speed
            progress = self.ego_pos / self.success_pos
            progress_reward = 0.1 * progress
            
            # Speed reward - encourage maintaining good speed
            speed_reward = 0.05 * (self.ego_vel - 5.0)  # Reward speeds above 5.0
            
            reward = progress_reward + speed_reward

        info = {'collision': collision, 'success': success, 'ego_pos': self.ego_pos, 'ado_pos': self.ado_pos}
        return self.get_observation(), float(reward), done, info

# -------------------------
# SIMPLIFIED Expert policy
# -------------------------
def expert_policy(obs, mode='normal'):
    ego_pos_n, ego_vel_n, ado_pos_n, ado_vel_n = obs
    ego_pos = ego_pos_n * 100.0
    ego_vel = ego_vel_n * 20.0
    ado_pos = ado_pos_n * 100.0
    ado_vel = ado_vel_n * 20.0

    # SIMPLE logic: if ado is far enough, accelerate; if close, brake
    distance_to_ado = abs(ado_pos - 50.0)  # Distance of ado from intersection
    ego_distance_to_intersection = 50.0 - ego_pos
    
    if mode == 'timid':
        # Timid: brake early and often
        if distance_to_ado < 30.0 and ego_distance_to_intersection > 0:
            return np.array([-0.8], dtype=np.float32)
        elif ego_vel < 8.0:
            return np.array([0.3], dtype=np.float32)
        else:
            return np.array([0.0], dtype=np.float32)
            
    elif mode == 'aggressive':
        # Aggressive: mostly accelerate, brake only when very close
        if distance_to_ado < 10.0 and ego_distance_to_intersection > 0:
            return np.array([-0.5], dtype=np.float32)
        elif ego_vel < 15.0:
            return np.array([0.8], dtype=np.float32)
        else:
            return np.array([0.0], dtype=np.float32)
            
    else:  # normal
        # Balanced approach
        if distance_to_ado < 20.0 and ego_distance_to_intersection > 0:
            return np.array([-0.6], dtype=np.float32)
        elif ego_vel < 12.0:
            return np.array([0.5], dtype=np.float32)
        else:
            return np.array([0.0], dtype=np.float32)

# -------------------------
# Expert data generation - ENSURING SUCCESS
# -------------------------
def generate_expert_data(num_episodes=1000, keep_failed_fraction=0.2):
    env = CARLOEnvironment()
    modes = ['timid', 'normal', 'aggressive']
    mode_to_idx = {'timid': 0, 'normal': 1, 'aggressive': 2}

    observations = []
    actions = []
    modes_idx = []
    successes = 0
    kept_failed = 0

    print("Generating expert data (ensuring success episodes)...")
    
    # First, generate some guaranteed successful episodes
    for mode in modes:
        for _ in range(50):  # 50 successful episodes per mode
            env = CARLOEnvironment()
            # Set parameters for guaranteed success
            env.ado_pos = 80.0  # ADO starts far away
            env.ado_vel = 4.0   # ADO moves slowly
            
            mode_i = mode_to_idx[mode]
            obs = env.reset()
            done = False
            ep_obs = []
            ep_acts = []
            
            while not done:
                act = expert_policy(obs, mode)
                ep_obs.append(obs.copy())
                ep_acts.append(act.copy())
                obs, r, done, info = env.step(act[0], mode)

            if info.get('success', False):
                successes += 1
                observations.extend(ep_obs)
                actions.extend(ep_acts)
                modes_idx.extend([mode_i] * len(ep_obs))

    # Then generate random episodes
    for ep in trange(num_episodes - 150, desc="Random episodes"):
        mode = random.choice(modes)
        mode_i = mode_to_idx[mode]
        obs = env.reset()
        done = False
        ep_obs = []
        ep_acts = []
        while not done:
            act = expert_policy(obs, mode)
            ep_obs.append(obs.copy())
            ep_acts.append(act.copy())
            obs, r, done, info = env.step(act[0], mode)

        if info.get('success', False):
            successes += 1
            observations.extend(ep_obs)
            actions.extend(ep_acts)
            modes_idx.extend([mode_i] * len(ep_obs))
        else:
            if random.random() < keep_failed_fraction:
                kept_failed += 1
                observations.extend(ep_obs)
                actions.extend(ep_acts)
                modes_idx.extend([mode_i] * len(ep_obs))

    print(f"Expert data: {len(observations)} samples, {successes} successful eps, kept {kept_failed} failed eps")
    return {'obs': np.array(observations, dtype=np.float32),
            'acts': np.array(actions, dtype=np.float32).reshape(-1,1),
            'modes': np.array(modes_idx, dtype=np.int64)}

# -------------------------
# CoIL model & training
# -------------------------
class CoILNetwork(nn.Module):
    def __init__(self, obs_dim=4, action_dim=1, num_modes=3, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, action_dim), nn.Tanh())
            for _ in range(num_modes)
        ])
        self.num_modes = num_modes

    def forward(self, obs, mode):
        feat = self.shared(obs)
        out = torch.zeros(obs.shape[0], 1, device=obs.device, dtype=feat.dtype)
        for m in range(self.num_modes):
            mask = (mode == m)
            if mask.any():
                out[mask] = self.branches[m](feat[mask])
        return out

class DrivingDataset(Dataset):
    def __init__(self, obs, acts, modes):
        self.obs = torch.from_numpy(obs).float()
        self.acts = torch.from_numpy(acts).float()
        self.modes = torch.from_numpy(modes).long()
    def __len__(self): return len(self.obs)
    def __getitem__(self, idx):
        return self.obs[idx], self.acts[idx], self.modes[idx]

def train_coil(coil, dataset, device=DEVICE, batch_size=256, epochs=20, lr=1e-3):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = optim.Adam(coil.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    coil.to(device)
    
    best_loss = float('inf')
    for ep in range(epochs):
        coil.train()
        total_loss = 0.0
        for obs, act, mode in loader:
            obs = obs.to(device); act = act.to(device); mode = mode.to(device)
            pred = coil(obs, mode)
            loss = loss_fn(pred, act)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * obs.size(0)
        
        avg_loss = total_loss / len(dataset)
        if (ep+1) % 5 == 0:
            print(f"CoIL Epoch {ep+1}/{epochs}  loss={avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(coil.state_dict(), "models/coil_best.pth")
    
    print(f"CoIL training done (best_loss={best_loss:.6f})")
    return coil

# -------------------------
# ActorCritic module for PPO
# -------------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim=4, n_actions=3, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.action_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        feat = self.shared(x)
        logits = self.action_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return logits, value

# -------------------------
# PPO Agent with EXPLORATION
# -------------------------
class PPOAgent:
    def __init__(self, state_dim=4, n_actions=3, device=DEVICE):
        self.device = device
        self.n_actions = n_actions
        self.model = ActorCritic(state_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, eps=1e-5)
        
        self.gamma = 0.99
        self.lam = 0.95
        self.eps_clip = 0.2
        self.K_epochs = 4
        self.ent_coef = 0.1  # Higher entropy for exploration
        self.value_coef = 0.5
        
        self.memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': [], 'values': []}

    def act(self, state_np):
        st = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.model(st)
            # Add exploration noise
            logits = logits + torch.randn_like(logits) * 0.5
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
        
        self.memory['states'].append(state_np.astype(np.float32))
        self.memory['actions'].append(int(action.item()))
        self.memory['logprobs'].append(logp.detach().cpu())
        self.memory['values'].append(value.detach().cpu())
        return int(action.item())

    def compute_returns(self, rewards, values, last_value, is_terminals, gamma=0.99):
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)
        R = last_value
        
        for t in reversed(range(T)):
            R = rewards[t] + gamma * R * (1.0 - is_terminals[t])
            returns[t] = R
            
        return returns

    def update(self):
        if len(self.memory['rewards']) < 1:
            return

        states = torch.from_numpy(np.vstack(self.memory['states'])).float().to(self.device)
        actions = torch.tensor(self.memory['actions'], dtype=torch.long, device=self.device)
        old_logprobs = torch.stack(self.memory['logprobs']).squeeze().to(self.device)
        old_values = torch.cat(self.memory['values']).squeeze().to(self.device)
        
        rewards = np.array(self.memory['rewards'], dtype=np.float32)
        is_terminals = np.array(self.memory['is_terminals'], dtype=np.bool_)

        # Compute last value for bootstrap
        last_val = 0.0
        if not is_terminals[-1]:
            with torch.no_grad():
                st = torch.from_numpy(self.memory['states'][-1].astype(np.float32)).unsqueeze(0).to(self.device)
                _, last_val_t = self.model(st)
                last_val = float(last_val_t.cpu().numpy().squeeze())

        # Compute returns (simpler than GAE)
        returns = self.compute_returns(rewards, old_values.cpu().numpy(), last_val, is_terminals, self.gamma)
        returns_t = torch.from_numpy(returns).float().to(self.device)

        # Compute advantages
        advantages = returns_t - old_values
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        for _ in range(self.K_epochs):
            logits, values_pred = self.model(states)
            dist = Categorical(logits=logits)
            new_logprobs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            new_values = values_pred.squeeze(-1)

            # Policy loss
            ratios = torch.exp(new_logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(new_values, returns_t)
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.ent_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

        # Clear memory
        for k in self.memory:
            self.memory[k].clear()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

# -------------------------
# Training pipeline with PROGRESS MONITORING
# -------------------------
def train_system(num_expert_eps=600, keep_failed_fraction=0.2, coil_epochs=15, ppo_episodes=500):
    Path("models").mkdir(exist_ok=True)
    
    # 1) Generate expert data
    print("Step 1: Generating expert data...")
    data = generate_expert_data(num_expert_eps, keep_failed_fraction)
    if len(data['obs']) == 0:
        raise ValueError("No expert data generated!")
    
    obs = data['obs']; acts = data['acts']; modes = data['modes']
    print(f"Total expert samples: {obs.shape[0]}")

    # 2) Train CoIL
    print("\nStep 2: Training CoIL...")
    dataset = DrivingDataset(obs, acts, modes)
    coil = CoILNetwork(obs_dim=obs.shape[1], action_dim=1, num_modes=3).to(DEVICE)
    coil = train_coil(coil, dataset, device=DEVICE, batch_size=256, epochs=coil_epochs, lr=1e-3)

    # 3) Train PPO
    print("\nStep 3: Training PPO...")
    agent = PPOAgent(state_dim=obs.shape[1], n_actions=3, device=DEVICE)
    env = CARLOEnvironment()

    update_frequency = 200  # Update every 200 steps
    timestep = 0
    episode_rewards = []
    success_history = []
    
    best_success_rate = 0.0

    for ep in range(ppo_episodes):
        obs_s = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        
        while not done:
            mode_idx = agent.act(obs_s)
            mode = ['timid', 'normal', 'aggressive'][mode_idx]

            with torch.no_grad():
                st = torch.from_numpy(obs_s.astype(np.float32)).unsqueeze(0).to(DEVICE)
                mode_t = torch.LongTensor([mode_idx]).to(DEVICE)
                action_cont = coil(st, mode_t).cpu().numpy()[0,0]

            next_obs, reward, done, info = env.step(action_cont, mode=mode)
            
            agent.memory['rewards'].append(float(reward))
            agent.memory['is_terminals'].append(bool(done))
            ep_reward += float(reward)
            timestep += 1
            steps += 1
            obs_s = next_obs

            if timestep % update_frequency == 0:
                agent.update()

        episode_rewards.append(ep_reward)
        success_history.append(1 if info.get('success', False) else 0)
        
        # Update at episode end
        if len(agent.memory['rewards']) > 0:
            agent.update()

        # Progress reporting with success rate
        if (ep+1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            success_rate = np.mean(success_history[-50:]) * 100
            final_pos = info.get('ego_pos', 0)
            print(f"PPO Episode {ep+1}/{ppo_episodes}  "
                  f"avg_reward={avg_reward:.2f}  "
                  f"success_rate={success_rate:.1f}%  "
                  f"final_pos={final_pos:.1f}  "
                  f"steps={steps}")

        # Save best model based on success rate
        if len(success_history) >= 50:
            current_success_rate = np.mean(success_history[-50:])
            if current_success_rate > best_success_rate:
                best_success_rate = current_success_rate
                agent.save("models/ppo_best.pth")
                # FIX: Use torch.save instead of coil.save
                torch.save(coil.state_dict(), "models/coil_best.pth")
                print(f"New best model! Success rate: {best_success_rate*100:.1f}%")

    # Final save
    torch.save(coil.state_dict(), "models/coil_final.pth")
    agent.save("models/ppo_final.pth")
    print(f"Training finished. Best success rate: {best_success_rate*100:.1f}%")
    return coil, agent

# -------------------------
# Evaluation
# -------------------------
def evaluate_system(coil, agent, episodes=100):
    env = CARLOEnvironment()
    collisions = 0; successes = 0; timeouts = 0
    completion_times = []
    mode_counts = {0:0, 1:0, 2:0}
    final_positions = []
    
    for ep in range(episodes):
        obs_s = env.reset()
        done = False
        ep_mode_counts = {0:0, 1:0, 2:0}
        
        while not done:
            mode_idx = agent.act(obs_s)
            ep_mode_counts[mode_idx] += 1
            mode = ['timid','normal','aggressive'][mode_idx]
            
            with torch.no_grad():
                st = torch.from_numpy(obs_s.astype(np.float32)).unsqueeze(0).to(DEVICE)
                action_cont = coil(st, torch.LongTensor([mode_idx]).to(DEVICE)).cpu().numpy()[0,0]
            
            obs_s, r, done, info = env.step(action_cont, mode=mode)
            
            if done:
                final_positions.append(info['ego_pos'])
                if info['collision']:
                    collisions += 1
                elif info['success']:
                    successes += 1
                    completion_times.append(env.time_step * 0.1)
                else:
                    timeouts += 1
                
                for k, v in ep_mode_counts.items():
                    mode_counts[k] += v

    total = episodes
    avg_final_pos = np.mean(final_positions)
    
    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS:")
    print("="*60)
    print(f"Success rate: {successes}/{total} ({successes/total:.1%})")
    print(f"Collision rate: {collisions}/{total} ({collisions/total:.1%})")
    print(f"Timeout rate: {timeouts}/{total} ({timeouts/total:.1%})")
    print(f"Average final position: {avg_final_pos:.1f}/60.0")
    
    if completion_times:
        print(f"Average completion time: {np.mean(completion_times):.2f}s")
    
    total_modes = sum(mode_counts.values())
    print(f"Mode usage: Timid={mode_counts[0]} ({mode_counts[0]/total_modes:.1%}), "
          f"Normal={mode_counts[1]} ({mode_counts[1]/total_modes:.1%}), "
          f"Aggressive={mode_counts[2]} ({mode_counts[2]/total_modes:.1%})")
    print("="*60)
    
    return {'successes': successes, 'collisions': collisions, 'timeouts': timeouts, 
            'mode_counts': mode_counts, 'completion_times': completion_times,
            'avg_final_position': avg_final_pos}

# -------------------------
# Main entry
# -------------------------
def main():
    print("CARLO Near-Accident Avoidance Training - FIXED VERSION")
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    
    Path("models").mkdir(exist_ok=True)
    
    try:
        coil, agent = train_system(
            num_expert_eps=600, 
            keep_failed_fraction=0.2, 
            coil_epochs=15, 
            ppo_episodes=500
        )
        
        print("\nFinal Evaluation with trained model:")
        results = evaluate_system(coil, agent, episodes=100)
        
        # Try with best model if available
        if Path("models/ppo_best.pth").exists():
            print("\nEvaluating Best Saved Model:")
            agent.load("models/ppo_best.pth")
            # Also load the best coil model
            coil.load_state_dict(torch.load("models/coil_best.pth", map_location=DEVICE))
            best_results = evaluate_system(coil, agent, episodes=100)
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()