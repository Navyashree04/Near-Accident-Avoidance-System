
# Near Accident Avoidance System

A hierarchical reinforcement learning system for autonomous intersection navigation using Conditional Imitation Learning (CoIL) and Proximal Policy Optimization (PPO).

---

## Project Overview

This project implements a two-level hierarchical control system for autonomous vehicles navigating intersections while avoiding collisions. The system combines:

* High-Level Policy (PPO): selects driving modes (timid, normal, aggressive)
* Low-Level Policy (CoIL): executes continuous throttle control
* Expert Demonstrations: rule-based Time-To-Collision (TTC) decision making

---

## Key Features

* 94% success rate in intersection navigation
* Zero collisions in final evaluation
* Multi-modal driving behavior selection
* Stable PPO training with GAE and entropy regularization
* Conditional Imitation Learning with shared backbone and specialized branches

---

## Performance Results

FINAL EVALUATION RESULTS (100 episodes):

* Success rate: 94/100 (94.0%)
* Collision rate: 0/100 (0.0%)
* Timeout rate: 6/100 (6.0%)
* Average completion time: 6.42 seconds
* Mode usage: Timid=87.9%, Normal=7.1%, Aggressive=5.0%

---

## System Architecture

### Hierarchical Structure

High-Level (PPO Agent) → Mode Selection → Low-Level (CoIL Network) → Throttle Control → CARLO Environment

### Main Components

1. CARLO Environment

   * Simulated intersection navigation
   * Continuous throttle control (-1 to 1)
   * TTC-based rewards and collision detection

2. Expert Policy

   * Rule-based TTC calculations
   * Three behavioral modes: timid, normal, aggressive

3. CoIL Network

   * Shared feature extraction backbone
   * Mode-specific regression branches
   * Behavior cloning from expert demonstrations

4. PPO Agent

   * Actor-Critic with Generalized Advantage Estimation
   * Entropy regularization for exploration stability

---

## Installation

### Prerequisites

* Python 3.8 or higher
* PyTorch 1.9 or higher
* NumPy, Matplotlib, tqdm

### Quick Start

1. Clone the repository: `git clone https://github.com/Navyashree04/Near-Accident-Avoidance-System.git`
2. Navigate into the folder: `cd near-accident-avoidance`
3. Install dependencies: `pip install torch numpy matplotlib tqdm`
4. Run training: `python main.py`

### Manual Installation (Conda Environment)

1. Create environment: `conda create -n near-accident python=3.8`
2. Activate environment: `conda activate near-accident`
3. Install PyTorch: `pip install torch torchvision torchaudio`
4. Install other dependencies: `pip install numpy matplotlib tqdm pathlib`

---

## Project Structure

* models/

  * coil_best.pth
  * coil_final.pth
  * ppo_best.pth
  * ppo_final.pth
* main.py
* requirements.txt
* README.md
* reports/

  * final_report.md

---

## Usage

### Train the System

```python
from main import train_system

coil, agent = train_system(
    num_expert_eps=600,
    keep_failed_fraction=0.2,
    coil_epochs=15,
    ppo_episodes=500
)
```

### Evaluate the System

```python
from main import evaluate_system

results = evaluate_system(coil, agent, episodes=100)
print(f"Success rate: {results['successes']}%")
```

### Run a Custom Environment

```python
env = CARLOEnvironment()
obs = env.reset()

done = False
while not done:
    mode_idx = agent.act(obs)
    action = coil.predict(obs, mode_idx)
    obs, reward, done, info = env.step(action)
```

---

## Configuration

### Environment Parameters

* ego_start_pos = 0.0
* ego_start_vel = 10.0
* ado_pos_range = (60.0, 100.0)
* ado_vel_range = (4.0, 8.0)
* crossing_pos = 50.0
* success_pos = 60.0
* max_steps = 200

### Training Hyperparameters

**CoIL**

* Learning rate: 1e-3
* Batch size: 256
* Epochs: 15

**PPO**

* Learning rate: 1e-4
* Gamma: 0.99
* Lambda: 0.95
* Epsilon: 0.2
* Epochs: 4
* Entropy coefficient: 0.1

---

## Training Process

**Phase 1: Expert Data Generation**

* Generate 600 expert episodes
* TTC-based rule policy
* 94% expert success rate

**Phase 2: CoIL Training**

* Behavior cloning from expert data
* Validation loss: 0.006253

**Phase 3: PPO Training**

* Reinforcement learning with CoIL controller
* 500 training episodes
* Final success rate: 94%

---

## Driving Modes

* **Timid:** Early braking, conservative acceleration, Usage: 87.9%
* **Normal:** Balanced control and moderate acceleration, Usage: 7.1%
* **Aggressive:** Late braking, assertive acceleration, Usage: 5.0%

---

## Customization

**Modify Reward Function**

```python
def custom_reward_function(self, info):
    if info['collision']:
        return -20.0
    elif info['success']:
        return 50.0 + time_bonus
    else:
        return progress_reward + speed_reward
```

**Add New Driving Mode**

```python
self.branches.append(new_branch_network)

def expert_policy(obs, mode='custom'):
    if mode == 'custom':
        return custom_behavior(obs)
```

---

## References

* Codevilla, F. et al., "End-to-End Driving via Conditional Imitation Learning", ICRA 2018
* Schulman, J. et al., "Proximal Policy Optimization Algorithms", arXiv 2017
* Kendall, A. et al., "Learning to Drive in a Day", ICRA 2019

---

## Troubleshooting

* **Training instability:** Reduce learning rate, increase batch size, use gradient clipping
* **Poor expert performance:** Adjust TTC thresholds, modify reward weights
* **Memory issues:** Reduce batch size, clear GPU cache, simplify network architecture

**Performance Tips:**

* Use GPU for faster training
* Monitor metrics during training
* Save model checkpoints regularly

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m "Add YourFeature"`)
4. Push to your branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

```
