<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=200&section=header&text=🧠%20SARSA%20Cliff%20Walking&fontSize=48&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Reinforcement%20Learning%20with%20On-Policy%20TD%20Control&descAlignY=60&descAlign=50" width="100%"/>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-CliffWalking--v1-00BFFF?style=for-the-badge&logo=openai&logoColor=white)](https://gymnasium.farama.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
</div>

---

## 📌 Project Overview

The **Cliff Walking Problem** is a classic Reinforcement Learning benchmark where an agent must navigate a 4×12 grid from a **Start** state to a **Goal** state while avoiding a cliff region that gives a heavy penalty of **-100**.

This project implements **SARSA (State-Action-Reward-State-Action)** — an **on-policy TD control** algorithm — to train an agent that learns a safe, reward-maximizing path over **500 episodes** using an **ε-greedy policy**.

Unlike Q-Learning (off-policy), SARSA accounts for the agent's own exploratory behavior during training, causing it to learn a **safer path** away from the cliff edge — even if it's slightly longer. This is the fundamental difference between on-policy and off-policy RL.

---

## 📂 Environment

| Property | Value |
|:---|:---|
| Environment | `CliffWalking-v1` (Gymnasium) |
| Grid Size | 4 rows × 12 columns = **48 states** |
| Action Space | 4 discrete actions (Up, Down, Left, Right) |
| Start State | Bottom-left corner |
| Goal State | Bottom-right corner |
| Cliff Penalty | **-100** per step into cliff |
| Step Penalty | **-1** per normal step |

---

## 🔄 Pipeline Workflow

```
Environment Init → Q-Table Init → ε-Greedy Policy → SARSA Update → Convergence → Greedy Rollout
```

### 1️⃣ Environment Setup
- Initialized `CliffWalking-v1` from Gymnasium
- Q-Table of shape `(48, 4)` initialized to zeros

### 2️⃣ Hyperparameter Configuration
- **α (learning rate):** `0.5` — Controls how fast Q-values update
- **γ (discount factor):** `0.99` — Prioritizes long-term rewards
- **ε (epsilon):** `0.1` — 10% random exploration via ε-greedy policy
- **Episodes:** `500`

### 3️⃣ SARSA On-Policy Training Loop
- Agent selects action using ε-greedy policy
- Takes step → observes `(next_state, reward, done)`
- Selects **next_action** using the same ε-greedy policy
- Updates Q-value using the SARSA rule:

```
Q(s,a) ← Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]
```

### 4️⃣ Greedy Rollout (Inference)
- After training, environment is re-initialized with `render_mode="human"`
- Agent navigates using **pure greedy policy** (`argmax Q[state]`)

---

## 🤖 Algorithm

### 1️⃣ SARSA — On-Policy TD Control ⭐ Best (Only) Model

```python
Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
```

- **On-policy:** learns Q-values based on the same policy it's currently following (including exploration)
- **TD(0):** single-step bootstrapping — no need for full episode rollouts
- **ε-greedy:** balances exploration vs exploitation throughout training
- Converges to a **near-optimal, safer path** — avoids cliff edge due to exploration risk awareness
- Key SARSA property: tends to choose **inland routes** over cliff-edge shortcuts, unlike Q-Learning

---

## 📊 Results

| Metric | Value |
|:---|:---|
| Total States | 48 |
| Total Actions | 4 |
| Training Episodes | 500 |
| Learning Rate (α) | 0.5 |
| Discount Factor (γ) | 0.99 |
| Exploration Rate (ε) | 0.1 |
| Convergence | ~200–300 episodes |
| Learned Policy | Safe inland path (avoids cliff edge) |

> After training, the greedy rollout consistently navigates from Start to Goal without falling off the cliff, demonstrating successful policy convergence.

---

## 🔍 Key Insights

- 🧠 **SARSA is on-policy** — it updates Q-values using the action *actually taken* by the current policy (including exploratory moves), making it more conservative near the cliff
- 📉 **γ = 0.99** ensures the agent strongly values future rewards, preventing greedy short-term cliff-edge paths
- 🔄 **ε = 0.1** means 10% random exploration during training — SARSA accounts for this risk, unlike Q-Learning which ignores exploration in its updates
- ⚠️ **The fundamental RL trade-off**: SARSA learns a safer-but-longer route; Q-Learning learns the optimal-but-riskier cliff-edge path
- 📈 Total reward per episode generally improves across 500 episodes, stabilizing as the Q-table converges

---

## 🗂️ Repository Structure

```
SARSA-Cliff-Walking-Problem/
│
├── SARSA_Cliff_Walking_Problem.py   # Main training + inference script
├── requiredment.txt                 # Python dependencies
├── LICENSE                          # MIT License
└── README.md                        # Project documentation
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/ronakrajput8882/SARSA-Cliff-Walking-Problem.git
cd SARSA-Cliff-Walking-Problem

# Install dependencies
pip install -r requiredment.txt

# Run SARSA training + greedy rollout
python SARSA_Cliff_Walking_Problem.py
```

> The script will print per-episode reward and episode length for all 500 training episodes, then open a visual render of the learned greedy policy.

---

## 🧠 Key Learnings

- SARSA's on-policy nature fundamentally shapes the path it learns — safety over optimality
- The SARSA update rule bootstraps from the **next action**, not the best possible action — this is what differentiates it from Q-Learning
- Q-table convergence on small environments like CliffWalking can be observed visually within a few hundred episodes
- Gymnasium's `render_mode="human"` provides an intuitive way to visualize RL policy behavior post-training
- Setting `gamma` close to 1 is critical for long-horizon navigation tasks

---

## 🛠️ Tech Stack

| Tool | Use |
|:---|:---|
| Python 3.10+ | Core language |
| NumPy | Q-table representation & argmax policy |
| Gymnasium (`CliffWalking-v1`) | RL environment |
| Random | ε-greedy exploration |

---

<div align="center">

### Connect with me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ronaksinh-rajput8882)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/techwithronak)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ronakrajput8882)

*If you found this useful, please ⭐ the repo!*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=100&section=footer" width="100%"/>

</div>
