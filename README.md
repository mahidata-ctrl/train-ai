# Maximizing Section Throughput Using AI-Powered Precise Train Traffic Control

This project demonstrates how Deep Reinforcement Learning (DQN) can be used to optimize train movements in a 20 km railway section, with the goal of **maximizing throughput** (trains per hour) while maintaining safe headway.

## Features

- **AI‑assisted driver advisory** – Select any train from a list of 100+ realistic Indian trains and get a recommended speed based on the current distance to the train ahead and behind.
- **Throughput simulation** – Compare the number of trains that complete the section **with and without AI control** over a user‑defined period. Trains are recycled to maintain constant density.
- **Realistic environment** – Each train observes its own speed and the distances to its immediate neighbours. The reward function encourages speeds of 30‑100 km/h and a safe headway of 1.5‑3 km.
- **Interactive visualisation** – Track view shows all train positions; the throughput tab displays cumulative completions and a performance comparison.

## How It Works

1. **Training** – A DQN agent is trained in a custom Gym environment (`SingleTrainControlEnv`) where it learns to accelerate/decelerate to balance speed and safety.
2. **Driver advisory** – The trained model is used in real time to suggest speed adjustments for a manually selected train.
3. **Throughput analysis** – A multi‑train simulation runs two scenarios:
   - **Baseline**: trains change speed randomly.
   - **AI‑controlled**: every train uses the same DQN policy to decide its speed every 10 seconds.
   Trains that reach the end (20 km) are counted and replaced by new trains at the start, allowing steady‑state throughput measurement.

## Installation

```bash
pip install -r requirements.txt
