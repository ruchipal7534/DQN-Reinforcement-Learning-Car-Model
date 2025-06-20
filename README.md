# Multi-Track Car Racing with Deep Q-Learning

A comprehensive autonomous racing simulation implementing Deep Q-Network (DQN) reinforcement learning for vehicle navigation. This project demonstrates self-supervised learning where an AI agent develops optimal driving strategies through environmental interaction without human demonstrations or labeled training data.

## Overview

This simulation combines real-time physics modeling with advanced reinforcement learning techniques to create an autonomous racing agent capable of navigating complex track geometries. The system features both manual play capabilities for human interaction and AI training modes for reinforcement learning research.

The agent employs a Dueling Double DQN architecture with curriculum learning across seven distinct track configurations, achieving human-level performance through pure environmental interaction. Training utilizes experience replay with prioritized sampling and progressive difficulty scaling to ensure robust generalization across diverse racing scenarios.

## Technical Architecture

### Neural Network Design
The DQN implementation features a 24-dimensional state vector comprising 15 ray-cast distance sensors and vehicle dynamics including velocity, acceleration, and angular momentum. The network architecture consists of three 256-neuron hidden layers with LayerNorm and dropout regularization, utilizing separate value and advantage heads for steering and acceleration outputs.

### Reinforcement Learning Framework
Training employs Double DQN with dueling architecture to address overestimation bias and improve learning stability. The agent uses epsilon-greedy exploration with adaptive decay from 1.0 to 0.1, while target network soft updates with τ=0.001 ensure stable Q-learning convergence. Experience replay maintains a 50,000-transition primary buffer supplemented by 10,000 high-impact experiences for prioritized sampling.

### Physics Simulation
The environment implements realistic vehicle dynamics including friction, momentum, and steering mechanics at 60 FPS. Collision detection utilizes precise geometric algorithms for boundary checking, while the 15-point sensor system provides comprehensive environmental awareness through ray-casting methods.

## Installation and Usage

### Requirements
```bash
pip install pygame torch numpy
```

### Quick Start
```bash
python main.py
```

The application provides four primary modes: manual play with keyboard controls, AI training visualization, trained model evaluation, and track environment preview. Manual controls use arrow keys or WASD for movement, with additional keys for reset, pause, and sensor visualization.

## Training Methodology

### Curriculum Learning
Training employs progressive difficulty scaling across track complexity. The agent begins with simple oval circuits before advancing through rectangular, L-shaped, U-shaped, curved, and figure-8 configurations. This curriculum approach ensures stable learning progression and improved generalization capabilities.

### Reward Structure
The multi-objective reward function balances forward progress incentives with collision avoidance penalties. Distance-based rewards encourage exploration, while proximity penalties and terminal crash costs promote safe navigation. Speed regulation components prevent excessive velocity near obstacles while maintaining racing performance.

### Performance Results
The system achieves basic navigation within 300 training episodes and optimal performance after 500+ episodes with curriculum learning. Final models demonstrate 85%+ track completion rates with sub-millisecond inference times, suitable for real-time autonomous navigation applications.

## Project Structure

```
├── main.py              # Application entry point
├── constants.py         # Global configuration
├── car.py              # Vehicle physics and sensors
├── track.py            # Environment generation
├── environment.py      # RL wrapper
├── dqn_network.py      # Neural network architecture
├── dqn_agent.py        # DQN implementation
├── training.py         # Training pipeline
└── models/             # Model persistence
```

## Track Environments

The simulation includes seven track configurations ranging from simple oval circuits to complex figure-8 designs. Each environment presents unique navigation challenges including sharp turns, variable radius curves, and multi-directional transitions. Track generation utilizes procedural algorithms with smooth boundary interpolation and configurable difficulty parameters.

## Technical Applications

This implementation serves as a foundation for autonomous vehicle research, reinforcement learning studies, and simulation development. The modular architecture supports algorithm modifications, environment extensions, and multi-agent scenarios. Key research applications include sensor fusion algorithms, decision-making frameworks, and curriculum learning methodologies.

The codebase demonstrates professional software architecture with clean separation of concerns, making it suitable for both academic research and industrial applications in autonomous systems development.