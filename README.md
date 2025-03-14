# Video Streaming Optimization in UAV Networks

This repository contains a Python implementation for optimizing video streaming quality in a UAV (Unmanned Aerial Vehicle) network by adjusting channel fading thresholds and video encoding rates.

## Overview

This project simulates a wireless network including ground-aerial nodes where some nodes act as video streamers or potential interferers. The code provides algorithms to optimize video transmission parameters for maximizing the average Peak Signal-to-Noise Ratio (PSNR) of the received video under realistic wireless channel conditions.

## Features

- **Probabilistic Line-of-Sight (LoS) Model**: Calculates the probability of line-of-sight connections between ground-aerial nodes based on their positions and environmental parameters.
- **Channel Fading Models**: Implements both Rayleigh (NLoS) and Rician (LoS) fading models for wireless channels.
- **Interference Modeling**: Models interference using the log-normal distribution.
- **Packet Loss Analysis**: Calculates packet loss probability based on:
  - Buffer overflow probability
  - Time threshold probability
  - Transmission error probability
- **PSNR Modeling**: Calculates PSNR based on packet loss distortion and compression distortion.
- **Throughput Modeling**: Calculates throughput based on overall packet loss.
- **UAV Position Adjustment**: Functions to adjust UAV positions based on distance or elevation angle.
- **Optimization Algorithms**:
  - `DVTC`: Distributed Video Transmission Control (DVTC) for Channel fading threshold optimization
  - `DVEC`: Distributed Video Encoding Control (DVEC) for video encoding rate optimization
  - `JDVT_EC`: Joint Distributed Video Transmission and Encoder Control (JDVT_EC) for joint optimization

## Requirements

- Python 3.x
- NumPy
- SciPy

## Usage

The main parameters are defined at the beginning of the code and can be adjusted according to the specific network scenario:

```python
F = 14          # Number of sub-channels
Tslt = 0.005    # Time slot duration
Tth = 0.08      # Time threshold
SINRTh = 10     # SINR threshold
Omega = 2       # Rayleigh Fading parameter
lambda0 = 0.001 # Network density
```

To run the joint optimization algorithm:

```python
# mp contains the indices of streamer nodes
result_beta, result_lambda = JDVT_EC(mp)
```

### Adjusting UAV Positions

The code includes functions to adjust UAV positions:

```python
# Adjust distance while maintaining the angle
old_angle, new_angle, old_dis, new_dis = adj_dis(streamer_index, receiver_index, additional_distance)

# Adjust elevation angle while maintaining the distance
old_angle, new_angle, old_dis, new_dis = adj_angle(streamer_index, receiver_index, additional_angle)
```

## Technical Details

### Network Model

The system simulates a 3D space where ground-aerial nodes are randomly distributed by the Poisson point process (PPP). Each node can act as:
- A video streamer (transmitter)
- A receiver
- An interferer

### Wireless Channel Model

The wireless channel incorporates:
- Path loss based on distance and line-of-sight probability
- Small-scale fading (Rayleigh/Rician)
- Interference from other transmissions

### Video Streaming Performance Metrics

The code calculates:
- Overall pakcet loss
- Expected throughput
- Packet loss distortion
- Compression distortion
- PSNR

### Optimization Approach

The optimization algorithm uses a consensus-based distributed approach to iteratively adjust:
1. Channel fading threshold (Beta)
2. Video encoding rate (Lambda)

## Citation

If you use this code in your research, please cite:

```
@article{Ghazikor-2024-Channel,
  title={Channel-Aware Distributed Transmission Control and Video Streaming in UAV Networks},
  author={Ghazikor, Masoud and Roach, Keenan and Cheung, Kenny and Hashemi, Morteza},
  journal={arXiv:2408.01885},
  year={2024}
}
```
