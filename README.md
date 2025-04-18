<img src="https://i.postimg.cc/NFPQmPB0/Mask-group.png" style="width: 50%;" />


# Project BEACON  
A Decentralized, Multi‑Layered Approach for the Convergence and Optimization of Hybrid LoRa–Web Communication Networks

![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Static Badge](https://img.shields.io/badge/NodeJS-green)
![Static Badge](https://img.shields.io/badge/LoRa-red)
![Static Badge](https://img.shields.io/badge/ESP32-blue)
![MacOS](https://img.shields.io/badge/MacOS-green)
![Windows](https://img.shields.io/badge/Windows-green)

Boot up the beacon device first by connecting it via a Micro‑USB cable. Once powered on, the device will generate its own Wi‑Fi access point—connect your computer or mobile device to that AP.

Next, in your local development environment:

```bash
cd ui
npm install
npm start
```

Then open your browser to http://localhost:3000. The web app will connect to the beacon’s AP and communicate over a REST API.

# Overview
Project BEACON provides a decentralized, multi‑layered bridge between LoRa radio networks and standard web clients. It enables browsers or HTTP‑capable devices to send and receive LoRa messages via a locally hosted REST API.

# Device Initialization
Power-up: Connect the beacon hardware to USB power via Micro‑USB.

Wi‑Fi AP: On boot, the device creates its own Wi‑Fi access point—clients simply join that network.

# Communication Flow
HTTP ↔ Beacon: The web app issues REST calls to the beacon’s local IP.

Beacon ↔ LoRa: Beacon translates incoming HTTP requests into outbound LoRa packets (and vice versa).

Bi‑directional Bridge: Enables seamless end‑to‑end messaging between web clients and remote LoRa nodes.

# Key Benefits
Decentralized: No cloud dependency—everything runs locally.

Multi‑layered: Combines long‑range, low‑power LoRa with familiar HTTP/Web interfaces.

Extensible: Testbed for custom routing logic, resilience strategies, and mixed‑media IoT scenarios.

# Typical Use Cases
Field deployments where cellular/internet is unreliable

Rapid prototyping of LoRa‑enabled sensor networks

Educational/demo setups illustrating cross‑stack IoT communication
