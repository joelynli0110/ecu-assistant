# ME Engineering Specs: ECU-800 Series

## Overview

The ECU-800 Series is the next-generation platform for ADAS and infotainment.
This document covers the baseline model: ECU-850.

## Key Features

- Over-the-Air (OTA) Update Capability
- Secure Boot and Hardware Security Module (HSM)
- AI/ML Acceleration (see ECU-850b for enhanced NPU)

## ECU-850 Technical Specifications

| Feature | Specification |
| --- | --- |
| Processor | Dual-core ARM Cortex-A53 @ 1.2 GHz |
| Memory (RAM) | 2 GB LPDDR4 |
| Storage | 16 GB eMMC |
| CAN Interface | Dual Channel, CAN FD up to 2 Mbps per channel |
| Ethernet | 1x 100BASE-T1 |
| Power Consumption | Idle: 500mA, Under Load: 1.5A |
| Operating Temperature | -40°C to +105°C |

## Software Configuration

The ECU-850 runs a Yocto-based Linux OS. Application deployment is managed via
Docker containers.

