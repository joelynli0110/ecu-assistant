# ME Engineering Specs: ECU-800 Series Addendum

## Model Variant: ECU-850b (AI Enhanced)

The ECU-850b includes all features of the base ECU-850, with these upgrades:

- Dedicated AI Accelerator: a Neural Processing Unit (NPU) capable of 5 TOPS.
- Increased Memory: 4 GB LPDDR4.
- Higher Clock Speed: Cortex-A53 cores at 1.5 GHz.

## Full Technical Specifications: ECU-850b

| Feature | Specification |
| --- | --- |
| Processor | Dual-core ARM Cortex-A53 @ 1.5 GHz |
| NPU | 5 TOPS AI Accelerator |
| Memory (RAM) | 4 GB LPDDR4 |
| Storage | 32 GB eMMC |
| CAN Interface | Dual Channel, CAN FD up to 2 Mbps per channel |
| Ethernet | 1x 100BASE-T1 |
| Power Consumption | Idle: 550mA, Under Load: 1.7A |
| Operating Temperature | -40°C to +105°C |

## NPU Configuration

To enable the NPU, run:

```bash
me-driver-ctl --enable-npu --mode=performance
```
