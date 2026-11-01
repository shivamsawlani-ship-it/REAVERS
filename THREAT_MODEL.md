# CAVShield Threat Model & System Security Boundaries

## 1. Executive Summary & Honest System Boundary

CAVShield is an **inline automotive gateway** and **cyber-physical intrusion detection and prevention system (IDPS)**. It operates at the network and message layer of connected and autonomous vehicles (CAVs):
- **Controller Area Network (CAN 2.0B / CAN-FD)**: Sits as an active, inline gateway between untrusted domain segments (infotainment, telematics, OBD-II port) and safety-critical vehicle actuator buses (powertrain, brakes, steering).
- **Vehicle-to-Everything (V2X / IEEE 1609.2 / ETSI ITS-G5)**: Sits as a software verification firewall validating incoming Basic Safety Messages (BSMs) and Cooperative Awareness Messages (CAMs) before forwarding them to ADAS planning engines.

### Crucial Distinction: What CAVShield Detects vs. Threat Model Assumptions

> [!IMPORTANT]
> **CAVShield does NOT claim to detect radio frequencies, physical connectors, or RF exploits.**
> CAVShield has no physical antennas or wireless baseband decoders to detect Bluetooth L2CAP packets, cellular LTE/5G baseband exploits, or physical pin insertions at the OBD-II port.
>
> When the dashboard, incident logs, or forensic reports show an **"Assumed Entry Point"**, this represents an **architectural threat modeling assumption (simulated scenario)** based on the standard automotive attack surface (UN R155 / ISO/SAE 21434). It is explicitly **not** a hardware or physical layer detection verdict.

---

## 2. Threat Modeling Matrix

| Attack Identifier | Attributed Threat Type | Assumed Entry Point (Simulation Assumption) | What CAVShield Observes at Gateway | CAVShield Enforcement & Mitigation Response |
|---|---|---|---|---|
| **`spoof_brake`** | Spoofing (Brake) | Compromised infotainment (Bluetooth/Wi-Fi) or telematics (cellular) | Hostile 0x200 `BRAKE_COMMAND` (7.0 m/s²) arriving while brake pedal = 0% and no radar obstacle | Inline gateway drops frame in < 0.15 ms; digital twin cruise speed is preserved. |
| **`spoof_accel`** | Spoofing (Accel) | Compromised infotainment (Bluetooth/Wi-Fi) or telematics (cellular) | Hostile 0x201 `ACCEL_COMMAND` (+4.5 m/s²) arriving without accelerator pedal actuation | Inline gateway drops frame; uncommanded acceleration is blocked. |
| **`spoof_speed`** | Spoofing (Speed) | Compromised infotainment (Bluetooth/Wi-Fi) or telematics (cellular) | 0x101 `VEHICLE_SPEED` reporting 125 km/h while independent wheel encoders report 72 km/h | Gateway rejects corrupt telemetry; failsafe enters DEGRADED sensor fallback mode. |
| **`spoof_rpm`** | Spoofing (RPM) | Compromised infotainment (Bluetooth/Wi-Fi) or telematics (cellular) | 0x100 `ENGINE_RPM` reporting 7,000 RPM while vehicle speed and gear ratio dictate ~2,760 RPM | Physics invariant check rejects frame; alerts dashboard. |
| **`spoof_gear`** | Spoofing (Gear) | Compromised infotainment (Bluetooth/Wi-Fi) or telematics (cellular) | 0x104 `GEAR_POSITION` commanding Reverse (gear=1) while vehicle speed > 5 km/h | Kinematic interlock check drops transmission command; prevents mechanical damage. |
| **`flood`** | Flooding | Malicious OBD-II dongle / physical diagnostic tool | High-rate bus saturation (700+ Hz) on high-priority IDs (0x000, 0x0A0) | Token-bucket rate limiter sheds flood frames in 2.2 µs; prevents bus starvation. |
| **`fuzz`** | Fuzzing | Malicious OBD-II dongle / rogue diagnostic tool | High-rate random arbitration IDs and non-conforming 8-byte bit patterns | DBC whitelist drops unregistered IDs; checksum validation drops corrupted frames. |
| **`replay`** | Replay | Recorded traffic replayed via a compromised ECU node | Stale alive counters, static payloads, and out-of-schedule timing matching past buffers | Sequential alive counter check and inter-arrival timing checks reject replayed frames. |
| **`v2x_phantom`** | V2X Phantom Obstacle | Compromised peer vehicle (valid key) or compromised on-board V2X unit | Validly signed BSM hazard broadcast claiming road obstacle 40 m ahead in ego lane | Multi-sensor cross-check reveals radar/LiDAR clear line-of-sight; phantom hazard rejected. |
| **`v2x_fake_ebrake`**| V2X Fake E-Brake | Malicious rogue V2X transmitter (spoofed identity) | Unauthenticated BSM message claiming emergency deceleration (9 m/s²) with invalid signature | Cryptographic verification fails; message dropped before reaching ADAS planner. |
| **`v2x_gps`** | V2X GPS Drift | Spoofed GPS position broadcast / compromised V2X beacon | Discontinuous kinematic jump (> 150 m in 0.1 s) inconsistent with vehicle velocity limits | Kinematic plausibility validator rejects beacon; prevents phantom tracking. |

---

## 3. Core System Security Assumptions

The effectiveness and functional safety guarantees of CAVShield rely on the following explicit assumptions:

1. **Trusted Gateway Root of Trust (Gateway Trust Assumption)**:
   - The gateway hardware platform (e.g. NXP S32G, Infineon AURIX TC4xx, or Renesas R-Car) and its real-time operating system (RTOS) microkernel are trusted.
   - The gateway enforces secure boot and cryptographic memory protection (ARM TrustZone / HSM). An adversary who achieves ring-0 kernel code execution on the gateway microcontroller is outside the defense model.
2. **Sensor Independence from Untrusted Bus (Sensor Separation Assumption)**:
   - Vehicle kinematic plant validation relies on secondary ground truth: wheel speed encoders, 6-DoF inertial measurement units (IMU), and forward Radar/LiDAR sensors.
   - These perception channels communicate with the gateway over physically isolated point-to-point links (e.g., private SPI, automotive Ethernet, or dedicated CAN FD subnetworks) that cannot be directly flooded or injected from the infotainment or OBD-II segments.
3. **Automotive Bus Topology & Inline Placement**:
   - CAVShield is situated as an **inline bridge**, not a passive sniffer. Frames cannot reach chassis actuators (electronic brake booster, electronic power steering, throttle controller) without passing through CAVShield's fast-path decision gate.
4. **V2X Cryptographic Key Management**:
   - The vehicle maintains an authenticated trust store containing authorized root certificates (IEEE 1609.2 PKI / shared HMAC secrets). Compromised private keys can be revoked via Certificate Revocation Lists (CRLs).

---

## 4. Out-of-Scope Items, Limitations & Roadmap

The following threat vectors are explicitly out of scope for the current software-defined CAVShield prototype:

### 1. Physical Sensor Spoofing & Blinding
- **Threat Vector**: Laser blinding (dazzling) of optical LiDAR lenses, acoustic jamming of ultrasonic parking sensors, or millimeter-wave radar ghost-target generation.
- **Current Posture**: CAVShield assumes that the primary physical sensors deliver unjammed telemetry to the gateway.
- **Roadmap (v2.0)**: Implementation of raw multi-sensor fusion cross-validation (e.g. radar doppler velocity vs. optical flow vs. wheel odometry) to detect physical sensor spoofing at the perception layer.

### 2. GPS RF Constellation Spoofing
- **Threat Vector**: Radio-frequency spoofing of civilian GNSS/GPS L1/L5 signals using software-defined radios (SDRs).
- **Current Posture**: CAVShield detects kinematics-violating position jumps (> 150 m jumps in V2X messages), but does not inspect RF signal arrival angle or carrier-to-noise ratios.
- **Roadmap (v2.0)**: Integration of GNSS carrier phase continuity verification and dual-antenna GNSS heading cross-checks.

### 3. Compromised Gateway Microcontroller / HSM
- **Threat Vector**: Firmware flashing attacks or hardware fault injection (voltage glitching) on the gateway chip.
- **Current Posture**: Out of scope; protected by hardware-level secure boot and write-protected flash memory.
- **Roadmap (v2.0)**: Formal verification of gateway fast-path logic using seL4 microkernel architecture and dual-core lockstep hardware.

### 4. Direct Actuator Wiretapping (Downstream Splice)
- **Threat Vector**: An adversary physically cutting harness wires between the gateway and the brake actuator to inject signals directly.
- **Current Posture**: Out of scope for a centralized gateway; physical harness protection is an OEM mechanical design responsibility.
- **Roadmap (v2.0)**: AUTOSAR Secure Onboard Communication (SecOC) deployment with end-to-end symmetric MAC authentication directly to smart actuators.

