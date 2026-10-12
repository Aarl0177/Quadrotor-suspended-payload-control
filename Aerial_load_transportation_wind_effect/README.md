# Quadrotor Suspended-Payload Control in Windy Environments

MATLAB/Simulink implementation associated with the published paper:

**A. A. Rezaei Lori, M. Danesh, P. Amiri, S. Y. Ashkoofaraz, and M. A. Azargoon,  
“Transportation of an Unknown Cable-Suspended Payload by a Quadrotor in Windy Environment under Aerodynamics Effects,”  
2021 7th International Conference on Control, Instrumentation and Automation (ICCIA), IEEE.**

[IEEE Paper](https://ieeexplore.ieee.org/document/9403559)

## Overview

This project investigates control of a quadrotor transporting an unknown cable-suspended payload in a windy environment.

The nonlinear dynamics of the quadrotor-payload system are derived using the Euler-Lagrange formulation, including translational and rotational aerodynamic drag effects. A sliding-mode controller is used for translational motion to address wind disturbances, payload swing, and uncertainty in payload mass and cable length. A PD controller is used for quadrotor attitude control. Wind disturbances are modeled using the Dryden turbulence model. :chatgpt-content-reference{index="0"}

## Implementation

The control system and dynamic model were implemented and tested in MATLAB/Simulink.

The Simulink implementation includes:

- Quadrotor and suspended-payload dynamics
- Position control using Sliding Mode Control
- Attitude control using PD control
- Payload mass and cable-length uncertainty
- Aerodynamic drag effects
- Payload swing disturbances
- Dryden wind turbulence model
- Closed-loop trajectory-tracking simulation

## Repository Structure

```text
.
├── src/
│   └── Quadrotor_aerial_load_wind_effect.slx
│
├── results/
│   └── Simulation results in SVG format
│
└── README.md
