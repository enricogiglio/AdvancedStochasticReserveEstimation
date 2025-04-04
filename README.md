# AdvancedStochasticReserveEstimation
This repository implements a stochastic reserve estimation model for power systems. It simulates up/down reserve needs considering outages, load and RES uncertainty, and ramping limits. Sobol sequences and Monte Carlo simulation are used to derive reserve estimates in line with ENTSO-E guidelines.

## Overview

The repository includes Python scripts for:
- Creation of the network over which the Sobol sequence is created and Monte Carlo simulation for stochastic power reserve estimation is performed.
- Generating sawtooth sequences to simulate ramping dynamics.
- Evaluating tripping events and their impact on reserve requirements.
- Evaluating frequency imbalance due to unpredictable load variations and renewable generation.
- Calculating reserve needs (FCR, aFRR, mFRR, RR) using a stochastic approach.
- Building reserve allocation strategies for different network configurations.

## Repository Structure
- `CreatorFileInputNetwork.py`: It creates the network over which the Sobol sequence is created 
- `CreatorCombinationsSobol.py`: It creates the Sobol sequence over which Monte Carlo simulation for stochastic power reserve estimation is performed. 
- `StochasticReserveSizing.py`: Main script containing functions for reserve estimation.

## Getting Started
### Prerequisites
- Required Python packages: pypsa, numpy, pandas, sklearn, matplotlib

### Usage
- Create your network `CreatorFileInputNetwork.py` at it will be saved in folder `CreatorFileInput`
- Create the Sobol sequence with `CreatorCombinationsSobol.py`. The Sobol sequence is saved in the `ModelloStocastico` folder. In the script, you must select:
	a) the maximum residual demand variation to be considered
	b) the length of the Sobol sequence (number of configurations).
- Run the main script
```
 python StochasticReserveSizing.py
```


# Authors
These codes were developed in collaboration with @davide-f and @edopasta
