# Project: Sod shock tube

## The program

The program numerically solves the special relativistic Euler equations in one dimension. It was tested on the Sod shock tube problem. The solution is computed using a RHLLE scheme. It is visualized in plots and animations. The final state is saved as .csv files. The results can be compared to the exact solution from the program RIEMANN.f (see [Martí and Müller 1999](https://link.springer.com/article/10.12942/lrr-1999-3)).

## Requirements

- Python 3 with libraries NumPy, Matplotlib, SymPy, sys, io
- FFmpeg

## Usage
1. configuration in config.yml
    - file already contains special relativistic shock tube problem with recommended values
2. execution of main.py
    - runs simulation
    - creates plots
    - renders animations
3. execution of RIEMANN.f from [Martí and Müller 1999](https://link.springer.com/article/10.12942/lrr-1999-3) (not contained in repository)
    - computes exact solution
4. execution of comparison.py
    - compares RHLLE and exact solution
    - creates plots
    - computes error
  
## Results
The results and the exact solution for the given config.yml are included in the repository. 
