# Permutation group sequential design simulations

This repository contains simulation scripts and their results meant to verify the error control and statistical power characteristics of the group sequential testing procedure implemented in the [niseq](https://github.com/john-veillette/niseq) package. 

* `FPR-simulation.ipynb` contains simulations to estimate the false positive rate of sequential permutation tests vs. regular permutation tests used with optional stopping.
* `erp_simulation.py` contains code to run power analyses by simulation for fixed-sample and sequential designs. It takes a long time to run a reasonable number of simulations, so we ran this on UChicago's Midway2 cluster.
  * `erp_simulation.py` draws from the files in the `erpcore` directory to perform its power analyses, which contains seven widely used evoked EEG responses measured from 40 different subjects.
  * The `results` directory contains the output of `erp_simulation.py`, which includes the parameters used for each simulation as well as the power estimates.
  * `results.ipynb` visualizes the results in `results` to compare the efficiency of fixed-sample and sequential designs.
* Environment files are provided specifying the software dependencies required to reproduce our computational environment.
  * `environment.yml` is the [conda](https://docs.conda.io/en/latest/) environment file we used to create our simulation environment. It will recreate our environment pretty accurately, but exact versions of some packages might vary depending on e.g. your operating system. We doubt any such minor differences would affect simulation results, but we provide a more detailed environment file as well just in case.
  * `manifest.yml` contains the specific package versions installed in our final environment. It should allow you to recreate our environment exactly on most Linux machines, or you could use it to create a [Docker](https://www.docker.com/) container or [Binder](https://mybinder.org/) environment if you don't have a Linux machine. 
