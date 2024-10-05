# Safe-MPC

This repo contains algorithms for Model Predictive Control, ensuring safety constraints and recursive feasibility.

## Installation
- Clone the repository\
`git clone https://github.com/EliasFontanari/safe-mpc.git`
- Install the requirements\
`pip install -r requirements.txt`
- Inside the root folder, download the zip (containing the Neural Network weights) from [here](https://drive.google.com/drive/folders/1RxXyuD6rPAJ7cdMhbY2nh_YfajpJ8Ku-?usp=sharing),
rename it as `nn_models.zip` and unzip it.
- Follow the instructions to install [CasADi](https://web.casadi.org/get/), [Acados](https://docs.acados.org/installation/index.html), [L4CasADi] (https://github.com/Tim-Salzmann/l4casadi) and [Pytorch](https://pytorch.org/get-started/locally/).

## Usage 
Configure the hyperparameters, such as the safety factor alpha and the dimension of the set of inital configurations to test in `config.yaml`.
Run the script `main.py` inside the root folder. One can consult the help for the available options:
```
python3 scripts/main.py --help
```
### Initial configurations finding
There are two methods:
- Without multiprocessing, it will found the initial configurations for the parameters set in `config.yaml` :
```
python3 scripts/main.py -i
```
- With multiprocessing: 
```
python3 scripts/init_parallelized.py
```
Both will save 3 files in `data` folder: they contain the state guesses, the control guesses for the first problems solved in complete SQP to find an initial feasible situation, and indeed the initial configurations.
Alternatively in data folder are already present the same files we used in tests.

### Guess refinement
After having found the initial configurations, it is necessary to generate the initial guesses for each controller one wants to test.
This is done as follows:
```
python3 scripts/main.py -g -c=${controller_of_interest}
```
The guesses are saved in `data` folder.
### Controller test
<!-- There are two methods:
- Without multiprocessing:
```
python3 scripts/main.py --rti -c=${controller_of_interest}
```
Each test executed will generate a folder in `data`. It contains data of simulations. Indeed results and states from which backup control starts are saved in `data`.    -->
With multiprocessing, in file `mpc_parallelized.py` set the controller to test and the number of processes, then run: 
```
python3 scripts/mpc_parallelized.py
```
For each test it saves in the folder `DATI_PARALLELIZED` a subfolder that contains safe abort states and all the data of simulations.

For both the methods, if one wants to test a parallel limited controller, it is necessary to set the number of computational units of the controller in `controller.yaml` and the method of allocation inside the constructor of the class `ParallelLimited` in the file `controller.py`.
The options are `self.constraint_mode = self.high_nodes_constraint` for Parallel High, `self.constraint_mode = self.uniform_constraint` for Parallel Uniform and `self.constraint_mode = self.CIS_distance_constraint` for Parallel Closest.

### Safe abort controller
In `controller.yaml` set alpha and settings equal to those of the previously executed test on which you want to execute the backup controland in file `backup_control.py` set also the controller for which the backup is executed, and run it:
```
python3 scripts/backup_control.py
```

## References
```bibtex
@misc{lunardi2023recedingconstraint,
      title={Receding-Constraint Model Predictive Control using a Learned Approximate Control-Invariant Set}, 
      author={Gianni Lunardi and Asia La Rocca and Matteo Saveriano and Andrea Del Prete},
      year={2023},
      eprint={2309.11124},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

