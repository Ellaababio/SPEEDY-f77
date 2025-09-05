## Running the SPEEDY T21 Data Assimilation Experiment

This guide describes the complete workflow for running a T21 data-assimilation experiment using the AMLCS Python framework on an HPC environment. The process involves three main stages:

1. Pre-Processing — generate a "true" reference solution and an initial ensemble of model states.  
2. Data Assimilation — run the main forecast-assimilation cycles.  
3. Post-Processing — plot results and analyze performance.

This guide assumes the legacy Fortran model (imp.exe) has already been compiled as described in the main README.md.

---

### Prerequisites

- Access to the HPC and Slurm scheduler.
- Anaconda/Conda available as an HPC module.
- The AMLCS repository checked out (scripts located under AMLCS/amlcs/).
- Replace `/gpfs/home/your_username/` with your actual home directory path in all examples below.

---

### 1. Create the Conda environment

On the HPC, load Anaconda and create the environment once:

```bash
# Load Anaconda module
module load anaconda/3.11.5

# Create environment (run once)
conda create -n speedy_da_env python=3.11 netcdf4 pandas scikit-learn scipy matplotlib seaborn

# Initialize conda for your shell (then log out/in)
conda init bash
```

Activate the environment in job scripts using:

```bash
eval "$(conda shell.bash hook)"
conda activate speedy_da_env
```

---

### 2. Run the experiment — 3 Slurm jobs in sequence

The experiment is executed as three separate Slurm jobs. Create one script for each stage and submit them sequentially (wait for each to finish before submitting the next).

#### 2.1 Pre-Processing (generate initial ensemble and reference)

Save as `launch_preprocess.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=DA_PREP
#SBATCH --output=DA_PREP_%j.out
#SBATCH --error=DA_PREP_%j.err
#SBATCH --account=chipilskigroup_q
#SBATCH --partition=chipilskigroup_q
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12g

echo "--- Pre-processing Job Started ---"

module purge
module load gnu/4.8.5
module load anaconda/3.11.5
eval "$(conda shell.bash hook)"
conda activate speedy_da_env
export LD_LIBRARY_PATH=$HOME/speedy_libs/lib:$LD_LIBRARY_PATH

cd amlcs/

srun python amlcs_pre.py amlcs_pre_t21.csv

echo "--- Pre-processing Job Finished ---"
```

Submit:

```bash
sbatch launch_preprocess.sh
```

#### 2.2 Main Data Assimilation

Save as `launch_main_da.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=DA_MAIN
#SBATCH --output=DA_MAIN_%j.out
#SBATCH --error=DA_MAIN_%j.err
#SBATCH --account=chipilskigroup_q
#SBATCH --partition=chipilskigroup_q
#SBATCH -t 04:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12G

echo "--- Main DA Job Started ---"

module purge
module load gnu/4.8.5
module load anaconda/3.11.5
eval "$(conda shell.bash hook)"
conda activate speedy_da_env
export LD_LIBRARY_PATH=$HOME/speedy_libs/lib:$LD_LIBRARY_PATH

cd /gpfs/home/<your_username>/AMLCS/amlcs/

srun python amlcs_da.py amlcs_da_t21_LEnKF_s2r1.csv

echo "--- Main DA Job Finished ---"
```

Submit:

```bash
sbatch launch_main_da.sh
```

#### 2.3 Post-Processing and Plotting

Create a plotting config file `plot_config.csv` in the AMLCS root:

```
exp_path,resolution,variable,level
t21_80_0.05_30_LEnKF_1_5_108,t21,,
```

Save Slurm script as `launch_plotting.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=DA_PLOT
#SBATCH --output=DA_PLOT_%j.out
#SBATCH --error=DA_PLOT_%j.err
#SBATCH --account=chipilskigroup_q
#SBATCH --partition=chipilskigroup_q
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G

echo "--- Plotting Job Started ---"

module purge
module load anaconda/3.11.5
eval "$(conda shell.bash hook)"
conda activate speedy_da_env

cd /gpfs/home/your_username/AMLCS/

srun python to_run/error_plots.py plot_config.csv

echo "--- Plotting Job Finished ---"
```

Submit:

```bash
sbatch launch_plotting.sh
```

---

### Notes and tips

- Ensure `LD_LIBRARY_PATH` points to any required custom libraries (e.g., compiled speed y libs).  
- Verify job output and error files (DA_PREP_*.out, DA_MAIN_*.out, DA_PLOT_*.out) for progress and errors.  
- Adjust time, memory, and partition options in Slurm headers according to cluster policies and job needs.  
- Replace `/gpfs/home/your_username/` with your actual home directory in all scripts.

This completes the steps required to run the T21 data-assimilation experiment using AMLCS on an HPC system.