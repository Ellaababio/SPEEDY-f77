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

The experiment is executed as three separate Slurm jobs. Wait for each to finish before submitting the next. Ensure that you are in your repository's root directory when submitting these jobs

#### 2.1 Pre-Processing (generate initial ensemble and reference)

Submit:

```bash
sbatch launch_preprocess.sh
```
Before moving onto the next step, verify that there are no fatal errors in the DA_PREP_%j.err file and that the final lines of the generate DA_PREP_%j.out file state the following:

```txt
* ENDJ - Finishing creating the free_run trajectory for M = 30
* ENDJ - All ensemble members have been collected
--- Pre-processing Job Finished ---
```


#### 2.2 Main Data Assimilation


Submit:

```bash
sbatch launch_main_da.sh
```
Before moving onto the next step, verify that there are no fatal errors in the DA_MAIN_11272653_%j.err and that the final lines of the DA_MAIN_11272653_%j.out file state the following:

```txt
* ENDJ - Performing forecast ensemble member 79
--- Main DA Job Finished ---
```
You should be able to view your results as csv files within the `/gpfs/home/jjs21b/AMLCS/runs/t21_80_0.05_30_LEnKF_1_5_108/results` directory
#### 2.3 Post-Processing and Plotting


Submit:

```bash
sbatch launch_plotting.sh
```

Once this job concludes, you should be able to view the error plots within the `runs/t21_80_0.05_30_LEnKF_1_5_108/plots` directory.

### Notes and tips

- Ensure `LD_LIBRARY_PATH` points to any required custom libraries (e.g., compiled speed y libs).  
- Verify job output and error files (DA_PREP_*.out, DA_MAIN_*.out, DA_PLOT_*.out) for progress and errors.  
- Adjust time, memory, and partition options in Slurm headers according to cluster policies and job needs.  

This completes the steps required to run the T21 data-assimilation experiment using AMLCS on an HPC system.