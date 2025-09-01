# Building and Running the Legacy SPEEDY Model

## 1. Introduction

This document provides a complete guide for compiling and running this legacy version of the SPEEDY atmospheric model on a modern Linux-based High-Performance Computing (HPC) environment, like Florida State's RCC.

The model's source code is a hybrid of older Fortran 77 (F77) format and newer Fortran 90 (F90) features. This requires a specific "period-correct" software environment, including an older compiler and specific versions of the NetCDF libraries. This guide will walk you through building this environment from source in your home directory.

---

## 2. Overview of Prerequisites

To successfully compile the model, you will need to build specific older versions of the following libraries from source: zlib, HDF5, NetCDF-C, and NetCDF-Fortran.

The following sections will guide you through downloading and compiling each of these required components in the correct order and location.

---

## 3. Environment Setup on the HPC

Before you can build anything, you must load an older, more permissive compiler.

1.  Log in to the HPC.
2.  Search for an available GNU compiler in the 4.x series:
    ```bash
    module avail 2>&1 | grep -i gnu
    ```
3.  Load the appropriate module. A version like `4.8.5` is a good choice.
    ```bash
    module load gnu/4.8.5
    ```
4.  Verify that the older compiler is active:
    ```bash
    gfortran --version
    ```

---

## 4. Compiling Dependencies from Source

We will create a directory in your home folder to build the software (`~/software_builds`) and a separate directory where the final libraries will be installed (`~/speedy_libs`).

#### Step 4.1: Create Directories
```bash
cd ~
mkdir software_builds
mkdir speedy_libs
cd software_builds
```

#### Step 4.2: Download & Build zlib
```bash
wget https://www.zlib.net/fossils/zlib-1.2.11.tar.gz 
tar -xvf zlib-1.2.11.tar.gz
cd zlib-1.2.11
./configure --prefix=$HOME/speedy_libs
make
make install
cd ..
```

#### Step 4.3: Download & Build HDF5
```bash
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.12/src/hdf5-1.8.12.tar.gz
tar -xvf hdf5-1.8.12.tar.gz
cd hdf5-1.8.12
./configure --prefix=$HOME/speedy_libs --with-zlib=$HOME/speedy_libs --enable-fortran
make
make install
cd ..
```

#### Step 4.4: Download & Build NetCDF-C
```bash
wget https://github.com/Unidata/netcdf-c/archive/refs/tags/v4.3.2.tar.gz
tar -xvf v4.3.2.tar.gz
cd netcdf-c-4.3.2
export CPPFLAGS="-I$HOME/speedy_libs/include"
export LDFLAGS="-L$HOME/speedy_libs/lib"
export LD_LIBRARY_PATH=$HOME/speedy_libs/lib:$LD_LIBRARY_PATH
./configure --prefix=$HOME/speedy_libs --disable-dap
make
# make check
make install
cd ..
```

#### Step 4.5: Download & Build NetCDF-Fortran
```bash
wget https://github.com/Unidata/netcdf-fortran/archive/refs/tags/netcdf-fortran-4.2.tar.gz 
tar -xvf netcdf-fortran-4.2.tar.gz
cd netcdf-fortran-netcdf-fortran-4.2
autoreconf -i # Makes the configure script executable
# The environment variables should still be set from the last step.
./configure --prefix=$HOME/speedy_libs

### IMPORTANT
# 1. Open the Makefile in a text editor.
nano Makefile

# 2. Find the line that starts with 'SUBDIRS ='. (NOT DISTSUBDIRS) Use Ctrl+W to search.
# 3. Delete the word 'man4' from that line.
#    FROM: SUBDIRS = fortran f90 nf_test man4 examples
#    TO:   SUBDIRS = fortran f90 nf_test examples
# 4. Save and exit by pressing Ctrl+X, then Y, then Enter.
### 

make
# make check
make install
cd ..
```
All required libraries are now built and installed in `~/speedy_libs`.

---

## 5. Building the SPEEDY Model

#### Step 5.1: Clone the Model Repository and Navigate

First, clone the repository containing the SPEEDY model source code and navigate to the correct directory where the `makefile` is located.

```bash
cd ~ # Or another directory of your choice
git clone https://github.com/jjs21b/SPEEDY-f77.git
cd SPEEDY-f77/models/speedy/t21
```
#### Step 5.2: Configure the `makefile`
This repository includes a `makefile` that is pre-configured for this build process. You only need to edit the variables that point to your custom library installation.

1.  Open the `makefile` in a text editor.
2.  Find the following configuration block near the top.
3.  Replace the placeholder path `/gpfs/home/your_username` with the actual path to your home directory.

    ```makefile
    # --- START OF USER CONFIGURATION ---

    # TODO: Replace the path below with the full path to your home directory.
    NETCDF_INSTALL_PATH = /gpfs/home/your_username/speedy_libs
    
    # These variables use the path defined above. No changes are needed here.
    NETCDF_FLAGS = -L$(NETCDF_INSTALL_PATH)/lib -lnetcdff
    NETCDF_INCLUDE = -I$(NETCDF_INSTALL_PATH)/include
    
    # --- END OF USER CONFIGURATION ---
    ```
    For example, if your username is `jjs21b`, the first line should be:
    `NETCDF_INSTALL_PATH = /gpfs/home/jjs21b/speedy_libs`

---
**Note**: If you started a new terminal session after installing the required libraries, rerun the following commands before compilation:

```bash
module load gnu/4.8.5
export LD_LIBRARY_PATH=$HOME/speedy_libs/lib:$LD_LIBRARY_PATH
```

#### Step 5.3: Compile the Model
With the `makefile` configured, navigate to the model's source directory (`.../t21/`) and run the compile script.
```bash
bash compile.sh
```
This should create the executable file `imp.exe`.



## 6. Running the Model

Ensure you have the necessary input files in your directory, then run the executable:
```bash
./imp.exe
```
The model should now start and run correctly.

