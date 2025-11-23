import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from pathlib import Path
import os

# Settings
FREE_RUN_DIR = "/gpfs/home/jjs21b/AMLCS/ENSF_gaussian_check/t21_50_0.05_5/free_run"
OUTPUT_DIR = "linearity_plots"
CYCLE = 0

# Variables to check: (Name, Level)
# Note: Using "1" suffix as seen in previous file checks (UG1, VG1, etc.)
VARIABLES = [
    ("UG1", 7),   # Zonal Wind at Level 7
    ("VG1", 7),   # Meridional Wind at Level 7
    ("TG1", 7),   # Temperature at Level 7
    ("TRG1", 7),  # Specific Humidity at Level 7 (4D variable)
    ("PSG1", 0)   # Surface Pressure (2D)
]

def check_linearity():
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    print(f"Saving plots to {Path(OUTPUT_DIR).absolute()}")
    
    nc_path = Path(FREE_RUN_DIR) / f"free_run_{CYCLE}.nc"
    if not nc_path.exists():
        print(f"Error: File not found: {nc_path}")
        return

    with Dataset(nc_path, 'r') as nc:
        print(f"Opened {nc_path}")
        print(f"Available variables: {list(nc.variables.keys())}")
        
        for var_name, level in VARIABLES:
            print(f"\nProcessing {var_name} at Level {level}...")
            
            if var_name not in nc.variables:
                # Try without suffix if not found
                base_name = var_name.rstrip("0123456789")
                if base_name in nc.variables:
                    print(f"  {var_name} not found, using {base_name}")
                    var_name = base_name
                else:
                    print(f"  Warning: {var_name} not found. Skipping.")
                    continue
            
            # Read Data
            var_data = nc.variables[var_name]
            data = None
            
            # Handle dimensions
            shape = var_data.shape
            print(f"  Shape: {shape}")
            
            if len(shape) == 4: # (tracer, lev, lat, lon)
                 if level < shape[1]:
                    data = var_data[0, level, :, :].flatten()
                 else:
                    print(f"  Level {level} out of bounds.")
                    continue
            elif len(shape) == 3: # (lev, lat, lon) or (time, lat, lon)?
                # Assume (lev, lat, lon) for 3D vars
                if level < shape[0]:
                    data = var_data[level, :, :].flatten()
                else:
                    # Could be (time, lat, lon) for 2D var
                    data = var_data[0, :, :].flatten()
            elif len(shape) == 2: # (lat, lon)
                data = var_data[:, :].flatten()
            else:
                data = var_data[:].flatten()
            
            if data is None:
                print("  Could not extract data.")
                continue

            # Calculate Statistics
            mean_val = np.mean(data)
            std_val = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            print(f"  Stats: Mean={mean_val:.2f}, Std={std_val:.2f}, Range=[{min_val:.2f}, {max_val:.2f}]")

            # Define Operators
            # Physical
            # Extend range slightly for plotting context
            x_phys = np.linspace(min_val - abs(std_val)*2, max_val + abs(std_val)*2, 1000)
            y_phys = np.arctan(x_phys)
            
            # Gradient Calculation: d(atan(x))/dx = 1 / (1 + x^2)
            grads_phys = 1.0 / (1.0 + data**2)
            mean_grad_phys = np.mean(grads_phys)

            # Normalized
            x_norm = np.linspace(-4, 4, 1000)
            y_norm = np.arctan(x_norm)
            data_norm = np.random.normal(0, 1, 10000)
            grads_norm = 1.0 / (1.0 + data_norm**2)
            mean_grad_norm = np.mean(grads_norm)

            # Plotting
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f"Linearity Check: {var_name} (Level {level})", fontsize=16)
            
            # --- Plot 1: Distributions and Operator (Physical) ---
            ax1 = axs[0, 0]
            ax1.hist(data, bins=50, density=True, alpha=0.5, color='blue', label='State Distribution')
            ax1_twin = ax1.twinx()
            ax1_twin.plot(x_phys, y_phys, 'r-', linewidth=2, label='arctan(x)')
            ax1_twin.set_ylabel('Observation Value', color='red')
            ax1_twin.tick_params(axis='y', labelcolor='red')
            ax1.set_title(f"Scenario 1: Nonlinear + No Norm\n(Input = Physical)")
            ax1.set_xlabel(f"{var_name} Value")
            
            # --- Plot 2: Gradients (Physical) ---
            ax2 = axs[1, 0]
            ax2.hist(grads_phys, bins=50, color='purple', alpha=0.7)
            ax2.set_title("Gradient Distribution (Physical)\nd(Obs)/d(State) = 1/(1+x²)")
            ax2.set_xlabel("Gradient Magnitude")
            ax2.set_yscale('log') 
            ax2.text(0.95, 0.95, f"Mean Gradient:\n{mean_grad_phys:.2e}", 
                     transform=ax2.transAxes, horizontalalignment='right', verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # --- Plot 3: Distributions and Operator (Normalized) ---
            ax3 = axs[0, 1]
            ax3.hist(data_norm, bins=50, density=True, alpha=0.5, color='green', label='Normalized State')
            ax3_twin = ax3.twinx()
            ax3_twin.plot(x_norm, y_norm, 'r-', linewidth=2, label='arctan(x)')
            ax3_twin.set_ylabel('Observation Value', color='red')
            ax3_twin.tick_params(axis='y', labelcolor='red')
            ax3.set_title("Scenario 2: Nonlinear + Norm\n(Input = Standardized N(0,1))")
            ax3.set_xlabel("Standardized Value")

            # --- Plot 4: Gradients (Normalized) ---
            ax4 = axs[1, 1]
            grads_norm = 1.0 / (1.0 + data_norm**2)
            ax4.hist(grads_norm, bins=50, color='orange', alpha=0.7)
            ax4.set_title("Gradient Distribution (Normalized)\nd(Obs)/d(State) = 1/(1+x²)")
            ax4.set_xlabel("Gradient Magnitude")
            mean_grad_norm = np.mean(grads_norm)
            ax4.text(0.05, 0.95, f"Mean Gradient:\n{mean_grad_norm:.4f}", 
                     transform=ax4.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            outfile = Path(OUTPUT_DIR) / f"linearity_{var_name}_lev{level}.png"
            plt.tight_layout()
            plt.savefig(outfile)
            plt.close()
            print(f"  Saved plot to {outfile}")
            
            # Interpretation
            if mean_grad_phys < 1e-3:
                print(f"  -> RESULT: VANISHING GRADIENT (Mean={mean_grad_phys:.2e}). Normalization REQUIRED.")
            else:
                print(f"  -> RESULT: Gradient might be okay (Mean={mean_grad_phys:.2e}).")

if __name__ == "__main__":
    check_linearity()
