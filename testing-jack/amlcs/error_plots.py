import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from grid_resolution import grid_resolution
from postpro_tools import postpro_tools

import matplotlib

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams.update({"font.size": 14})
sns.set_style("darkgrid")

model_vars = ["PSG0", "PSG1", "TG0", "TG1", "TRG0", "TRG1", "UG0", "UG1", "VG0", "VG1"]

var_codes = {
    "TG0": "T_0",
    "UG0": "u_0",
    "VG0": "v_0",
    "TRG0": "Hq_0",
    "TG1": "T_1",
    "UG1": "u_1",
    "VG1": "v_1",
    "TRG1": "Hq_1",
    "PSG0": "PS_0",
    "PSG1": "PS_1",
}

pslvl = [30, 100, 200, 300, 500, 700, 850, 925]


def single_error_plotter(analysis, background, var, lvl, ppt, plots_path):

    lvl_str = str(lvl)
    eps = 1e-12

    ana_raw = analysis[lvl_str].to_numpy(dtype=float)
    bkg_raw = background[lvl_str].to_numpy(dtype=float)
    noda_raw = np.asarray(ppt.noda[var][lvl, :], dtype=float)

    # equalize lengths first
    L = min(len(ana_raw), len(bkg_raw), len(noda_raw))
    ana_raw, bkg_raw, noda_raw = ana_raw[:L], bkg_raw[:L], noda_raw[:L]
    if L == 0:
        return

    # ---- choose a common anchor ----
    # Option A (recommended): anchor on NoDA's first value
    anchor = noda_raw[:1]
    # Option B: anchor on Background's first value
    # anchor = bkg_raw[:1]

    # prepend the SAME anchor to all
    ana = np.concatenate([anchor, ana_raw])
    bkg = np.concatenate([anchor, bkg_raw])
    noda = np.concatenate([anchor, noda_raw])

    # log with epsilon
    ana_log = np.log(ana + eps)
    bkg_log = np.log(bkg + eps)
    noda_log = np.log(noda + eps)

    # first TRUE cycle is index 1 (index 0 is the visual anchor)
    if abs(bkg_log[1] - noda_log[1]) > 1e-6:
        print(f"[plot][warn] {var} lvl={lvl}: Background vs NoDA differ at M=1 by {abs(bkg_log[1]-noda_log[1]):.3e} (log-scale).")

    xs = np.arange(0, L + 1)  # 0 = anchor, 1..L = cycles

    plt.figure(figsize=(9, 4))
    plt.title(rf"$\mathrm{{{var_codes[var]}}} \ at \ {pslvl[lvl]} \ \mathrm{{mb}}$")
    plt.plot(xs, ana_log, color="r", label="Analysis")
    plt.plot(xs, bkg_log, color="b", label="Background")
    plt.plot(xs, noda_log, color="k", label="NODA")
    plt.ylabel(r"$\log(\mathcal{l}_2)$")
    plt.xlabel(r"$\mathrm{Assimilation\ Step}$")
    plt.legend(loc="best", prop={"size": 14})
    plt.tight_layout()
    plt.savefig(plots_path / f"single_error_{var}_{lvl}.png", bbox_inches="tight")
    plt.close()



def main_general_plotter(df_params):
    root_path = Path.cwd()
    exp_pth = root_path.parent / "runs" 

    for _, row in df_params.iterrows():
        method_path = exp_pth / row["exp_path"]

        variables = row["variable"]
        levels = row["level"]
        grid_res = row["resolution"]
        M = int(row["M"])
        # ADDED: Read the new directory name parameter from the CSV
        plot_dir_name = row["plot_dir_name"]
        if pd.isna(variables):
            variables = model_vars
        else:
            variables = variables.strip().split(",")

        if pd.isna(levels):
            levels = range(8)
        else:
            levels = levels.strip().split(",")
            levels = [int(v) for v in levels]

        # ADDED: Provide a default name if the parameter is not in the CSV
        # First, define the base path to the "errors" directory
        plots_path = method_path / "plots" / "errors"

        # If a specific subdirectory name is given in the CSV, append it
        if not pd.isna(plot_dir_name):
            plots_path = plots_path / plot_dir_name

        # UPDATED: Construct the plots_path using the new directory name
        

        gs = grid_resolution(grid_res)
        ppt = postpro_tools(grid_res, gs, method_path, M)
        ppt.compute_NODA()

        Path(plots_path).mkdir(parents=True, exist_ok=True)

        for var in variables:
            analysis_path = method_path / "results" / f"{var}_ana.csv"
            bckg_path = method_path / "results" / f"{var}_bck.csv"

            if not analysis_path.exists() or not bckg_path.exists():
                print(f"Warning: Result files for {var} not found. Skipping.")
                continue

            analysis = pd.read_csv(analysis_path)
            bckg = pd.read_csv(bckg_path)

            for lvl in levels:
                if ("PSG" in var) and lvl > 0:
                    break
                if ("TRG" in var) and lvl < 2:
                    continue
                single_error_plotter(analysis, bckg, var, lvl, ppt, plots_path)
            print(f"* ENDJ - Plot {var} Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates error plots for a defined set of parameters.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "file", help="The name of the CSV file containing the configuration"
    )
    args = parser.parse_args()

    input_file = args.file
    print("* STARTJ - Reading input file {0}".format(input_file))
    df_params = pd.read_csv(input_file)
    main_general_plotter(df_params)

