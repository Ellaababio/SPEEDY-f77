sbatch --account=chipilskigroup_q --partition=chipilskigroup_q --time=12:00:00 --mem=12G --nodes=1 --ntasks-per-node=1 --wrap="./run_py.sh amlcs_da.py ensf_runner_wdg_wsg.csv"
