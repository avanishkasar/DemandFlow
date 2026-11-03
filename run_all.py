"""Regenerates every number, table and figure of the revised manuscript (RINENG-D-26-22476, revision 1).
Usage:  python run_all.py            (full run, several hours on 2 CPU cores)
        python run_all.py --quick    (main experiment, audit and figures only, ~15 minutes)
The leakage audit additionally needs the ORIGINAL repository files in ../data/processed (see leakage_audit.py)."""
import subprocess, sys
steps = [["select_sarima.py"], ["run_main.py"], ["leakage_audit.py"]]
if "--quick" not in sys.argv:
    steps += [["run_replications.py", "beta"], ["run_replications.py", "sens"], ["decision_modules.py"]]
steps += [["summarize.py"], ["make_figures.py"]]
for s in steps:
    print(">>", " ".join(s), flush=True)
    subprocess.run([sys.executable] + s, check=True)
