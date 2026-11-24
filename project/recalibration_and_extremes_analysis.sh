#!/usr/bin/env bash
# Recalibrates PEBSI on the five glaciers, then reruns the extremes experiments on the new parameters.
#   1  calibration  AD_optimize.py, og_kappa held at Output/AD_optimize/og_kappa_kp2.yaml, fitting kp, wind_factor
#                   and precgrad to albedo, SAR wet extent, Hugonnet and winter index balance; summer index balance
#                   is tracked, not fitted (Output/AD_optimize/CALIBRATION_NOTES.md). Stops if wind_factor moved
#                   by 1 or more (step 3 of the notes: re-pick og_kappa).
#   2  controlled   base / no_fires / no_heatwaves / no_either, 2000-2025, MERRA-2 deposition,
#                   then per_glacier.nc, per_point_annual.nc and per_point_daily.nc for each
#   3  synthetic    heatwave and wildfire grids in 2021 on all five glaciers (spin-ups, every run, heatwave splits)
#   4  analysis     figures: Output/AD_optimize, Output/controlled/figs_current, Figs/synthetic_current
# Everything it writes is named *_current (AD_optimize_result_current.yaml, controlled/base_current_0,
# heatwave_grid_current, ...). A full rerun first renames all of those *_iter<n>, n one more than the highest
# already there.
#
# From PEBSI/ (progress goes to Output/logs/recalibration_and_extremes_analysis_current.log):
#   setsid nohup project/recalibration_and_extremes_analysis.sh &> /dev/null < /dev/null &
# To resume at a stage, pass its number (e.g. 3), after deleting that stage's partial *_current outputs;
# nothing is renamed on a resume.
set -euo pipefail
shopt -s nullglob

STAGE=${1:-1}
PY=$HOME/miniconda3/envs/gpu-env/bin/python
OUT=/home/claire/Research/Output
FIGS=/home/claire/Research/Figs
INIT=$OUT/AD_init_kappa_kp2.yaml   # starting point: the previous fit and its fixed og_kappa
RESULT=$OUT/AD_optimize/AD_optimize_result_current.yaml   # what the run_*.py drivers read
N_PARALLEL=2                       # grid runs at once on the GPU (~10 GB host RAM each)
LOG=$OUT/logs/recalibration_and_extremes_analysis_current.log
export MPLBACKEND=Agg PY OUT

cd "$(dirname "$0")/.."   # PEBSI/

# a full rerun renames the last run's outputs *_current -> *_iter<n>, n one more than the highest already there
if (( STAGE == 1 )); then
    n=1
    for f in configs_*_iter* "$OUT"/AD_optimize/*_iter* "$OUT"/logs/*_iter* "$OUT"/controlled/*_iter* \
             "$OUT"/controlled/logs/*_iter* "$OUT"/*_iter* "$FIGS"/*_iter*; do
        if [[ $f =~ _iter([0-9]+) ]] && (( BASH_REMATCH[1] >= n )); then n=$(( BASH_REMATCH[1] + 1 )); fi
    done
    [[ ! -e $LOG ]] || mv "$LOG" "${LOG/_current/_iter$n}"
fi

exec >> "$LOG" 2>&1
log() { echo "[$(date '+%F %T')] $*"; }
trap 'log "FAILED at line $LINENO, see the log of that step"' ERR
log "starting at stage $STAGE"

if (( STAGE == 1 )); then
    for p in configs_*_AD_optimize_current.yaml "$OUT"/AD_optimize/*_current.* "$OUT"/logs/*_current.log \
             "$OUT"/controlled/*_current* "$OUT"/controlled/logs/*_current.log "$OUT"/*_grid_current \
             "$FIGS"/synthetic_current; do
        if [[ -e $p && $p != "$LOG" ]]; then
            b=$(basename "$p")
            mv "$p" "$(dirname "$p")/${b/_current/_iter$n}"
            log "  renamed $p -> ${b/_current/_iter$n}"
        fi
    done
fi

# a stage's outputs must not exist yet (PEBSI would number a second run dir _1; the grid driver refuses)
fresh() { [[ ! -e $1 ]] || { log "$1 exists from an earlier attempt; delete it to rerun this stage"; exit 1; }; }

# one synthetic run: EVENT RUN (spinup, a run number or split_<n>)
grid_job() {
    local name=$2
    [[ $2 =~ ^[0-9]+$ ]] && name=$(printf 'run_%03d' "$2")
    local fn=$OUT/$1_grid_current/logs/$name.log
    if EVENT=$1 RUN=$2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 "$PY" -u project/synthetic/run_event_grid.py > "$fn" 2>&1; then
        log "  $1 $2 done"
    else
        log "  $1 $2 FAILED, see $fn"
        return 1
    fi
}
export -f log grid_job

if (( STAGE <= 1 )); then
    log "stage 1: calibration -> $RESULT"
    "$PY" -u AD_optimize.py --hybrid --fit_precgrad --index_mb --summer_weight 0 -n 10 \
        --init_from "$INIT" --tag current > "$OUT/logs/AD_optimize_current.log" 2>&1
    "$PY" - "$INIT" "$RESULT" <<'EOF'
import sys, yaml
init, new = (yaml.safe_load(open(fn)) for fn in sys.argv[1:])
moved = []
for j, g in enumerate(new['glaciers']):
    i = init['glaciers'].index(g)
    print(f'  {g:<12}' + ''.join(f'  {p} {init[p][i]:.4g} -> {new[p][j]:.4g}'
                                 for p in ['kp', 'wind_factor', 'precgrad']), flush=True)
    if abs(new['wind_factor'][j] - init['wind_factor'][i]) >= 1:
        moved.append(g)
if moved:
    sys.exit(f"wind_factor moved by 1 or more on {', '.join(moved)}: re-pick og_kappa "
             '(CALIBRATION_NOTES.md step 3), or rerun this script with stage 2 to go on anyway')
EOF
fi

SIMS=(base no_fires no_heatwaves no_either)
if (( STAGE <= 2 )); then
    log "stage 2: controlled runs"
    for sim in "${SIMS[@]}"; do fresh "$OUT/controlled/${sim}_current_0"; done
    mkdir -p "$OUT/controlled/logs"
    for sim in "${SIMS[@]}"; do
        log "  $sim"
        SIM_TYPE=$sim DEPOSITION=MERRA2 "$PY" -u project/controlled/run_controlled_experiment.py \
            > "$OUT/controlled/logs/${sim}_current.log" 2>&1
    done
    log "  aggregating"
    "$PY" -u shop/postprocessing/aggregate_per_glacier.py "$OUT"/controlled/{base,no_fires,no_heatwaves,no_either}_current_0 \
        > "$OUT/controlled/logs/aggregate_current.log" 2>&1
fi

if (( STAGE <= 3 )); then
    log "stage 3: synthetic event grids"
    for event in heatwave wildfire; do
        fresh "$OUT/${event}_grid_current"
        mkdir -p "$OUT/${event}_grid_current/logs"
    done
    n_runs() { EVENT=$1 RUN=0 PYTHONPATH=project/synthetic "$PY" -c "from run_event_grid import $2; print(len($2()))"; }
    n_hw=$(n_runs heatwave design); n_split=$(n_runs heatwave split_design); n_wf=$(n_runs wildfire design)
    log "  $n_hw heatwave + $n_split split + $n_wf wildfire runs, $N_PARALLEL at a time"
    # every run resumes from its event's spin-up
    printf '%s\n' 'heatwave spinup' 'wildfire spinup' | xargs -P 2 -L 1 bash -c 'grid_job "$@"' _
    {
        seq 0 $((n_hw - 1)) | sed 's/^/heatwave /'
        seq 0 $((n_split - 1)) | sed 's/^/heatwave split_/'
        seq 0 $((n_wf - 1)) | sed 's/^/wildfire /'
    } | xargs -P "$N_PARALLEL" -L 1 bash -c 'grid_job "$@"' _
fi

if (( STAGE <= 4 )); then
    log "stage 4: analysis"
    CLOGS=$OUT/controlled/logs
    mkdir -p "$OUT/controlled/figs_current" "$FIGS/synthetic_current"
    log "  calibration"
    "$PY" -u plot_AD_iterations.py "$OUT/AD_optimize/AD_optimize_progress_current.yaml" \
        "$OUT/AD_optimize/AD_iteration_errors_current.png"
    (cd project && "$PY" -u plot_calibrated_mb.py) > "$CLOGS/plot_calibrated_mb_current.log" 2>&1
    log "  controlled"
    "$PY" -u project/controlled/plot_controlled_effects.py > "$CLOGS/plot_controlled_effects_current.log" 2>&1
    "$PY" -u project/controlled/plot_fire_albedo.py > "$CLOGS/plot_fire_albedo_current.log" 2>&1
    "$PY" -u project/controlled/plot_heatwave_energy.py > "$CLOGS/plot_heatwave_energy_current.log" 2>&1
    log "  synthetic"
    for event in heatwave wildfire; do
        EVENT=$event "$PY" -u project/synthetic/plot_event_grid.py > "$OUT/${event}_grid_current/logs/plot.log" 2>&1
    done
    "$PY" -u project/synthetic/plot_grid_scaling.py > "$OUT/heatwave_grid_current/logs/plot_scaling.log" 2>&1
fi

log "done"
