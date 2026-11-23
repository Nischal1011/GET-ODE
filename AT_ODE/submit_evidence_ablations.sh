#!/usr/bin/env bash
set -euo pipefail

PBS_SCRIPT="${PBS_SCRIPT:-submit_polaris_obs_job.pbs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-experiments/evidence_transport}"
NITERS="${NITERS:-100}"

submit_job() {
  local dataset="$1"
  local mode="$2"
  local seed="$3"
  local source="$4"
  local ablation="$5"
  local alias_name="$6"
  local extra_args

  extra_args="--edge-posterior-source ${source} --edge_prior_mode graph --edge-prior-strength 1.0 --age-kernel uniform --alias ${alias_name} --save ${CHECKPOINT_DIR}"
  if [[ "$source" == "evidence" ]]; then
    extra_args+=" --evidence-ablation ${ablation}"
  fi

  qsub -v "DATASET=${dataset},MODE=${mode},OBS_RATIO=0.6,NITERS=${NITERS},SEED=${seed},EXTRA_ARGS=${extra_args}" "$PBS_SCRIPT"
}

for seed in 1991 42; do
  submit_job dataset1 extrap "$seed" none full "lgode_spring_06_extrap_s${seed}"
  submit_job dataset1 extrap "$seed" attention full "attention_spring_06_extrap_s${seed}"
  submit_job dataset1 extrap "$seed" evidence instantaneous "evidence_instant_spring_06_extrap_s${seed}"
  submit_job dataset1 extrap "$seed" evidence full "evidence_transport_spring_06_extrap_s${seed}"
  submit_job dataset1 extrap "$seed" evidence shuffle-times "evidence_shuffle_times_spring_06_extrap_s${seed}"
  submit_job dataset1 extrap "$seed" evidence shuffle-values "evidence_shuffle_values_spring_06_extrap_s${seed}"
  submit_job dataset1 extrap "$seed" evidence prior-only "prior_only_spring_06_extrap_s${seed}"

  submit_job dataset2 interp "$seed" evidence full "evidence_transport_charged_06_interp_s${seed}"
  submit_job dataset2 extrap "$seed" evidence full "evidence_transport_charged_06_extrap_s${seed}"
done
