#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/project"
DATA="/home/ext-z/data/vision/iNat2021_extracted"
LOGDIR="$ROOT/training_logs"
mkdir -p "$LOGDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_try() {
  local phase="$1"; shift
  local img="$1"; shift
  local epochs="$1"; shift
  local lr="$1"; shift
  local wdec="$1"; shift
  local mixup="$1"; shift
  local lsmooth="$1"; shift
  local freeze_flag="$1"; shift   # "yes" / "no"
  local -a batches=("$@")

  local ts
  ts="$(date +%F_%H-%M)"
  local log="$LOGDIR/${phase}_${ts}.log"

  echo "[INFO] Phase=${phase} img=${img} epochs=${epochs} lr=${lr} wd=${wdec} mixup=${mixup} ls=${lsmooth} freeze=${freeze_flag}" | tee -a "$log"

  for bs in "${batches[@]}"; do
    echo "[TRY ] ${phase}: batch_size=${bs}" | tee -a "$log"
    if [[ "$freeze_flag" == "yes" ]]; then
      CMD=(python training/train.py --dataset_root "$DATA" --img_size "$img" --batch_size "$bs" --epochs "$epochs" --lr "$lr" --weight_decay "$wdec" --label_smoothing "$lsmooth" --mixup "$mixup" --amp --freeze_backbone --num_workers 8)
    else
      CMD=(python training/train.py --dataset_root "$DATA" --img_size "$img" --batch_size "$bs" --epochs "$epochs" --lr "$lr" --weight_decay "$wdec" --label_smoothing "$lsmooth" --mixup "$mixup" --amp --num_workers 8)
    fi

    if "${CMD[@]}" 2>&1 | tee -a "$log"; then
      echo "[OK  ] ${phase}: succeeded with batch_size=${bs}" | tee -a "$log"
      grep -E "val@1|val_acc|Val Acc|val_loss|Train Loss|Epoch" -n "$log" | tail -n 20 || true
      return 0
    else
      if grep -qiE "CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|no kernel image|resource busy" "$log"; then
        echo "[WARN] ${phase}: OOM with batch_size=${bs}, trying smaller..." | tee -a "$log"
        continue
      else
        echo "[FAIL] ${phase}: failed for non-OOM reason. See log: $log" | tee -a "$log"
        return 1
      fi
    fi
  done

  echo "[FAIL] ${phase}: exhausted all batch sizes" | tee -a "$log"
  return 1
}

main() {
  cd "$ROOT"
  source venv/bin/activate

  echo "[PHASE1] Warmup @160 with frozen backbone"
  run_try "phase1_160_frozen" 160 12 3e-4 1e-4 0.2 0.1 "yes" 8 6 4 2

  echo "[PHASE2] Finetune @160 (unfrozen)"
  run_try "phase2_160_finetune" 160 20 2e-4 1e-4 0.2 0.1 "no" 6 4 2

  echo "[PHASE3] OPTIONAL finetune @224 (unfrozen)"
  run_try "phase3_224_finetune" 224 10 1.5e-4 1e-4 0.1 0.05 "no" 2 1 || echo "[INFO] Phase 3 skipped/failed — ממשיכים הלאה"

  echo "[DONE] Checkpoints at training/checkpoints/, logs at $LOGDIR, plots in plots/"
}
main

