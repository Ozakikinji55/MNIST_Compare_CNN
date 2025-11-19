#!/usr/bin/env bash
# start.sh

#set -euo pipefail

DATA_DIR="./data"
OUT_DIR="./outputs/baseline"
CKPT="$OUT_DIR/model.pt"
PRED_PUBLIC="$OUT_DIR/pred_public.csv"
PRED_PRIVATE="$OUT_DIR/pred_private.csv"  
PUBLIC_LABELS="$DATA_DIR/test_public_labels.csv"
PUBLIC_TEST_FILE="test_public.npz"
PRIVATE_TEST_FILE="test_private.npz"  

mkdir -p "$OUT_DIR"

echo "1) Training baseline model..."
python3 -m scripts.train_baseline --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" 

echo "2) Generating predictions on the public test set..."
python3 -m scripts.baseline_inference --data_dir "$DATA_DIR" --ckpt "$CKPT" --out "$PRED_PUBLIC"

echo "3) Evaluating predictions on the public test set locally..."
python3 -m scripts.eval_public --data_dir "$DATA_DIR" --pred "$PRED_PUBLIC" --labels "$PUBLIC_LABELS"

echo "4) Generating predictions on the private test set..."
python3 -m scripts.baseline_inference --data_dir "$DATA_DIR" --ckpt "$CKPT" --out "$PRED_PRIVATE" --private "True"

echo "5) Checking submission file format for public test set..."
python3 -m scripts.check_submission --data_dir "$DATA_DIR" --pred "$PRED_PUBLIC" --test_file "$PUBLIC_TEST_FILE"

echo "6) Checking submission file format for private test set..."
python3 -m scripts.check_submission --data_dir "$DATA_DIR" --pred "$PRED_PRIVATE" --test_file "$PRIVATE_TEST_FILE"

echo "Done."