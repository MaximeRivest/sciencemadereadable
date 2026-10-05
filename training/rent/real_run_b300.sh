# on the B300: wait for the smoke test to pass, stop launch.sh before its 3-pass run, start 1 pass instead
cd ~/srl
until grep -q "smoke test passed" train.log; do sleep 15; grep -q "Traceback" <(tail -c 3000 train.log) && { echo "smoke failed"; exit 1; }; done
tmux kill-session -t train; sleep 10; pkill -f train_multi.py; sleep 10
echo "== $(date -Is) real run: 1 pass, checkpoint every 60 min" >> train.log
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 .venv/bin/python train_multi.py \
  --model Qwen/Qwen3.5-9B --name qwen35-9b-full-glossary --data data/examples-glossary.parquet \
  --epochs 1 --lr 1e-5 --save-every 4 --checkpoint-minutes 60 >> train.log 2>&1
echo "== $(date -Is) exit $?" >> train.log
