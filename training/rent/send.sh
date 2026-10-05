#!/usr/bin/env bash
# From lambda: copy what the rented 8-GPU machine needs.   send.sh user@host [ssh port]
set -e
cd "$(dirname "$0")/.."
H=$1; P=${2:-22}
ssh -p $P $H "mkdir -p ~/srl/data ~/srl/runs"
rsync -avP -e "ssh -p $P" train.py train_multi.py rent/launch.sh $H:~/srl/
rsync -avP -e "ssh -p $P" data/examples-glossary.parquet data/tokens-examples-glossary-f796c991319e-*.pt $H:~/srl/data/
echo "now: ssh -p $P $H 'cd ~/srl && bash launch.sh'"
