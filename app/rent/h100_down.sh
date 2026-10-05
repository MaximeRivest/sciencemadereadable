#!/usr/bin/env bash
# Stop the rented H100: its worker, the tunnel, then the machine and its disk (billing stops).
# Afterwards nothing rewrites papers until a GPU is back: the site says "GPU offline", and saved
# rewrites and the Library still work. To use the home GPUs again: app/rent/home_gpus.sh on
export PATH=$HOME/.nebius/bin:$PATH
P=project-e00fbrcapr00yr015zfmw3; NAME=smr-h100
tmux kill-session -t smr-worker-h100 2>/dev/null; tmux kill-session -t h100-tunnel 2>/dev/null
ID=$(nebius compute instance get-by-name --parent-id $P --name $NAME --format json 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin)['metadata']['id'])" 2>/dev/null)
[ -n "$ID" ] && nebius compute instance delete --id $ID && echo "deleted $NAME"
sleep 5
echo "left in the project: $(nebius compute instance list --parent-id $P --format json | python3 -c "import sys,json;print([i['metadata']['name'] for i in json.load(sys.stdin).get('items',[])])") disks: $(nebius compute disk list --parent-id $P --format json | python3 -c "import sys,json;print([i['metadata']['name'] for i in json.load(sys.stdin).get('items',[])])")"
