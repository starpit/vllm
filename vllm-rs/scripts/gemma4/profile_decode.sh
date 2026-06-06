#!/bin/bash
# Serialized per-kernel decode profile (one decode forward, num_tokens=1).
# Output: [dump-timing] table in /tmp/gemma4_prof.log
set -e
cd "$(dirname "$0")/../.."
pkill -x vllm 2>/dev/null || true
sleep 2
rm -rf /tmp/g4prof && mkdir -p /tmp/g4prof
FERRITE_DUMP_DIR=/tmp/g4prof FERRITE_DUMP_NUM_TOKENS=1 \
FERRITE_DUMP_KERNELS=all FERRITE_DUMP_TIMING_ONLY=1 \
  target/release/vllm serve mlx-community/gemma-4-12B-it-4bit --port 8399 \
  > /tmp/gemma4_prof.log 2>&1 &
SPID=$!
until curl -sf http://127.0.0.1:8399/health >/dev/null 2>&1; do
  sleep 2
  kill -0 $SPID 2>/dev/null || { echo "SERVER DIED"; tail -5 /tmp/gemma4_prof.log; exit 1; }
done
# 3 decode steps -> 3 serialized replays (table printed per replay)
curl -s http://127.0.0.1:8399/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/gemma-4-12B-it-4bit","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":3,"temperature":0}' >/dev/null
sleep 2
kill $SPID 2>/dev/null
grep -A 30 "dump-timing" /tmp/gemma4_prof.log | tail -34
