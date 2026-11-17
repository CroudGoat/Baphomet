#!/bin/bash
# kill+resume chain: kill in round 1 (init), right after the init state, mid-iteration, right after the round-1 IRT refit, at a later
# iteration state, mid-validation, at evo-done; then finish. Compare to an uninterrupted run (mk.sh /tmp/dfs_A2 "$X").
X='"hard_h": 99, "max_iter": 6'
cd /workspace/work/evogm-dfs/mock
./mk.sh /tmp/dfs_H1 "$X, \"kill_at\": \"eval:6\""
./mk.sh /tmp/dfs_H2 "$X, \"kill_at\": \"state:1\"" /tmp/dfs_H1
./mk.sh /tmp/dfs_H3 "$X, \"kill_at\": \"eval:30\"" /tmp/dfs_H2
./mk.sh /tmp/dfs_H4 "$X, \"kill_at\": \"refit:1\"" /tmp/dfs_H3
./mk.sh /tmp/dfs_H5 "$X, \"kill_at\": \"state:6\"" /tmp/dfs_H4
./mk.sh /tmp/dfs_H6 "$X, \"kill_at\": \"val:2\"" /tmp/dfs_H5
./mk.sh /tmp/dfs_H7 "$X" /tmp/dfs_H6
echo CHAIN DONE
