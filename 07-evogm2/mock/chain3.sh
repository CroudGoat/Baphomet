#!/bin/bash
# kill+resume chain (2026-11-16, with the round-boundary IRT refit): kill in round 1, mid-round-1, at the "iterations finished"
# state, right after the refit, mid-round-2, mid-validation, then finish. Compare to an uninterrupted run with the same X.
X='"hard_h": 99, "max_round": 2'
./mk.sh /tmp/mk_H1 "$X, \"kill_at\": \"r1:4\""
./mk.sh /tmp/mk_H2 "$X, \"kill_at\": \"state:1,3\"" /tmp/mk_H1
./mk.sh /tmp/mk_H3 "$X, \"kill_at\": \"state:1,4\"" /tmp/mk_H2
./mk.sh /tmp/mk_H4 "$X, \"kill_at\": \"refit:1\"" /tmp/mk_H3
./mk.sh /tmp/mk_H5 "$X, \"kill_at\": \"state:2,2\"" /tmp/mk_H4
./mk.sh /tmp/mk_H6 "$X, \"kill_at\": \"val:3\"" /tmp/mk_H5
./mk.sh /tmp/mk_H7 "$X" /tmp/mk_H6
echo CHAIN DONE
