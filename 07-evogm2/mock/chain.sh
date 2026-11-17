#!/bin/bash
# kill+resume chain: kill in round 1, mid-evolution, mid-validation, then finish
X='"hard_h": 99, "max_round": 2'
./mk.sh /tmp/mk_B1 "$X, \"kill_at\": \"r1:5\""
./mk.sh /tmp/mk_B2 "$X, \"kill_at\": \"state:1,3\"" /tmp/mk_B1
./mk.sh /tmp/mk_B3 "$X, \"kill_at\": \"state:2,2\"" /tmp/mk_B2
./mk.sh /tmp/mk_B4 "$X, \"kill_at\": \"val:3\"" /tmp/mk_B3
./mk.sh /tmp/mk_B5 "$X" /tmp/mk_B4
echo CHAIN DONE
