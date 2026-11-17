#!/bin/bash
# kill+resume chain: kill in round 1, mid-evolution, mid-validation, then finish
X='"hard_h": 99, "max_round": 2'
./mk.sh /tmp/mk_G1 "$X, \"kill_at\": \"r1:4\""
./mk.sh /tmp/mk_G2 "$X, \"kill_at\": \"state:1,3\"" /tmp/mk_G1
./mk.sh /tmp/mk_G3 "$X, \"kill_at\": \"state:2,2\"" /tmp/mk_G2
./mk.sh /tmp/mk_G4 "$X, \"kill_at\": \"val:3\"" /tmp/mk_G3
./mk.sh /tmp/mk_G5 "$X" /tmp/mk_G4
echo CHAIN DONE
