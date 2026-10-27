"""Invented role-mass invariants; imports only the audited stdlib prototype."""
import json, math, sys
sys.path.insert(0, "/home/hayden/projects/wethervane-redesign-20261006/integration/research/senate-stack-integration-20261020/national-2018-gpu-preparation-root-v1/candidate-calibration-review-v1")
from role_aware_prototype import Candidate, category_probabilities, expected_marks, partition_observed

def candidate_masses(p):
    return p["named"]
major=[Candidate("D","major",True,math.log(3)),Candidate("R","major",True,0)]
printed=[Candidate("P","printed_other",True,0)]
params=dict(response=.8,other_mass=.1,writein_fraction=.25,
            noncandidate={"blank":.3,"invalid":.2,"no_response":.5})
one=category_probabilities(major+printed+[Candidate("W1","write_in",True,0)],**params)
many=category_probabilities(major+printed+[Candidate(f"W{i}","write_in",True,i/10) for i in range(1,33)],**params)
for p in (one,many):
    assert math.isclose(p["named"]["D"]+p["named"]["R"],.72)
    assert math.isclose(sum(p["named"][n] for n in p["named"] if n.startswith("W")),.02)
    assert math.isclose(p["named"]["P"],.06)
    assert math.isclose(sum(p["noncandidate"].values()),.2)
    assert math.isclose(sum(p["named"].values())+sum(p["noncandidate"].values()),1)
    assert math.isclose(p["named"]["D"]/p["named"]["R"],3)
marks=expected_marks(1000,many)
assert math.isclose(sum(marks["named"].values())+sum(marks["noncandidate"].values()),1000)
assert math.isclose(sum(marks["named"][n] for n in marks["named"] if n.startswith("W")),20)
# Zero-event support is explicit; positive response with no candidate mass fails.
empty=category_probabilities([],response=0,other_mass=0,writein_fraction=0,noncandidate={"blank":1})
assert empty=={"named":{},"noncandidate":{"blank":1.0}}
try:
    category_probabilities([],response=.5,other_mass=0,writein_fraction=0,noncandidate={"blank":1})
except ValueError: pass
else: raise AssertionError("positive response without candidates must fail")
# Observed partition preserves counts, not modeled allocation.
observed=partition_observed(major+printed+[Candidate("W","write_in",True,0)],["blank"],
                            {"D":40,"R":30,"P":3,"W":1},{"blank":5})
assert observed["named"]=={"D":40,"R":30,"P":3,"W":1} and observed["total"]==79
print(json.dumps({"status":"PASS","eligible_writein_count":32,"principal_pair_named_mass":.72,
 "printed_nonmajor_named_mass":.06,"writein_named_mass":.02,"noncandidate_mass":.2,
 "expected_marks_total":1000,"observed_marks_partition_total":79},indent=2))
