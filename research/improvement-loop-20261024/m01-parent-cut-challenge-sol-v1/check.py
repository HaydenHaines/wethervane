from fractions import Fraction as F
import json
# Two positive states per parent, eight fixed upstream masses. No statistical imports.
w=[F(1,8)]*8
K=[[F(1),F(1)]]*7+[[F(2),F(2)]]
Z=[sum(k) for k in K]
q=[[k/z for k in row]for row,z in zip(K,Z)]
assert q==[[F(1,2),F(1,2)]]*8
cut=[[mass*x for x in row]for mass,row in zip(w,q)]
cut_parent=[sum(row)for row in cut]
uncut_raw=[[mass*x for x in row]for mass,row in zip(w,K)]
uncut_total=sum(sum(row)for row in uncut_raw)
uncut_parent=[sum(row)/uncut_total for row in uncut_raw]
assert cut_parent==w and uncut_parent==[F(1,9)]*7+[F(2,9)]
# Predict a future observable that equals 1 only under the final parent.
assert cut_parent[-1]==F(1,8) and uncut_parent[-1]==F(2,9)
print(json.dumps({'scope':'invented exact finite-support counterexample, not actual model evidence','Z':[str(x)for x in Z],'fixedCutParentMasses':[str(x)for x in cut_parent],'forbiddenEvidenceNormalizedParentMasses':[str(x)for x in uncut_parent],'predictiveProbabilityLastParentEvent':{'cut':str(cut_parent[-1]),'uncut':str(uncut_parent[-1])},'allWithinParentConditionalsIdentical':True},indent=2))
