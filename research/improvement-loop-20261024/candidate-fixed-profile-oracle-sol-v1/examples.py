import json,math
from oracle import allocate
out={}
for k in [0,1,4,12]:out['single_minor_count_'+str(k)]=allocate([(1000,[math.log(3),0]+[math.log(.25)]*k)],.8,(.5,.25,.25))
for minor in [False,True]:
 tail=[math.log(10)]if minor else[]
 out['multi_minor_'+str(minor)]=allocate([(100,[math.log(9),0]+tail),(100,[0,0]+tail)],.8)
print(json.dumps(out,indent=2))
