"""Deterministic expected allocation for invented single-mark profiles; no RNG or fit."""
import math

def softmax(utilities):
 values=tuple(utilities)
 if len(values)<2 or any(type(x)not in(int,float)or not math.isfinite(x)for x in values):raise ValueError('finite D/R plus optional minor utilities required')
 peak=max(values);weights=[math.exp(x-peak)for x in values];total=math.fsum(weights)
 return tuple(x/total for x in weights)

def allocate(profiles,response,noncandidate=(1.,)):
 if type(response)not in(int,float)or not math.isfinite(response)or not 0<=response<=1:raise ValueError('response probability required')
 nc=tuple(noncandidate)
 if not nc or any(type(x)not in(int,float)or not math.isfinite(x)or x<0 for x in nc)or not math.isclose(math.fsum(nc),1.,rel_tol=0,abs_tol=1e-14):raise ValueError('noncandidate simplex required')
 rows=[];width=None
 for total,utilities in profiles:
  if type(total)not in(int,float)or not math.isfinite(total)or total<0:raise ValueError('nonnegative finite profile participant total required')
  probability=softmax(utilities)
  if width is not None and width!=len(probability):raise ValueError('identical candidate roster across profiles required')
  width=len(probability);named=tuple(total*response*x for x in probability);non=tuple(total*(1-response)*x for x in nc)
  rows.append({'participants':total,'named':named,'noncandidate':non,'major_ratio':named[0]/named[1]if named[1]>0 else None})
 if not rows:raise ValueError('profiles required')
 named=tuple(math.fsum(r['named'][i]for r in rows)for i in range(width));non=tuple(math.fsum(r['noncandidate'][i]for r in rows)for i in range(len(nc)))
 return {'profiles':rows,'participants':math.fsum(r['participants']for r in rows),'named':named,'noncandidate':non,'major_ratio':named[0]/named[1]if named[1]>0 else None,'major_pair_share':named[0]/math.fsum(named[:2])if math.fsum(named[:2])>0 else None,'minor_marks':math.fsum(named[2:])}
