import json
from pathlib import Path
from gvf import MAPPINGS,checked_se,rounded

def check():
    rows=json.loads((Path(__file__).parent/'examples.json').read_text());out=[]
    for row in rows:
        m=MAPPINGS[row['id']]
        for field in ['year','geography','unit','parameter_table','illustration_table']:
            if getattr(m,field)!=row[field]:raise ValueError('example metadata mismatch: '+field)
        se=checked_se(row['id'],m,**{k:row[k]for k in ['x','p','base']if k in row})
        observed=rounded(se,row['rounding_quantum'])
        if observed!=row['published_se']:raise ValueError('published rounded illustration mismatch')
        out.append({**row,'mapping':m.__dict__,'computed_se':se,'rounded_se':observed,'passed':True})
    return {'scope':'four national worked-example arithmetic checks only; no microdata/covariance/fit','examples':out,'passed':len(out)}
if __name__=='__main__':print(json.dumps(check(),indent=2,allow_nan=False))
