"""Only the four explicitly transcribed Census worked examples, not a cell adapter."""
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import math

@dataclass(frozen=True)
class Mapping:
    year:int
    geography:str
    characteristic:str
    parameter_table:int
    illustration_table:int
    a:float|None
    b:float
    unit:str

MAPPINGS={
 '2018-total':Mapping(2018,'US','unemployed_men_age16plus',7,3,-.000031,2947.,'persons'),
 '2020-total':Mapping(2020,'US','unemployed_men_age16plus',7,3,-.000031,2947.,'persons'),
 '2018-percent':Mapping(2018,'US','reported_voting_among_elementary_education_plus',8,4,None,3496.,'percentage_points'),
 '2020-percent':Mapping(2020,'US','reported_voting_among_elementary_education_plus',8,4,None,5411.,'percentage_points'),
}

def finite(x):
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):raise ValueError('finite real required')
    return x

def root(variance):
    if not math.isfinite(variance) or variance<0:raise ValueError('invalid variance; no clipping')
    return math.sqrt(variance)

def total_se(x,a,b):
    for v in (x,a,b):finite(v)
    if x<0 or b<0:raise ValueError('negative total/b coefficient')
    return root(a*x*x+b*x)

def percentage_se(p,base,b):
    for v in (p,base,b):finite(v)
    if not 0<=p<=100 or base<=0 or b<0:raise ValueError('percentage/base/coefficient outside domain')
    return root((b/base)*p*(100-p))

def checked_se(key,mapping,*,x=None,p=None,base=None):
    if key not in MAPPINGS or mapping!=MAPPINGS[key]:raise ValueError('unsupported year/table/geography/characteristic/coefficient/unit mapping')
    if mapping.unit=='persons':
        if p is not None or base is not None or x is None:raise ValueError('total requires persons x only')
        return total_se(x,mapping.a,mapping.b)
    if x is not None or p is None or base is None:raise ValueError('percentage requires p in 0..100 and person base only')
    return percentage_se(p,base,mapping.b)

def rounded(value,quantum):
    if quantum<=0:raise ValueError('positive rounding quantum')
    return float((Decimal(str(value))/Decimal(str(quantum))).quantize(Decimal(1),rounding=ROUND_HALF_UP)*Decimal(str(quantum)))
