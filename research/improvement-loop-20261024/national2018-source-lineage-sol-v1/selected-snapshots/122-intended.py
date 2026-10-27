"""Intended first-preference marginal; never literal marks or final tabulation."""
import numpy as np
from .dependencies import ranking
from .emission import report_probabilities_tensor, validate_channel

VERSION='finite_menu_PL_intended_first_preference_report_v1'

def first_preference_tensor(worth,style,*,array_module=np):
    """Exact sum over selected finite menus; remaining ranks/marks integrate to1."""
    xp=array_module
    names=ranking.validate_style(style);menus=ranking.menus(style)
    if any(not m for m in menus):raise ValueError('Every supported menu needs a person')
    wi={names.index(x) for x in style['writein_ids']}
    menu_weights=ranking._menu_probabilities(worth,menus,wi)
    # Accumulate by fixed person axis without reassigning any missing person/tail.
    columns=[]
    for i in range(len(names)):
        value=worth[...,0]*0
        for menu,weight in zip(menus,menu_weights,strict=True):
            if i in menu:value=value+weight*worth[...,i]/xp.sum(worth[...,list(menu)],axis=-1)
        columns.append(value)
    return xp.stack(columns,axis=-1)

def first_preference(worth,style):
    x=np.asarray(worth,dtype=float)
    if x.ndim<1 or x.shape[-1]!=len(style['candidate_ids']) or not np.isfinite(x).all() or np.any(x<=0):raise ValueError('Strictly positive finite worths on exact person axis required')
    return first_preference_tensor(x,style)

def reported_pair_tensor(worth,style,channel,epsilon,profile_intensity,response,selection_weights,first_label,second_label,*,array_module=np):
    """Reported-pair share: Q_a/(Q_a+Q_b), Q_g=sum_dc I_dc*r_dc*s_dc*q_dcg.

    The denominator excludes other report labels/statuses, not other people from
    the latent law. No floor/clipping: zero pair mass is undefined.
    """
    xp=array_module;c=validate_channel(channel)
    if tuple(style['candidate_ids'])!=c.candidates:raise ValueError('Style/channel person ordering differs')
    if first_label==second_label or first_label not in c.labels or second_label not in c.labels:raise ValueError('Two distinct explicit report labels required')
    p=first_preference_tensor(worth,style,array_module=xp)
    q=report_probabilities_tensor(p,c.mapping,c.contamination,epsilon,array_module=xp)
    rates=xp.sum((profile_intensity*response*selection_weights)[...,None]*q,axis=(0,1))
    first,second=rates[c.labels.index(first_label)],rates[c.labels.index(second_label)]
    return first/(first+second), xp.log(first)-xp.log(second), first+second
