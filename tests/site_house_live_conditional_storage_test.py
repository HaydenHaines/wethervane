"""Tiny INVENTED storage/law fixtures only; no actual forecast or scientific admission."""
from __future__ import annotations
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scripts.house_live_conditional_storage as adapter

def npy(dtype,shape,data):
    raw=repr({'descr':dtype,'fortran_order':False,'shape':tuple(shape)}).encode();raw+=b' '*((64-(10+len(raw)+1)%64)%64)+b'\n'
    return b'\x93NUMPY\x01\x00'+struct.pack('<H',len(raw))+raw+data

def array(dtype,shape,values):
    if dtype.startswith('<U'):
        width=int(dtype[2:]);data=b''.join(v.ljust(width,'\0').encode('utf-32-le') for v in values)
    else:data=struct.pack('<'+('q' if dtype=='<i8' else 'd')*len(values),*values)
    return dtype,list(shape),data

def make_fixture(root,ids=None,configuration=None,lengths=(2,4)):
    ids=ids or ['2026:AL-01:house:regular','2026:AL-02:house:regular']
    configuration=configuration or {'id':'house-current-polling','base':'fundamentals','polling':'standard','uncertainty':'standard'}
    def pin(path):return {'path':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    def put(path,value):
        path=root/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value if isinstance(value,bytes) else (json.dumps(value,indent=2)+'\n').encode());return pin(path)
    def archive(path,members):
        path=root/path;path.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for name,(dtype,shape,data) in members.items():z.writestr(name+'.npy',npy(dtype,shape,data))
        return pin(path)
    law=put('research/edition/law.json',{'INVENTED':'frozen law'});group={'schema':'InventedGroupLawV1'}
    law_fields={'configuration':configuration,'polling_error_scales':dict(independent=.01,cycle=.01,race=.02,house=.01,movement=.01),
        'exact_electoral_rule_simulation':False,'source_2026_outcomes_loaded':False,'group_completion':group}
    records=[];sources=[];source_arrays=[];offset=chains=0
    for i,count in enumerate(lengths):
        directory=f'predictions-{i}/run';base=f'research/edition/{directory}';settings={'chains':count//2,'draws':2,'warmup':1,'seed':2000+i}
        values=[(offset+j+1)/(sum(lengths)+2) for j in range(count) for _ in ids]
        members={'shares':array('<f8',(count,len(ids)),values),'race_ids':array('<U24',(len(ids),),ids),'joint_draw_id':array('<i8',(count,),list(range(count))),
            'shared_cycle':array('<f8',(count,),[-0.0]+[.25]*(count-1)),'coefficients':array('<f8',(count,2),[.1,.2]*count),'tau_cycle':array('<f8',(count,),[.3]*count),'tau_race':array('<f8',(count,),[.4]*count)}
        conditional=archive(base+'/conditional-shares.npz',members)
        joint=archive(base+'/joint-draws.npz',{k:array('<i8',(count,),v) for k,v in {'joint_draw_id':list(range(count)),'predictive_chain':[j//2 for j in range(count)],'predictive_draw':[j%2 for j in range(count)]}.items()})
        diagnostic=put(base+'/diagnostics.json',{'race_passed':True,'chamber_passed':True});means=put(base+'/conditional-group-probability-means.npy',b'INVENTED small means bytes');polls=put(base+'/polls.csv',b'INVENTED polls\n')
        driver=put(base+'/run-source.py',b'INVENTED original same driver bytes\n')
        plan={'schema':'HouseLivePredictionPlanV1','issued_at':'2099-01-01T00:00:00Z','pilot':False,'settings':settings,'inputs':[law],'source':driver,**copy.deepcopy(law_fields)}
        plan_pin=put(base+'/plan.json',plan);receipt_pin=put(base+'/receipt.json',{'schema':'HouseLivePredictionReceiptV1','configuration':configuration,'status':'passed','plan':plan_pin,'outputs':[diagnostic,joint,conditional,means,polls]})
        records.append({'directory':directory,'receipt':receipt_pin,'plan':plan_pin,'settings':settings,'joint_draw_offset':offset,'predictive_chain_offset':chains})
        sources.append(dict(conditional,rows=count,row_offset=offset,row_stop=offset+count,members={name:{'dtype':dtype,'shape':shape} for name,(dtype,shape,_) in members.items()}));source_arrays.append(members)
        offset+=count;chains+=settings['chains']
    combined={}
    for name,(dtype,shape,_) in source_arrays[0].items():
        if name=='race_ids':combined[name]=source_arrays[0][name]
        elif name=='joint_draw_id':combined[name]=array(dtype,(offset,),list(range(offset)))
        else:combined[name]=(dtype,[offset]+shape[1:],b''.join(s[name][2] for s in source_arrays))
    raw=archive('legacy/conditional-shares.npz',combined)
    specs={name:{'dtype':dtype,'shape':shape,'row_axis':None if name=='race_ids' else 0,'reader_rule':'shared_exact' if name=='race_ids' else 'global_arange_local_verified' if name=='joint_draw_id' else 'copy_source_rows'} for name,(dtype,shape,_) in combined.items()}
    manifest={'schema':adapter.SCHEMA,'source_count':len(sources),'total_rows':offset,'member_names':list(specs),'members':specs,'race_ids':ids,'sources':sources}
    base='research/edition/combined-v2/run';manifest_pin=put(base+'/conditional-shares.index.json',manifest)
    joint_pin=archive(base+'/joint-draws.npz',{k:array('<i8',(offset,),v) for k,v in {'joint_draw_id':list(range(offset)),'predictive_chain':[j//2 for j in range(offset)],'predictive_draw':[j%2 for j in range(offset)]}.items()})
    diagnostic=put(base+'/diagnostics.json',{'race_passed':True,'chamber_passed':True});means=put(base+'/conditional-group-probability-means.npy',b'INVENTED selected means');polls=put(base+'/polls.csv',b'INVENTED selected polls\n');driver=put(base+'/run-source.py',b'INVENTED combine driver\n')
    plan={'schema':'HouseLiveCombinedPredictionPlanV2','issued_at':'2099-01-01T00:00:00Z','pilot':False,'settings':{'chains':chains,'draws':2,'warmup_per_source':1},'source_runs':records,
        'inputs':[law]+[r['receipt'] for r in records],'source':driver,'combination':'INVENTED all rows preserving original chains',**copy.deepcopy(law_fields),
        'conditional_storage':{'schema':adapter.SCHEMA,'manifest':manifest_pin,'weights':adapter.WEIGHTS}}
    plan_pin=put(base+'/plan.json',plan)
    receipt_pin=put(base+'/receipt.json',{'schema':'HouseLivePredictionReceiptV2','configuration':configuration,'status':'passed','plan':plan_pin,'outputs':[diagnostic,joint_pin,manifest_pin,means,polls]})
    return dict(manifest=manifest_pin,plan=plan_pin,receipt=receipt_pin,raw=raw,ids=ids,sourceRecords=records,sourceArrays=source_arrays,combined=combined,put=put,pin=pin,archive=archive)

def rebound_child(root,f,mutate):
    p=json.loads((root/f['plan']['path']).read_text());record=p['source_runs'][1];old_receipt=record['receipt'];child=json.loads((root/record['plan']['path']).read_text());mutate(child)
    record['plan']=f['put'](record['plan']['path'],child);receipt=json.loads((root/old_receipt['path']).read_text());receipt['plan']=record['plan'];record['receipt']=f['put'](old_receipt['path'],receipt)
    p['inputs']=[record['receipt'] if item==old_receipt else item for item in p['inputs']];f['plan']=f['put'](f['plan']['path'],p)
    r=json.loads((root/f['receipt']['path']).read_text());r['plan']=f['plan'];f['receipt']=f['put'](f['receipt']['path'],r)

class ConditionalStorageTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.addCleanup(self.temp.cleanup);self.f=make_fixture(self.root)
    def verify(self):return adapter.verify_prediction_storage(self.root,self.f['manifest'],self.f['plan'],self.f['receipt'],self.f['ids'])
    def test_every_member_bit_parity_original_order_and_cross_source_rebatching(self):
        raw=adapter.ConditionalStorage(self.root,self.f['raw']);seg=adapter.ConditionalStorage(self.root,self.f['manifest'])
        for name in raw.files:
            for chunk in (1,3,8):
                a=list(raw.iter_blocks(name,chunk));b=list(seg.iter_blocks(name,chunk));self.assertEqual(a,b,name)
                self.assertEqual(b''.join(x[2] for x in b),self.f['combined'][name][2])
        result=self.verify();self.assertEqual((result['rows'],result['chains'],result['sourceCount']),(6,3,2));self.assertEqual(result['weightPerRow'],1/6);self.assertFalse(result['publicationAdmission'])
    def test_manifest_mutations_gaps_overlaps_duplicates_shapes_rules_and_traversal_fail(self):
        original=json.loads((self.root/self.f['manifest']['path']).read_text())
        changes=[lambda m:m['sources'][1].__setitem__('row_offset',1),lambda m:m['sources'][1].__setitem__('row_stop',7),lambda m:m['sources'].reverse(),lambda m:m['sources'][1].update({k:m['sources'][0][k] for k in ('path','sha256')}),lambda m:m['sources'][0].__setitem__('path','../escape.npz'),lambda m:m['members']['shares'].__setitem__('reader_rule','resample'),lambda m:m['sources'][0]['members']['shares'].__setitem__('dtype','<f4'),lambda m:m['race_ids'].reverse()]
        for change in changes:
            m=copy.deepcopy(original);change(m);pin=self.f['put']('mutated.json',m)
            with self.subTest(change=change),self.assertRaises((adapter.StorageError,FileNotFoundError)):adapter.ConditionalStorage(self.root,pin)
    def test_manifest_child_bytes_and_symlink_escape_fail(self):
        target=self.root/self.f['sourceRecords'][0]['plan']['path'];m=json.loads((self.root/self.f['manifest']['path']).read_text());path=self.root/m['sources'][0]['path'];path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(adapter.StorageError,'hash'):adapter.ConditionalStorage(self.root,self.f['manifest'])
        self.assertTrue(target.exists())
        outside=Path(self.temp.name+'-outside');outside.write_bytes(b'outside');self.addCleanup(lambda:outside.unlink(missing_ok=True));(self.root/'escape.npz').symlink_to(outside)
        with self.assertRaisesRegex(adapter.StorageError,'escapes'):adapter.ConditionalStorage(self.root,{'path':'escape.npz','sha256':hashlib.sha256(b'outside').hexdigest()})
    def test_original_law_seed_lengths_and_input_binding_mutations_are_rejected(self):
        for change in [lambda p:p['group_completion'].__setitem__('schema','different law'),lambda p:p['configuration'].__setitem__('polling','none'),lambda p:p['settings'].__setitem__('seed',2000),lambda p:p['settings'].__setitem__('draws',1),lambda p:p.__setitem__('inputs',[])]:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);f=make_fixture(root);rebound_child(root,f,change)
                with self.subTest(change=change),self.assertRaises(adapter.StorageError):adapter.verify_prediction_storage(root,f['manifest'],f['plan'],f['receipt'],f['ids'])
    def test_selected_failure_is_not_relaxed_and_original_failure_must_match_diagnostic(self):
        r=json.loads((self.root/self.f['receipt']['path']).read_text());r['status']='diagnostic_failure';pin=self.f['put'](self.f['receipt']['path'],r)
        with self.assertRaisesRegex(adapter.StorageError,'not passed'):adapter.verify_prediction_storage(self.root,self.f['manifest'],self.f['plan'],pin,self.f['ids'])
    def test_late_source_drift_does_not_yield_stale_rows(self):
        storage=adapter.ConditionalStorage(self.root,self.f['manifest']);child=storage.sources[0];path=self.root/child['path'];path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(adapter.StorageError,'hash'):list(storage.iter_blocks('shares'))
        with self.assertRaisesRegex(adapter.StorageError,'hash'):list(storage.iter_blocks('joint_draw_id'))
    def test_completed_original_precision_failure_is_allowed_only_with_truthful_diagnostics(self):
        plan=json.loads((self.root/self.f['plan']['path']).read_text());record=plan['source_runs'][0];old=record['receipt']
        receipt=json.loads((self.root/old['path']).read_text());receipt['status']='diagnostic_failure'
        diagnostic=next(p for p in receipt['outputs'] if p['path'].endswith('/diagnostics.json'))
        updated=self.f['put'](diagnostic['path'],{'race_passed':False,'chamber_passed':True});receipt['outputs']=[updated if p==diagnostic else p for p in receipt['outputs']]
        record['receipt']=self.f['put'](old['path'],receipt);plan['inputs']=[record['receipt'] if p==old else p for p in plan['inputs']]
        self.f['plan']=self.f['put'](self.f['plan']['path'],plan);selected=json.loads((self.root/self.f['receipt']['path']).read_text());selected['plan']=self.f['plan'];self.f['receipt']=self.f['put'](self.f['receipt']['path'],selected)
        self.assertEqual(self.verify()['rows'],6)
        self.f['put'](diagnostic['path'],{'race_passed':True,'chamber_passed':True})
        with self.assertRaisesRegex(adapter.StorageError,'hash'):self.verify()

    def test_local_draw_permutation_is_rejected_even_if_archive_and_manifest_rehashed(self):
        m=json.loads((self.root/self.f['manifest']['path']).read_text());members=copy.deepcopy(self.f['sourceArrays'][0]);members['joint_draw_id']=array('<i8',(2,),[1,0]);pin=self.f['archive'](m['sources'][0]['path'],members);m['sources'][0]['sha256']=pin['sha256'];manifest=self.f['put']('permuted.json',m)
        with self.assertRaisesRegex(adapter.StorageError,'draw/chain order'):adapter.ConditionalStorage(self.root,manifest)

    def test_npy_header_accepts_bounded_seven_million_rows_without_allocating_payload(self):
        def synthetic_header(rows):
            encoded=repr({'descr':'<i8','fortran_order':False,'shape':(rows,)}).encode('latin1')
            padding=(-(6+2+2+len(encoded)+1))%16;header=encoded+b' '*padding+b'\n'
            prefix=b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header
            info=SimpleNamespace(file_size=len(prefix)+rows*8)
            return prefix,info
        prefix,info=synthetic_header(7_000_000)
        parsed=adapter.header(io.BytesIO(prefix),info)
        self.assertEqual(parsed,{'dtype':'<i8','shape':[7_000_000]})
        prefix,info=synthetic_header(7_000_001)
        with self.assertRaisesRegex(adapter.StorageError,'oversized'):
            adapter.header(io.BytesIO(prefix),info)

    def test_segmented_manifest_total_is_also_bounded_at_seven_million(self):
        manifest=json.loads((self.root/self.f['manifest']['path']).read_text())
        manifest['total_rows']=7_000_001
        pin=self.f['put']('oversized-total.json',manifest)
        with self.assertRaisesRegex(adapter.StorageError,'manifest totals'):
            adapter.ConditionalStorage(self.root,pin)

if __name__=='__main__':
    if '--fixture' in sys.argv:
        request=json.load(sys.stdin);f=make_fixture(Path(request['root']),request['ids'],request['configuration'],(2,2));print(json.dumps({k:v for k,v in f.items() if k in ('manifest','plan','receipt','raw','ids','sourceRecords')}))
    else:unittest.main()
