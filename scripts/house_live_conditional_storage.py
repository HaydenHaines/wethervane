"""Offline authenticated conditional storage; no prediction, source discovery or publication admission.

Raw NPZ V1 remains supported. Segmented V1 storage is an ordered concatenation
of original NPZ row bytes, with shared race IDs and verified local/global draw
IDs. The explicit operator root is the sole base for research-relative paths.
This reader supports the native C-order primitive NPY arrays, without NumPy.
"""
from __future__ import annotations
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import zipfile

SCHEMA = 'ConditionalSegmentsManifestV1'
WEIGHTS = 'equal_weight_per_original_joint_row'
MAX_ROWS = 7_000_000
SHA = re.compile(r'^[0-9a-f]{64}$')
PATH = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._/-]*$')
LAW_KEYS = ('configuration','polling_error_scales','exact_electoral_rule_simulation','source_2026_outcomes_loaded','group_completion')

class StorageError(ValueError): pass

def need(v, message):
    if not v: raise StorageError(message)

def integer(v, minimum=0): return type(v) is int and v >= minimum

def valid_pin(v):
    return isinstance(v,dict) and set(v)=={'path','sha256'} and isinstance(v['path'],str) and PATH.fullmatch(v['path']) and all(x not in ('','.','..') for x in v['path'].split('/')) and isinstance(v['sha256'],str) and SHA.fullmatch(v['sha256'])

def unique_object(pairs):
    result={}
    for k,v in pairs:
        need(k not in result,'Duplicate JSON field');result[k]=v
    return result

class PinnedFiles:
    def __init__(self,root):
        self.root=Path(root).resolve(strict=True);need(self.root.is_dir(),'Explicit conditional storage root must be a directory');self.checked={}
    def path(self,pin):
        need(valid_pin(pin),'Unsafe source pin/path')
        path=(self.root/pin['path']).resolve(strict=True)
        need(path.is_relative_to(self.root) and path!=self.root and path.is_file(),'Source path escapes explicit conditional storage root')
        return path
    @staticmethod
    def identity(path):
        s=path.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    def verify(self,pin):
        path=self.path(pin);before=self.identity(path);key=(pin['path'],pin['sha256'])
        if self.checked.get(key)!=before:
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
            need(digest.hexdigest()==pin['sha256'] and self.identity(path)==before,'Source byte hash changed: '+pin['path'])
            self.checked[key]=before
        return path
    def read_json(self,pin):
        path=self.verify(pin);value=json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=unique_object,parse_constant=lambda x:(_ for _ in ()).throw(StorageError('Nonfinite JSON')))
        self.verify(pin);need(isinstance(value,dict),'Expected JSON object');return value

def dtype_size(dtype):
    match=re.fullmatch(r'([<>|])([iufUbS])(\d+)',dtype or '')
    need(match is not None,'Unsupported primitive array dtype')
    size=int(match[3]);need(size>0,'Invalid dtype width')
    if match[2]=='U':
        need(size<=512,'Unicode dtype exceeds native identity bound');return 4*size
    need(size in (1,2,4,8),'Unsupported primitive item size');return size

def header(stream,info):
    need(stream.read(6)==b'\x93NUMPY','Invalid NPY magic');version=stream.read(2)
    need(version in (b'\x01\x00',b'\x02\x00',b'\x03\x00'),'Unsupported NPY version')
    length_bytes=2 if version[0]==1 else 4;raw=stream.read(length_bytes);need(len(raw)==length_bytes,'Truncated NPY header')
    length=int.from_bytes(raw,'little');need(0<length<=16384,'NPY header exceeds bound')
    raw=stream.read(length);need(len(raw)==length,'Truncated NPY header')
    value=ast.literal_eval(raw.decode('utf-8' if version[0]==3 else 'latin1').strip())
    need(isinstance(value,dict) and set(value)=={'descr','fortran_order','shape'} and value['fortran_order'] is False,'Only primitive native C-order arrays supported')
    shape=value['shape'];need(isinstance(shape,tuple) and shape and all(integer(x) for x in shape) and shape[0]<=MAX_ROWS,'Invalid or oversized native array shape')
    size=dtype_size(value['descr']);need(size*math.prod(shape[1:])<=16384,'Native row exceeds bounded block width');need(info.file_size==8+length_bytes+length+math.prod(shape)*size,'NPY payload length differs')
    return {'dtype':value['descr'],'shape':list(shape)}

def inspect(files,pin):
    path=files.verify(pin)
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist();need(names and all(name.endswith('.npy') and '/' not in name for name in names),'Only flat NPZ members supported')
        names=[name[:-4] for name in names];need(len(set(names))==len(names),'Duplicate NPZ member')
        metadata={}
        for name in names:
            with archive.open(name+'.npy') as stream:metadata[name]=header(stream,archive.getinfo(name+'.npy'))
    files.verify(pin);return names,metadata

def member_blocks(files,pin,name,chunk_rows=2048):
    need(integer(chunk_rows,1) and chunk_rows<=4096,'Invalid block size');path=files.verify(pin)
    with zipfile.ZipFile(path) as archive:
        with archive.open(name+'.npy') as stream:
            meta=header(stream,archive.getinfo(name+'.npy'));row_bytes=dtype_size(meta['dtype'])*math.prod(meta['shape'][1:]);start=0
            while start<meta['shape'][0]:
                rows=min(chunk_rows,meta['shape'][0]-start);data=stream.read(rows*row_bytes);need(len(data)==rows*row_bytes,'Truncated member block')
                yield start,rows,data;start+=rows
            need(stream.read(1)==b'','Trailing member bytes')
    files.verify(pin)

def identities(files,pin,meta):
    names=meta.keys();need({'shares','race_ids','joint_draw_id'}<=set(names),'Missing required conditional members')
    races=meta['race_ids'];need(re.fullmatch(r'[<>]U\d+',races['dtype']) and len(races['shape'])==1,'Invalid race identity dtype/shape')
    ids=meta['joint_draw_id'];need(ids['dtype'] in ('<i8','>i8','<u8','>u8') and len(ids['shape'])==1 and ids['shape'][0]>0,'Invalid draw identity dtype/shape')
    rows=ids['shape'][0]
    need(all(name=='race_ids' or value['shape'][0]==rows for name,value in meta.items()),'Conditional row member lengths differ')
    raw=b''.join(b for _,_,b in member_blocks(files,pin,'race_ids'));size=dtype_size(races['dtype']);encoding='utf-32-le' if races['dtype'][0]=='<' else 'utf-32-be'
    race_ids=[raw[i:i+size].decode(encoding).rstrip('\0') for i in range(0,len(raw),size)]
    need(race_ids and all(race_ids) and len(set(race_ids))==len(race_ids),'Duplicate/empty race identity')
    need(meta['shares']['dtype'] in ('<f8','>f8') and meta['shares']['shape']==[rows,len(race_ids)],'Share row/race shape differs')
    check_integer_member(files,pin,'joint_draw_id',ids,lambda i:i)
    return rows,race_ids

def check_integer_member(files,pin,name,meta,expected):
    need(meta['dtype'] in ('<i8','>i8','<u8','>u8'),'Unsupported identity integer dtype');fmt=meta['dtype'][0]+('q' if meta['dtype'][1]=='i' else 'Q')
    need(len(meta['shape'])==1,'Identity must be one-dimensional')
    for start,rows,block in member_blocks(files,pin,name):
        need(all(value[0]==expected(start+i) for i,value in enumerate(struct.iter_unpack(fmt,block))),'Original draw/chain order changed: '+name)

class ConditionalStorage:
    def __init__(self,root,pin,expected_race_ids=None):
        self.filesystem=PinnedFiles(root);self.pin=dict(pin);self.manifest=None;path=self.filesystem.verify(pin)
        if path.suffix=='.npz':
            self.files,self.members=inspect(self.filesystem,pin);self.total_rows,self.race_ids=identities(self.filesystem,pin,self.members)
            self.sources=[dict(pin,rows=self.total_rows,row_offset=0,row_stop=self.total_rows,members=self.members)]
        else:
            need(path.suffix=='.json','Unsupported conditional storage suffix');m=self.filesystem.read_json(pin);self.manifest=m
            need(set(m)=={'schema','source_count','total_rows','member_names','members','race_ids','sources'} and m['schema']==SCHEMA,'Unsupported manifest schema/fields')
            self.files=m['member_names'];self.members=m['members'];self.total_rows=m['total_rows'];self.race_ids=m['race_ids'];self.sources=m['sources']
            need(isinstance(self.files,list) and self.files and len(set(self.files))==len(self.files) and all(isinstance(x,str) and x for x in self.files),'Invalid member order')
            need(isinstance(self.members,dict) and set(self.members)==set(self.files) and isinstance(self.sources,list) and self.sources and integer(m['source_count'],1) and m['source_count']==len(self.sources) and integer(self.total_rows,1) and self.total_rows<=MAX_ROWS,'Invalid manifest totals/support')
            offset=0;seen_paths=set();seen_hashes=set();first=None
            for source in self.sources:
                need(isinstance(source,dict) and set(source)=={'path','sha256','rows','row_offset','row_stop','members'},'Invalid manifest source fields')
                child={k:source[k] for k in ('path','sha256')};need(valid_pin(child) and source['path'].endswith('.npz'),'Invalid original child pin')
                need(source['path'] not in seen_paths and source['sha256'] not in seen_hashes,'Duplicate source path/hash');seen_paths.add(source['path']);seen_hashes.add(source['sha256'])
                need(integer(source['rows'],1) and type(source['row_offset']) is int and type(source['row_stop']) is int and source['row_offset']==offset and source['row_stop']==offset+source['rows'],'Source gap/overlap or reorder')
                names,meta=inspect(self.filesystem,child);rows,races=identities(self.filesystem,child,meta)
                need(names==self.files and meta==source['members'] and rows==source['rows'] and races==self.race_ids,'Source members/race identity differ')
                for name in self.files:
                    combined=self.members[name];need(set(combined)=={'dtype','shape','row_axis','reader_rule'},'Invalid combined member fields')
                    shape=meta[name]['shape'];need(combined['dtype']==meta[name]['dtype'],'Source dtype changed')
                    if name=='race_ids':need(combined==dict(meta[name],row_axis=None,reader_rule='shared_exact'),'Shared race rule differs')
                    else:
                        rule='global_arange_local_verified' if name=='joint_draw_id' else 'copy_source_rows'
                        need(combined=={'dtype':meta[name]['dtype'],'shape':[self.total_rows]+shape[1:],'row_axis':0,'reader_rule':rule},'Combined shape/reader rule differs')
                    if first is not None:need(meta[name]['dtype']==first[name]['dtype'] and (shape==first[name]['shape'] if name=='race_ids' else shape[1:]==first[name]['shape'][1:]),'Source trailing shape/dtype changed')
                first=meta;offset+=rows
            need(offset==self.total_rows,'Source rows do not cover total')
            dtype=self.members['joint_draw_id']['dtype'];need(self.total_rows-1<2**(63 if dtype[1]=='i' else 64),'Global draw IDs overflow source dtype')
        need(expected_race_ids is None or self.race_ids==expected_race_ids,'Ordered pair cohort differs')
    def iter_blocks(self,name,chunk_rows=2048):
        need(name in self.files,'Unknown member');need(integer(chunk_rows,1) and chunk_rows<=4096,'Invalid block size')
        self.filesystem.verify(self.pin)
        for source in self.sources:self.filesystem.verify({k:source[k] for k in ('path','sha256')})
        if name=='race_ids':
            yield from member_blocks(self.filesystem,{k:self.sources[0][k] for k in ('path','sha256')},name,chunk_rows);return
        if name=='joint_draw_id' and self.manifest:
            fmt=self.members[name]['dtype'][0]+('q' if self.members[name]['dtype'][1]=='i' else 'Q')
            for start in range(0,self.total_rows,chunk_rows):
                rows=min(chunk_rows,self.total_rows-start);yield start,rows,b''.join(struct.pack(fmt,i) for i in range(start,start+rows))
            return
        size=dtype_size(self.members[name]['dtype'])*math.prod(self.members[name]['shape'][1:]);pending=bytearray();start=0
        for source in self.sources:
            for _,_,block in member_blocks(self.filesystem,{k:source[k] for k in ('path','sha256')},name,chunk_rows):
                pending.extend(block)
                while len(pending)>=chunk_rows*size:
                    yield start,chunk_rows,bytes(pending[:chunk_rows*size]);del pending[:chunk_rows*size];start+=chunk_rows
        if pending:
            rows=len(pending)//size;need(rows*size==len(pending),'Partial row');yield start,rows,bytes(pending);start+=rows
        need(start==self.total_rows,'Stream row coverage changed');self.filesystem.verify(self.pin)


def verify_prediction_storage(root,manifest_pin,plan_pin,receipt_pin,expected_race_ids):
    """Bind exact V2 selected plan and original V1 sources; scientific qualification remains separate."""
    files=PinnedFiles(root);plan=files.read_json(plan_pin);receipt=files.read_json(receipt_pin)
    need(plan.get('schema')=='HouseLiveCombinedPredictionPlanV2' and plan.get('pilot') is False and plan.get('source_2026_outcomes_loaded') is False,'Unsupported V2 plan/safety fields')
    need(receipt.get('schema')=='HouseLivePredictionReceiptV2' and receipt.get('status')=='passed' and receipt.get('plan')==plan_pin and receipt.get('configuration')==plan.get('configuration'),'Selected V2 receipt/plan mismatch or not passed')
    need(plan.get('conditional_storage')==dict(schema=SCHEMA,manifest=manifest_pin,weights=WEIGHTS),'V2 manifest/weight binding differs')
    storage=ConditionalStorage(root,manifest_pin,expected_race_ids)
    need(storage.manifest is not None and len(plan.get('source_runs',[]))==len(storage.sources)>=2,'V2 source count differs')
    need(all(integer(plan['settings'][k],1) for k in ('chains','draws')) and integer(plan['settings']['warmup_per_source']),'Invalid selected chain settings')
    def bound_run(value):
        inputs=value['inputs'];need(isinstance(inputs,list) and len({p['path'] for p in inputs})==len(inputs),'Duplicate law input path')
        for p in inputs+[value['source']]:files.verify(p)
    bound_run(plan)
    need(isinstance(receipt.get('outputs'),list) and len({p['path'] for p in receipt['outputs']})==len(receipt['outputs']),'Duplicate selected output')
    for p in receipt['outputs']:files.verify(p)
    outputs={Path(p['path']).name:p for p in receipt['outputs']}
    need(set(outputs)=={'diagnostics.json','joint-draws.npz','conditional-shares.index.json','conditional-group-probability-means.npy','polls.csv'} and outputs['conditional-shares.index.json']==manifest_pin,'Selected V2 output membership differs')
    diagnostic=files.read_json(outputs['diagnostics.json']);need(diagnostic.get('race_passed') is True and diagnostic.get('chamber_passed') is True,'Selected diagnostics not passed')
    edition=Path(plan_pin['path']).parents[2];need(Path(receipt_pin['path']).parent==Path(plan_pin['path']).parent,'Selected receipt/plan directory differs')
    rows=chains=0;seeds=set();originals=[]
    for record,segment in zip(plan['source_runs'],storage.sources,strict=True):
        directory=record['directory'];need(isinstance(directory,str) and PATH.fullmatch(directory) and all(x not in ('','.','..') for x in directory.split('/')),'Unsafe original source directory')
        parent=edition/directory;need(record['plan']['path']==str(parent/'plan.json') and record['receipt']['path']==str(parent/'receipt.json'),'Original source directory/pins differ')
        original=files.read_json(record['plan']);r=files.read_json(record['receipt']);settings=original['settings']
        need(original.get('schema')=='HouseLivePredictionPlanV1' and original.get('pilot') is False and r.get('schema')=='HouseLivePredictionReceiptV1' and r.get('status') in ('passed','diagnostic_failure') and r.get('plan')==record['plan'] and r.get('configuration')==original.get('configuration'),'Original source is not a completed exact V1 run')
        need(settings==record['settings'] and all(integer(settings[k],1) for k in ('chains','draws')) and integer(settings['warmup']) and integer(settings['seed']) and settings['seed'] not in seeds,'Original source settings/seed differ')
        seeds.add(settings['seed']);count=settings['chains']*settings['draws']
        need(type(record['joint_draw_offset']) is int and type(record['predictive_chain_offset']) is int and record['joint_draw_offset']==rows and record['predictive_chain_offset']==chains and segment['row_offset']==rows and segment['rows']==count and segment['path']==str(parent/'conditional-shares.npz'),'Original chain/row offsets or archive differ')
        need(settings['draws']==plan['settings']['draws'] and settings['warmup']==plan['settings']['warmup_per_source'],'Retained chain lengths/warmup differ')
        need(all(original.get(k)==plan.get(k) for k in LAW_KEYS),'Original predictive law changed')
        if originals:need(original['inputs']==originals[0]['inputs'] and original['source']['sha256']==originals[0]['source']['sha256'],'Original law inputs/driver changed')
        bound_run(original)
        need(record['receipt'] in plan['inputs'] and all(p in plan['inputs'] for p in original['inputs']),'Original law/receipt omitted from combined inputs')
        need(isinstance(r.get('outputs'),list) and len({p['path'] for p in r['outputs']})==len(r['outputs']),'Duplicate original output')
        for p in r['outputs']:files.verify(p)
        original_outputs={Path(p['path']).name:p for p in r['outputs']}
        need(set(original_outputs)=={'diagnostics.json','joint-draws.npz','conditional-shares.npz','conditional-group-probability-means.npy','polls.csv'} and original_outputs['conditional-shares.npz']=={k:segment[k] for k in ('path','sha256')},'Original conditional output substitution')
        d=files.read_json(original_outputs['diagnostics.json']);need((r['status']=='passed')==(d.get('race_passed') is True and d.get('chamber_passed') is True),'Original receipt contradicts precision diagnostics')
        names,metadata=inspect(files,original_outputs['joint-draws.npz'])
        for name,expected in [('joint_draw_id',lambda i:i),('predictive_chain',lambda i:i//settings['draws']),('predictive_draw',lambda i:i%settings['draws'])]:
            need(name in names and metadata[name]['shape']==[count],'Original joint identity shape differs');check_integer_member(files,original_outputs['joint-draws.npz'],name,metadata[name],expected)
        rows+=count;chains+=settings['chains'];originals.append(original)
    need(rows==storage.total_rows==plan['settings']['chains']*plan['settings']['draws'] and chains==plan['settings']['chains'],'Combined total row/chain coverage differs')
    return {'schema':'HouseConditionalStorageVerificationV1','storageSchema':SCHEMA,'manifestSha256':manifest_pin['sha256'],'planSha256':plan_pin['sha256'],'receiptSha256':receipt_pin['sha256'],'rows':rows,'chains':chains,'sourceCount':len(storage.sources),'weights':WEIGHTS,'weightPerRow':1/rows,'orderedPairRaceIds':storage.race_ids,'publicationAdmission':False}
