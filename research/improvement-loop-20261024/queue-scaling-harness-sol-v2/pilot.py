"""Scratch-only queue pilot. --execute is never invoked by the unit tests."""
import argparse,hashlib,json,math,os,signal,socket,stat,struct,subprocess,sys,time,uuid
from pathlib import Path
from sensors import RealSensors,Uncertain,birth,fingerprint,numbers,cgroup_mount
HERE=Path(__file__).resolve().parent
MIB=1024**2

class Stop(RuntimeError):pass

def require(value,message):
 if not value:raise Stop(message)
def journal(root,name,value):
 path=Path(root)/name
 with path.open('x')as f:json.dump(value,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
 return fingerprint(path)

def fresh_scratch(parent,source):
 parent=Path(parent);source=Path(source).resolve()
 require(parent.is_absolute()and parent==parent.resolve(),'canonical scratch parent required')
 require(not parent.is_relative_to(source)and not source.is_relative_to(parent),'scratch parent cannot contain or live within source')
 require(not parent.is_relative_to(HERE)and not HERE.is_relative_to(parent),'scratch cannot overlap harness packet')
 require(parent.stat().st_dev==source.stat().st_dev,'scratch must share selected source filesystem')
 root=parent/('queue-pilot-'+uuid.uuid4().hex);root.mkdir(mode=0o700)
 fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
 return root

class Policy:
 def __init__(self,expected_mode='invented_test_only',admitted_events=None):
  require(expected_mode in ('real_unmodified_queue','invented_test_only'),'unknown policy mode')
  if expected_mode=='real_unmodified_queue':require(admitted_events is not None,'immutable admitted events required')
  self.expected_mode=expected_mode;self.events=dict(admitted_events if admitted_events is not None else {'oom':0,'oom_kill':0,'high':0});self.start=None;self.previous=None
  require(all(type(v)is int and v==0 for v in self.events.values()),'admitted events must be fresh zero counters')
 def seed(self,sample):self.check(sample,'warmup',cpu_warmup=True)
 def check(self,sample,phase,remaining=8,cpu_warmup=False):
  require(sample.get('mode')==self.expected_mode,'sample mode differs from admitted driver mode')
  for key in ['cpu_util','cpu_count','load1','host_memory_available','host_memory_total','free_bytes','free_inodes','probe_fsync_ms','scratch_allocated','scratch_entries','monotonic']:
   value=sample.get(key)
   if cpu_warmup and key=='cpu_util'and value is None:continue
   require(type(value)in(int,float)and math.isfinite(value),'missing/nonfinite sensor '+key)
  require(not sample.get('owners'),'protected owner workload present or started')
  c=sample['cgroup']
  require(sample['harness']['rss_bytes']<=256*MIB,'outer controller RSS cap exhausted')
  require(sample['cpu_count']>=c['quota_cores']+.25*sample['cpu_count'],'CPU capacity reserve unavailable')
  for ancestor in c['ancestors']:
   if cpu_warmup and ancestor['other_used_cores']is None:continue
   require(ancestor['other_used_cores']is not None and ancestor['capacity_cores']-ancestor['other_used_cores']>=c['quota_cores']+.25*ancestor['capacity_cores'],'ancestor CPU headroom unavailable')
  require(c['pids_current']<=8 and not any(c['pids_events'].values()),'process headroom exhausted')
  require(c['memory_current']+64*MIB<=c['identity']['memory_max'],'phase memory headroom unavailable')
  require(c['memory_peak']<=c['identity']['memory_max'],'group peak exceeds envelope')
  require(c['parent_memory_remaining']is None or c['parent_memory_remaining']>=512*MIB+64*MIB,'parent memory headroom unavailable')
  require((cpu_warmup and sample['cpu_util']is None or sample['cpu_util']<=.5)and sample['load1']/sample['cpu_count']<=.5,'host CPU headroom unavailable')
  require(sample['host_memory_available']>=max(2*1024*MIB,.2*sample['host_memory_total']),'host memory headroom unavailable')
  for source in [sample['host_pressure'],c['pressure']]:
   for kind,threshold in [('cpu',5.),('memory',.5),('io',1.)]:
    require(source[kind]['some']['avg10']<=threshold,'pressure admission/continuation stop: '+kind)
  require(sample['free_bytes']>=2*1024*MIB and sample['free_inodes']>=20000,'storage headroom unavailable')
  require(sample['probe_fsync_ms']<=50,'storage latency admission/continuation stop')
  require(sample['scratch_allocated']+max(1,remaining)*MIB<=128*MIB,'per-cycle storage reserve exhausted')
  require(sample['scratch_entries']+max(1,remaining)*128<=10000,'per-cycle inode reserve exhausted')
  if self.start is None:self.start=sample['monotonic']
  require(c['memory_events']==self.events,'memory pressure/OOM event changed')
  require(sample['monotonic']-self.start<=120,'batch wall budget exhausted')
  if self.previous:
   dt=sample['monotonic']-self.previous['monotonic'];require(dt>0,'nonmonotonic sample')
   # cpu.stat usage includes every detached member of the dedicated leaf.
   for key in ['usage_usec','throttled_usec']:
    require(c['cpu'][key]>=self.previous['cgroup']['cpu'][key],'cgroup CPU counter reset')
   throttled=(c['cpu']['throttled_usec']-self.previous['cgroup']['cpu']['throttled_usec'])/(dt*1e6)
   require(throttled<=.25,'CPU quota contention stop')
  self.previous=sample

class Pilot:
 def __init__(self,root,sensor,driver,clock=time.monotonic,sleep=time.sleep):
  require(sensor.mode==driver.mode,'real/fake path mixing prohibited');self.root=Path(root);self.sensor=sensor;self.driver=driver;self.clock=clock;self.sleep=sleep;self.policy=Policy(driver.mode,sensor.admitted_events);self.seq=0;self.phases=[]
 def sample(self,phase,remaining=8,warmup=False):
  self.seq+=1;s=self.sensor.sample();self.last_sensor_pin=journal(self.root,f'{self.seq:05d}-sensor.json',dict(phase=phase,sample=s,mode=self.driver.mode));self.policy.seed(s)if warmup and s['cpu_util']is None else self.policy.check(s,phase,remaining);return s
 def phase(self,operation,**args):
  self.sample('before-'+operation);self.driver.set_gate(self.last_sensor_pin);require(operation!='enqueue'or self.driver.launches()<16,'launch budget pre-admission stop');started=self.clock();handle=self.driver.start(operation,args);result=None
  while result is None:
   self.sample('during-'+operation);require(self.clock()-started<=5,'phase wall deadline: '+operation);result=self.driver.poll(handle)
   if result is None:self.sleep(.1)
  elapsed=self.clock()-started;self.sample('after-'+operation)
  require(result.get('mode')==self.driver.mode and result.get('exit_code')==0,'worker uncertain/nonzero return')
  require(self.driver.launches()<=16,'actual launch ceiling exceeded');record=dict(operation=operation,args=args,elapsed_seconds=elapsed,worker=result,actual_launches=self.driver.launches(),mode=self.driver.mode);self.phases.append(record);journal(self.root,f'phase-{len(self.phases):04d}.json',record)
  return result['value']
 def run(self):
  status='aborted';error=None
  try:
   # No queue worker starts until two successful resource samples.
   self.sample('baseline-1',warmup=True);self.sleep(.1);self.sample('baseline-2');self.phase('control');self.phase('initialize')
   for n in range(1,9):
    self.phase('enqueue',n=n)
    for _ in range(100):
     state=self.phase('tick');self.driver.verify_capacity(state)
     if state['all_jobs_recorded']and state['reserved']==0:break
     self.sleep(.1)
    else:raise Stop('polling ceiling without terminal readiness')
    self.driver.verify_terminal_prefix(n)
    if n in [1,2,4,8]:
     for _ in range(3):self.phase('tick')
     self.phase('status');self.phase('status') # separate fresh controller, never a cold-cache claim
     journal(self.root,f'prefix-{n}.json',dict(n=n,counts=self.driver.counts(),mode=self.driver.mode))
   self.phase('seal');self.phase('finalize');self.phase('finalize');self.phase('control');status='pilot_completed'if self.driver.mode=='real_unmodified_queue'else'invented_test_completed'
  except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc));status='aborted'
  try:cleanup=self.driver.cleanup()
  except BaseException as exc:cleanup=dict(verified=False,error=str(exc))
  if not cleanup.get('verified'):status='cleanup_uncertain'
  else:
   try:self.sample('after-cleanup')
   except BaseException as exc:
    status='aborted';error=dict(type=type(exc).__name__,message=str(exc))
  try:launches=self.driver.launches()
  except BaseException as exc:
   launches=None;status='cleanup_uncertain'if not cleanup.get('verified')else'aborted';error=dict(type=type(exc).__name__,message=str(exc))
  final=dict(status=status,error=error,mode=self.driver.mode,cleanup=cleanup,phases=len(self.phases),actual_launches=launches,scientific_cycles=0,scientific_qualification='unassessed',cache_policy='OS-managed; no drop; fresh-process != cold-cache')
  journal(self.root,'pilot-terminal.json',final);return final

FIXTURE='''import hashlib,json,os,pathlib,sys
p=pathlib.Path(sys.argv[1]);p.mkdir();data=b'x'*1024
with (p/'data.txt').open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
receipt={'schema':'TinyPilotV1','status':'computed','data_sha256':hashlib.sha256(data).hexdigest()}
with (p/'receipt.json').open('x')as f:json.dump(receipt,f);f.flush();os.fsync(f.fileno())
fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
'''

class RealDriver:
 mode='real_unmodified_queue'
 def __init__(self,root,source,cgroup):self.root=Path(root);self.source=Path(source);self.cg=Path(cgroup);self.index=0;self.children=[];self.maximum_observed_launches=0;self.identity=(self.cg.stat().st_dev,self.cg.stat().st_ino)
 def set_gate(self,pin):self.gate_pin=pin
 def start(self,operation,args):
  require(hasattr(self,'gate_pin'),'fresh controller gate absent')
  self.index+=1;name=f'worker-{self.index:04d}';request=dict(operation=operation,args=args,root=str(self.root),source=str(self.source),cgroup=str(self.cg),cgroup_identity=self.identity,parent_identity=birth(os.getpid()));journal(self.root,name+'-request.json',request)
  stdout=(self.root/(name+'-stdout')).open('xb');stderr=(self.root/(name+'-stderr')).open('xb')
  parent_channel,child_channel=socket.socketpair(socket.AF_UNIX,socket.SOCK_STREAM)
  try:
   p=subprocess.Popen([sys.executable,'-B',str(HERE/'pilot.py'),'--worker',str(self.root/(name+'-request.json')),'--cap-fd',str(child_channel.fileno())],stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,cwd=self.root,env={'PYTHONDONTWRITEBYTECODE':'1'},start_new_session=True,pass_fds=(child_channel.fileno(),))
   handle=dict(process=p,name=name,identity=None,pidfd=None,registration_error=None);self.children.append(handle)
  except BaseException:
   parent_channel.close();raise
  finally:stdout.close();stderr.close();child_channel.close()
  try:
   # The unreaped directly created child cannot have its PID recycled here.
   handle['pidfd']=os.pidfd_open(p.pid)
   handle['identity']=birth(p.pid)
   grant=dict(schema='PilotWorkerGrantV2',request=fingerprint(self.root/(name+'-request.json')),admission=fingerprint(self.root/'admission.json'),leaf_admission=fingerprint(self.root/'leaf-admission.json'),sensor=self.gate_pin,child=handle['identity'],parent=request['parent_identity'],issued_monotonic=time.monotonic())
   parent_channel.sendall((json.dumps(grant)+'\n').encode());return handle
  except BaseException as error:
   handle['registration_error']=str(error)
   if handle['pidfd']is not None:
    try:signal.pidfd_send_signal(handle['pidfd'],signal.SIGKILL)
    except ProcessLookupError:pass
   # No grant was delivered on identity failure; retained child waits or exits.
   try:p.wait(timeout=3)
   except subprocess.TimeoutExpired:pass
   raise
  finally:parent_channel.close()

 def poll(self,h):
  code=h['process'].poll()
  if code is None:return None
  result=self.root/(h['name']+'-result.json')
  require(result.is_file(),'worker exited without durable result');value=json.loads(result.read_text());require(all(value['identity'][k]==h['identity'][k]for k in ['pid','start_ticks','boot_id']),'worker birth acknowledgment differs')
  return dict(exit_code=code,mode=self.mode,value=value['value'],result=fingerprint(result),worker_operation_seconds=value['operation_seconds'],stdout=fingerprint(self.root/(h['name']+'-stdout')),stderr=fingerprint(self.root/(h['name']+'-stderr')))
 def launches(self):
  count=len(list((self.root/'queue'/'executions'/'attempts').glob('*/launch-intent.json')));require(count>=self.maximum_observed_launches,'launch history vanished or rolled back');self.maximum_observed_launches=count;return count
 def verify_capacity(self,state):require(state['capacity']==1 and state['reserved']<=1,'capacity-one invariant failed')
 def verify_terminal_prefix(self,n):
  results=list((self.root/'queue'/'queues').glob('*/jobs/*/result.json'));require(len(results)==n,'prefix result count differs')
  for path in results:
   value=json.loads(path.read_text());require(value['outcome']=='accepted_execution'and value['executor_terminal']is not None,'prefix not accepted/released')
   item=value['executor_terminal'];require(fingerprint(item['path'])['sha256']==item['sha256'],'terminal bytes differ')
 def counts(self):
  root=self.root/'queue';plans=[json.loads(p.read_text())for p in (root/'queues').glob('*/plan.json')]
  return dict(P=len(plans),J=sum(len(p['jobs'])for p in plans),R=len(list(root.glob('queues/*/jobs/*/reservation.json'))),A=len(list((root/'executions'/'ownership-intents').glob('*.json'))),E=sum(len(j['dependencies'])for p in plans for j in p['jobs']),H=1,scientific_cycles=0)
 def _recorded_identities(self):
  result={}
  for path in (self.root/'queue'/'executions'/'attempts').glob('*/*.json'):
   if path.name in ['supervisor.json','child.json']:
    item=json.loads(path.read_text())['identity'];result[item['pid']]=item
  return result
 def _leaf_only(self):
  require((self.cg.stat().st_dev,self.cg.stat().st_ino)==self.identity and self.cg.resolve()==self.cg,'cleanup cgroup identity changed')
  require(not any(p.is_dir()for p in self.cg.iterdir()),'unexpected cgroup descendant; no recursive kill')
 def cleanup(self):
  errors=[]
  # Retained creator pidfds remain valid even if optional /proc birth capture failed.
  for h in self.children:
   if h['process'].poll()is None:
    if h['pidfd']is None:errors.append('owned child lacks pidfd; retained without guessed-PID signal');continue
    try:signal.pidfd_send_signal(h['pidfd'],signal.SIGKILL)
    except ProcessLookupError:pass
    try:h['process'].wait(timeout=3)
    except subprocess.TimeoutExpired:errors.append('owned child wait unresolved')
   if h['pidfd']is not None:os.close(h['pidfd']);h['pidfd']=None
  try:self._leaf_only()
  except BaseException as error:return dict(verified=False,error=str(error),errors=errors,recursive_kill=False)
  deadline=time.monotonic()+3
  while numbers(self.cg/'cgroup.events')['populated']and time.monotonic()<deadline:time.sleep(.05)
  if numbers(self.cg/'cgroup.events')['populated']:
   self._leaf_only();known=self._recorded_identities()
   for pid in (self.cg/'cgroup.procs').read_text().split():
    pid=int(pid);require(pid in known,'unproven member; no guessed or recursive signal');expected=known[pid];current=birth(pid)
    allowed={str(HERE/'pilot.py').encode(),str(self.source/'execution.py').encode(),str(self.root/'fixture.py').encode()}
    require(any(arg in allowed for arg in Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0')),'member command family not owned')
    require(all(current[k]==expected[k]for k in ['pid','start_ticks','boot_id']),'member birth differs')
    fd=os.pidfd_open(pid)
    try:
     require(all(birth(pid)[k]==expected[k]for k in ['pid','start_ticks','boot_id']),'member identity raced')
     self._leaf_only();signal.pidfd_send_signal(fd,signal.SIGKILL)
    finally:os.close(fd)
  deadline=time.monotonic()+3
  while numbers(self.cg/'cgroup.events')['populated']and time.monotonic()<deadline:time.sleep(.05)
  self._leaf_only()
  for expected in self._recorded_identities().values():
   try:
    current=birth(expected['pid'])
    if all(current[k]==expected[k]for k in ['pid','start_ticks','boot_id'])and current['state']not in ('Z','X'):errors.append('recorded fixture still live; cgroup emptiness is insufficient')
   except (FileNotFoundError,ProcessLookupError):pass
   except PermissionError:errors.append('recorded fixture identity unobservable')
  return dict(verified=not errors and numbers(self.cg/'cgroup.events')['populated']==0,errors=errors,recursive_kill=False,cgroup=str(self.cg),counter_snapshot=dict(cpu=numbers(self.cg/'cpu.stat'),memory_peak=int((self.cg/'memory.peak').read_text()),io=(self.cg/'io.stat').read_text()))


def receive_grant(fd,request_path):
 require(type(fd)is int and fd>=0,'controller capability descriptor required')
 require(stat.S_ISSOCK(os.fstat(fd).st_mode),'controller capability must be a socket')
 channel=socket.socket(fileno=os.dup(fd))
 try:
  require(channel.family==socket.AF_UNIX and channel.type==socket.SOCK_STREAM,'local stream capability required')
  peer_pid,peer_uid,_=struct.unpack('3i',channel.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
  channel.settimeout(2);raw=b''
  while b'\n'not in raw:
   part=channel.recv(4096);require(part,'controller denied worker grant');raw+=part;require(len(raw)<=65536,'oversized controller grant')
  grant=json.loads(raw.decode());require(grant['schema']=='PilotWorkerGrantV2','invalid controller grant')
  require(peer_pid==os.getppid()==grant['parent']['pid']and peer_uid==os.getuid(),'controller peer differs from live parent')
  for expected,pid in [(grant['parent'],peer_pid),(grant['child'],os.getpid())]:
   current=birth(pid);require(all(current[k]==expected[k]for k in ['pid','start_ticks','boot_id']),'controller/worker birth identity differs')
  require(0<=time.monotonic()-grant['issued_monotonic']<=2,'controller grant expired')
  require(str(Path(request_path).resolve())==grant['request']['path']and fingerprint(request_path)==grant['request'],'controller request bytes differ')
  return grant
 finally:channel.close()

def checked_grant_file(item,path):
 require(item['path']==str(Path(path).resolve())and fingerprint(path)==item,'granted file changed or moved')
 return json.loads(Path(path).read_text())

def worker_preflight(request_path,grant):
 # Every operation here is read-only. No group join or queue import is permitted.
 request=checked_grant_file(grant['request'],request_path);root=Path(request['root']);cg=Path(request['cgroup']);source=Path(request['source'])
 require(root.is_absolute()and root==root.resolve()and root.name.startswith('queue-pilot-')and root.stat().st_mode&0o777==0o700,'foreign scratch root')
 require(Path(request_path).resolve().parent==root,'foreign request location')
 admission=checked_grant_file(grant['admission'],root/'admission.json');require(admission['schema']=='PilotAdmissionV2','old or unknown worker admission');cfg=admission['config'];selected=json.loads((HERE/'queue-pins.json').read_text())
 require(str(source)==selected['source']==cfg['source']and str(cg)==cfg['cgroup'],'foreign source/cgroup')
 require(root.parent==Path(cfg['scratch_parent'])and root.stat().st_dev==source.stat().st_dev,'scratch envelope differs')
 require(0<=admission['admitted_at_unix']-cfg['fresh_at_unix']<=60 and 0<=time.time()-admission['admitted_at_unix']<=125,'admission provenance expired')
 require(cfg['protected_roots']and cfg['protected_files']and 'protected_identities'in cfg,'owner handoff absent')
 for protected in [source,HERE]+[Path(x).resolve()for x in cfg['protected_roots']]:require(not root.is_relative_to(protected)and not protected.is_relative_to(root),'scratch overlaps protected tree')
 expected=selected['pins']+[fingerprint(HERE/n)for n in ['pilot.py','sensors.py','queue-pins.json']]
 require(cfg['harness_pins']==expected[-3:]and admission['source_pins']==expected,'reviewed source/harness set differs')
 for item in expected+cfg['protected_files']:require(fingerprint(item['path'])==item,'worker source/owner pin drift')
 require(fingerprint(Path(sys.executable).resolve())==admission['runtime'],'worker runtime differs')
 leaf_doc=checked_grant_file(grant['leaf_admission'],root/'leaf-admission.json');require(leaf_doc['schema']=='PilotLeafAdmissionV2','old leaf admission');leaf=leaf_doc['leaf']
 require(cg.is_absolute()and cg.resolve()==cg and [cg.stat().st_dev,cg.stat().st_ino]==request['cgroup_identity']==[leaf['device'],leaf['inode']],'worker cgroup identity differs')
 require(str(cgroup_mount(cg))==leaf['mount'],'worker cgroup filesystem differs')
 require(str(cg)==leaf['path']and not any(p.is_dir()for p in cg.iterdir()),'worker leaf topology differs')
 require((cg/'cpu.max').read_text().strip()==leaf['cpu_max']and int((cg/'memory.max').read_text())==leaf['memory_max']and int((cg/'pids.max').read_text())==leaf['pids_max'],'worker cgroup limits differ')
 require(numbers(cg/'memory.events')==leaf['admitted_memory_events']and not any(numbers(cg/'pids.events').values()),'worker observed new pressure/limit event')
 require(os.access(cg/'cgroup.procs',os.W_OK),'worker join not delegated')
 require(Path(grant['sensor']['path']).parent==root and Path(grant['sensor']['path']).name.endswith('-sensor.json'),'foreign resource receipt')
 sensor=checked_grant_file(grant['sensor'],grant['sensor']['path']);require(sensor['phase']=='before-'+request['operation']and sensor['mode']=='real_unmodified_queue','wrong controller resource gate')
 require(0<=time.monotonic()-sensor['sample']['monotonic']<=2,'resource gate stale')
 Policy('real_unmodified_queue',leaf['admitted_memory_events']).check(sensor['sample'],'worker-gate')
 observer=object.__new__(RealSensors);observer.config=cfg;require(not observer.owner_processes(),'worker owner guard changed')
 return request,root,cg

def worker(request_path,capability_fd=None):
 grant=receive_grant(capability_fd,request_path)
 request,root,cg=worker_preflight(request_path,grant)
 os.close(capability_fd);name=Path(request_path).name.removesuffix('-request.json')
 os.nice(max(0,15-os.getpriority(os.PRIO_PROCESS,0)));(cg/'cgroup.procs').write_text(str(os.getpid()))
 require(str(os.getpid())in(cg/'cgroup.procs').read_text().split(),'worker cgroup join unverified');journal(root,name+'-entry.json',dict(identity=birth(os.getpid()),cgroup=str(cg),grant=grant['schema']))
 sys.path.insert(0,request['source']);import experiment_queue as q
 require(Path(q.__file__).resolve()==Path(request['source'])/'experiment_queue.py','foreign queue import')
 ex=q.executor;queue=root/'queue';operation=request['operation'];args=request['args'];operation_started=time.monotonic()
 if operation=='control':value=dict(control='no queue operation',qualification='unassessed')
 elif operation=='initialize':value=q.initialize(queue,1)
 elif operation=='enqueue':
  n=args['n'];require(type(n)is int and 1<=n<=8,'pilot prefix out of bounds');job=f'job-{n:04d}';output=root/('output-'+job);spec=dict(schema='LocalExecutionSpecV1',cycle_id='invented-pilot-series',job_key=job,attempt_id=job,argv=[sys.executable,str(root/'fixture.py'),str(output)],executable=ex.executable_pin(sys.executable),cwd=str(root),environment={},input_pins=[ex.pin(root/'fixture.py')],output_directory=str(output),native_receipts=[dict(path='receipt.json',required_fields=dict(schema='TinyPilotV1',status='computed'))],timeout_seconds=2,grace_seconds=.05,previous_attempt=None,retry_reason='');plan=dict(schema='LocalExperimentQueuePlanV1',queue_id=job,jobs=[dict(id=job,execution_spec=spec,dependencies=[],outputs=[dict(role='receipt',path='receipt.json',minimum_bytes=1)],resume_prelaunch=None)]);value=q.enqueue(queue,plan,q.digest(plan))
 elif operation=='seal':value=q.seal(queue)
 else:
  require(operation in ['tick','status','finalize'],'unknown operation');value=getattr(q,operation)(queue)
 journal(root,name+'-result.json',dict(identity=birth(os.getpid()),value=value,operation_seconds=time.monotonic()-operation_started,mode='real_unmodified_queue'))

def execute(config_path):
 config=json.loads(Path(config_path).read_text());require(time.time()-config['fresh_at_unix']<=60 and config['fresh_at_unix']<=time.time(),'fresh resource admission manifest expired')
 require(config['protected_roots']and config['protected_files']and 'protected_identities'in config,'explicit owner identity/state guards required')
 require(len(config['protected_roots'])<=16 and len(config['protected_files'])<=16,'guard envelope too large')
 source=Path(config['source']).resolve();expected=json.loads((HERE/'queue-pins.json').read_text())
 require(str(source)==expected['source'],'source path differs from reviewed queue')
 harness_pins=[fingerprint(HERE/n)for n in ['pilot.py','sensors.py','queue-pins.json']];require(config['harness_pins']==harness_pins,'reviewed harness pins differ')
 pins=expected['pins']+harness_pins
 for item in pins:require(fingerprint(item['path'])==item,'source/harness drift')
 for protected in config['protected_roots']:
  owner=Path(protected).resolve();scratch_parent=Path(config['scratch_parent']).resolve();require(not scratch_parent.is_relative_to(owner)and not owner.is_relative_to(scratch_parent),'scratch overlaps protected owner root')
 os.nice(max(0,15-os.getpriority(os.PRIO_PROCESS,0)))
 root=fresh_scratch(config['scratch_parent'],source);journal(root,'admission.json',dict(schema='PilotAdmissionV2',config=config,source_pins=pins,harness_mode='real_unmodified_queue',admitted_at_unix=time.time(),runtime=fingerprint(Path(sys.executable).resolve()),scratch_device=root.stat().st_dev))
 try:
  with (root/'fixture.py').open('x')as f:f.write(FIXTURE);f.flush();os.fsync(f.fileno())
  sensor=RealSensors(config,root,pins);journal(root,'leaf-admission.json',dict(schema='PilotLeafAdmissionV2',leaf=sensor.leaf));driver=RealDriver(root,source,config['cgroup']);result=Pilot(root,sensor,driver).run()
 except BaseException as e:
  result=dict(status='not_admitted',error=str(e),scientific_cycles=0);journal(root,'bootstrap-terminal.json',result)
 print(json.dumps(dict(root=str(root),result=result)));return result

if __name__=='__main__':
 parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--worker');group.add_argument('--execute');parser.add_argument('--cap-fd',type=int);a=parser.parse_args()
 if a.worker:worker(a.worker,a.cap_fd)
 else:raise SystemExit(execute(a.execute)['status']!='pilot_completed')
