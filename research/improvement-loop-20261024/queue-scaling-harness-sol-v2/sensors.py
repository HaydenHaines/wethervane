"""Production read-only Linux sensors; scratch probe is the only sensor write."""
import hashlib,json,os,time
from pathlib import Path
class Uncertain(RuntimeError):pass

def fingerprint(path):
 p=Path(path);h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return dict(path=str(p.resolve()),sha256=h.hexdigest(),bytes=p.stat().st_size)
def numbers(path):return {k:int(v)for k,v in (line.split()for line in Path(path).read_text().splitlines())}
def pressure(path):
 return {parts[0]:{k:float(v)for k,v in (x.split('=')for x in parts[1:])}for parts in (line.split()for line in Path(path).read_text().splitlines())}
def limit(path):
 value=Path(path).read_text().strip();return None if value=='max'else int(value)
def birth(pid):
 p=Path('/proc')/str(pid);v=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=int(pid),start_ticks=int(v[19]),state=v[0],ppid=int(v[1]),utime=int(v[11]),stime=int(v[12]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
def proc_sample(pid):
 p=Path('/proc')/str(pid);a=birth(pid);mem={}
 for line in (p/'smaps_rollup').read_text().splitlines():
  if line.startswith(('Rss:','Pss:')):mem[line.split(':')[0].lower()+'_bytes']=int(line.split()[1])*1024
 if birth(pid)['start_ticks']!=a['start_ticks']:raise Uncertain('PID reused during sample')
 if set(mem)!= {'rss_bytes','pss_bytes'}:raise Uncertain('incomplete process memory observation')
 return dict(**a,**mem)
def usage(root):
 size=0;entries=0
 for p in Path(root).rglob('*'):
  if p.is_symlink():raise Uncertain('scratch symlink')
  entries+=1;size+=p.stat().st_blocks*512
 return size,entries

def cgroup_mount(path):
 p=Path(path);matches=[]
 if not p.is_absolute()or p!=p.resolve():raise Uncertain('canonical cgroup path required')
 for line in Path('/proc/self/mountinfo').read_text().splitlines():
  left,right=line.split(' - ',1)
  if right.split()[0]=='cgroup2':
   mount=Path(left.split()[4])
   if p.is_relative_to(mount)and p!=mount:matches.append(mount)
 if not matches:raise Uncertain('path is not a nonroot cgroup-v2 member')
 return max(matches,key=lambda m:len(m.parts))

def other_cpu_cores(parent_before,parent_now,leaf_before,leaf_now,dt):
 if dt<=0 or parent_now<parent_before or leaf_now<leaf_before:raise Uncertain('ancestor or leaf CPU counter reset')
 return max(0,((parent_now-parent_before)-(leaf_now-leaf_before))/(dt*1e6))

def validate_leaf(path):
 p=Path(path);resolved=p.resolve()
 if not p.is_absolute()or p!=resolved:raise Uncertain('canonical absolute cgroup leaf required')
 mounts=[]
 for line in Path('/proc/self/mountinfo').read_text().splitlines():
  left,right=line.split(' - ',1)
  if right.split()[0]=='cgroup2':mounts.append(Path(left.split()[4]))
 if not any(p.is_relative_to(m)and p!=m for m in mounts):raise Uncertain('not a cgroup-v2 delegated leaf')
 if any(x.is_dir()for x in p.iterdir()):raise Uncertain('cgroup leaf has descendant groups')
 for name in ['cgroup.procs']:
  if not os.access(p/name,os.W_OK):raise Uncertain('no leaf permission: '+name)
 required=['cpu.max','cpu.stat','cpu.pressure','memory.max','memory.current','memory.peak','memory.events','memory.pressure','pids.max','pids.current','pids.events','io.stat','io.pressure','cgroup.events']
 if not all((p/n).is_file()for n in required):raise Uncertain('missing leaf controller/accounting files')
 if (p/'cgroup.procs').read_text().strip()or numbers(p/'cgroup.events').get('populated')!=0:raise Uncertain('leaf not empty at admission')
 quota,period=(p/'cpu.max').read_text().split()
 if quota=='max'or not 0<int(quota)/int(period)<=1:raise Uncertain('finite <=1 CPU quota required')
 if not 0<(limit(p/'memory.max')or 0)<=256*1024**2:raise Uncertain('finite <=256 MiB memory limit required')
 if not 0<(limit(p/'pids.max')or 0)<=16:raise Uncertain('finite <=16 pid limit required')
 admitted_events=numbers(p/'memory.events')
 if numbers(p/'cpu.stat').get('usage_usec')!=0 or int((p/'memory.peak').read_text())!=0 or any(admitted_events.values())or any(numbers(p/'pids.events').values()):raise Uncertain('fresh zero-counter leaf required')
 return dict(admitted_memory_events=admitted_events,mount=str(max((m for m in mounts if p.is_relative_to(m)),key=lambda x:len(x.parts))),path=str(p),device=p.stat().st_dev,inode=p.stat().st_ino,quota_cores=int(quota)/int(period),cpu_max=(p/'cpu.max').read_text().strip(),memory_max=limit(p/'memory.max'),pids_max=limit(p/'pids.max'))

class RealSensors:
 mode='real_unmodified_queue'
 def __init__(self,config,scratch,source_pins):
  self.config=config;self.scratch=Path(scratch);self.cg=Path(config['cgroup']);self.leaf=validate_leaf(self.cg);self.admitted_events=dict(self.leaf['admitted_memory_events']);self.source_pins=source_pins;self.previous=None
  self.files=config['protected_files'];self.group_previous=None
  for item in self.files:
   if item['bytes']>1048576 or fingerprint(item['path'])!=item:raise Uncertain('owner guard handoff changed/too large')
  self.ancestors=[];node=self.cg.parent
  while str(node)!=self.leaf['mount']:
   self.ancestors.append(node);node=node.parent
 def owner_processes(self):
  found=[];roots=[str(Path(x).resolve()).encode()for x in self.config['protected_roots']]
  for path in Path('/proc').iterdir():
   if not path.name.isdigit():continue
   try:
    if path.stat().st_uid!=os.getuid():continue
    raw=(path/'cmdline').read_bytes();cwd=os.readlink(path/'cwd').encode()
    if any(root in raw or cwd==root or cwd.startswith(root+b'/')for root in roots):found.append(dict(**birth(path.name),command_sha256=hashlib.sha256(raw).hexdigest(),cwd=cwd.decode(errors='replace')))
   except (FileNotFoundError,ProcessLookupError):continue
   except PermissionError as e:raise Uncertain('incomplete same-user owner discovery')from e
  for expected in self.config['protected_identities']:
   try:current=birth(expected['pid'])
   except (FileNotFoundError,ProcessLookupError):continue
   if any(current[k]!=expected[k]for k in ['pid','start_ticks','boot_id']):raise Uncertain('protected process birth identity changed')
   if current['state']not in ['Z','X']:found.append(current)
  return found
 def sample(self):
  if (self.cg.stat().st_dev,self.cg.stat().st_ino)!=(self.leaf['device'],self.leaf['inode']):raise Uncertain('cgroup identity changed')
  if (self.cg/'cpu.max').read_text().strip()!=self.leaf['cpu_max']or limit(self.cg/'memory.max')!=self.leaf['memory_max']or limit(self.cg/'pids.max')!=self.leaf['pids_max']:raise Uncertain('cgroup envelope changed')
  for item in self.source_pins+self.files:
   if fingerprint(item['path'])!=item:raise Uncertain('source or protected owner-state change: '+item['path'])
  owners=self.owner_processes();mem={k:int(v.split()[0])*1024 for k,v in (x.split(':',1)for x in Path('/proc/meminfo').read_text().splitlines())}
  cpu=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));stamp=time.monotonic();total=sum(cpu[:8]);idle=cpu[3]+cpu[4];util=None
  if self.previous:
   dt=total-self.previous[0]
   if dt<=0:raise Uncertain('invalid CPU interval')
   util=1-(idle-self.previous[1])/dt
  self.previous=(total,idle)
  procs=[];vanished=[]
  for pid in (self.cg/'cgroup.procs').read_text().split():
   try:procs.append(proc_sample(pid))
   except (FileNotFoundError,ProcessLookupError):
    if pid in (self.cg/'cgroup.procs').read_text().split():raise Uncertain('member unobservable')
    vanished.append(int(pid))
   except PermissionError as e:raise Uncertain('member memory unreadable')from e
  probe=self.scratch/'storage-probe';started=time.monotonic()
  with probe.open('ab')as f:f.write(b'x');f.flush();os.fsync(f.fileno())
  fd=os.open(self.scratch,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
  latency=(time.monotonic()-started)*1000;disk=os.statvfs(self.scratch);allocated,entries=usage(self.scratch)
  parent_limit=limit(self.cg.parent/'memory.max');parent_used=int((self.cg.parent/'memory.current').read_text());parent_remaining=None if parent_limit is None else parent_limit-parent_used
  events=numbers(self.cg/'memory.events');group_cpu=numbers(self.cg/'cpu.stat');ancestors=[];finite_memory=[]
  for node in self.ancestors:
   q,period=(node/'cpu.max').read_text().split();capacity=len(os.sched_getaffinity(0))if q=='max'else min(len(os.sched_getaffinity(0)),int(q)/int(period));pcpu=numbers(node/'cpu.stat');mlimit=limit(node/'memory.max');remaining=None if mlimit is None else mlimit-int((node/'memory.current').read_text())
   if remaining is not None:finite_memory.append(remaining)
   other=None
   if self.group_previous:
    before=self.group_previous;dt=stamp-before['stamp']
    other=other_cpu_cores(before['ancestors'][str(node)],pcpu['usage_usec'],before['leaf'],group_cpu['usage_usec'],dt)
   ancestors.append(dict(path=str(node),capacity_cores=capacity,other_used_cores=other,cpu=pcpu,memory_remaining=remaining))
  self.group_previous=dict(stamp=stamp,leaf=group_cpu['usage_usec'],ancestors={a['path']:a['cpu']['usage_usec']for a in ancestors})
  parent_remaining=min(finite_memory)if finite_memory else None
  return dict(mode=self.mode,harness=proc_sample(os.getpid()),monotonic=stamp,cpu_util=util,cpu_count=len(os.sched_getaffinity(0)),host_cpu_count=os.cpu_count(),load1=os.getloadavg()[0],host_memory_available=mem['MemAvailable'],host_memory_total=mem['MemTotal'],cache=dict(Cached=mem['Cached'],Buffers=mem['Buffers'],policy='OS-managed, no drop_caches, no cold-cache claim'),host_pressure={k:pressure('/proc/pressure/'+k)for k in ['cpu','memory','io']},cgroup=dict(identity=self.leaf,cpu=group_cpu,quota_cores=self.leaf['quota_cores'],ancestors=ancestors,memory_current=int((self.cg/'memory.current').read_text()),memory_peak=int((self.cg/'memory.peak').read_text()),memory_events=events,parent_memory_remaining=parent_remaining,pressure={k:pressure(self.cg/(k+'.pressure'))for k in ['cpu','memory','io']},io_stat=(self.cg/'io.stat').read_text(),pids_current=int((self.cg/'pids.current').read_text()),pids_events=numbers(self.cg/'pids.events'),events=numbers(self.cg/'cgroup.events'),members=procs,vanished_between_samples=vanished,rss_sum=sum(v['rss_bytes']for v in procs),pss_sum=sum(v['pss_bytes']for v in procs)),owners=owners,protected_file_pins=self.files,free_bytes=disk.f_bavail*disk.f_frsize,free_inodes=disk.f_favail,probe_fsync_ms=latency,scratch_allocated=allocated,scratch_entries=entries,diskstats=Path('/proc/diskstats').read_text())
