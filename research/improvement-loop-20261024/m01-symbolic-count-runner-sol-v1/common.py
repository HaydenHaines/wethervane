"""Bounded metadata primitives; local cooperating-writer contract, not authentication service."""
import hashlib,json,os,stat,time
from pathlib import Path
MAX_FILE=1024**2;MAX_REPORTS=16*1024**2

def need(ok,message):
 if not ok:raise ValueError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def pin(path):
 path=Path(path);raw=read_regular(path);return dict(path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
def read_regular(path):
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 try:
  before=os.fstat(fd);need(stat.S_ISREG(before.st_mode)and before.st_size<=MAX_FILE,'bounded regular metadata file required')
  with os.fdopen(fd,'rb',closefd=False)as stream:raw=stream.read(MAX_FILE+1)
  after=os.fstat(fd);need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'metadata changed during read')
  need(len(raw)==before.st_size,'short metadata read');return raw
 finally:os.close(fd)
def checked(item):
 need(not str(item['path']).endswith(('.pkl','.pickle')),'pickle file access prohibited')
 raw=read_regular(item['path']);need(hashlib.sha256(raw).hexdigest()==item['sha256'],'pinned bytes changed')
 if 'bytes'in item:need(len(raw)==item['bytes'],'pinned size changed')
 return raw

class Writer:
 def __init__(self,directory,attempt,fail_fsync=False):self.directory=Path(directory);self.attempt=Path(attempt);self.fail_fsync=fail_fsync
 def text(self,name,text):
  need(Path(name).name==name and not name.endswith(('.pkl','.pickle')),'metadata names only')
  raw=text.encode();need(len(raw)<=MAX_FILE,'metadata per-file cap')
  size=sum(p.stat().st_size for p in self.attempt.rglob('*')if p.is_file());need(size+len(raw)<=MAX_REPORTS,'synchronous metadata aggregate cap includes temporary writes')
  temporary=self.directory/(name+'.pending');target=self.directory/name
  with temporary.open('xb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
  os.replace(temporary,target);fd=os.open(self.directory,os.O_RDONLY|os.O_DIRECTORY)
  try:
   if self.fail_fsync:raise OSError('invented post-rename directory fsync failure')
   os.fsync(fd)
  finally:os.close(fd)
  return pin(target)
 def json(self,name,value):return self.text(name,(canonical(value)+b'\n').decode())
