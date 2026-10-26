"""Invented entry fixture only; never imports actual scientific model."""
import json,sys,time
import entry
entry.read_contract=lambda:{'caps':{'cpu_soft_seconds':2,'cpu_hard_seconds':3,'address_space_bytes':256*1024**2,'file_bytes':1024**2,'wall_seconds':2}}
a,c,receipt=entry.accept('invented',sys.argv[1]);print(json.dumps(receipt),flush=True)
if len(sys.argv)>2:time.sleep(10)
