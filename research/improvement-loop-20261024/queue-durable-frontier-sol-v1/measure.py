"""Bounded SQL-selection fixture; NOT an authenticated executor-history benchmark."""
import json,resource,sqlite3,time,hashlib
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(15,20));resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,)*2);resource.setrlimit(resource.RLIMIT_FSIZE,(4*1024**2,)*2)
HERE=Path(__file__).resolve().parent;start=time.monotonic();results=[]
queries={'old_active':'SELECT r.attempt FROM reservations r LEFT JOIN releases l ON l.attempt=r.attempt WHERE l.attempt IS NULL','old_pending':'SELECT o.attempt FROM owners o LEFT JOIN reservations r ON r.attempt=o.attempt WHERE r.attempt IS NULL ORDER BY o.rowid','new_active':'SELECT attempt FROM active','new_ready':'SELECT attempt FROM pending WHERE remaining=0 AND blocked IS NULL ORDER BY seq LIMIT 1'}
source_paths=[HERE/'dispatch.py',HERE/'measure.py',HERE.parent/'queue-v2-incremental-assessment-sol-v1/dispatch.py'];pins={str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in source_paths}
for name,query in queries.items():assert query in (source_paths[2]if name.startswith('old_')else source_paths[0]).read_text()
for size in [0,256,4096]:
 db=sqlite3.connect(':memory:');db.executescript('''CREATE TABLE owners(attempt TEXT PRIMARY KEY,output TEXT UNIQUE,admission TEXT,job TEXT,UNIQUE(admission,job));CREATE TABLE reservations(attempt TEXT PRIMARY KEY,body TEXT,digest TEXT);CREATE TABLE releases(attempt TEXT PRIMARY KEY,body TEXT,digest TEXT);CREATE TABLE pending(seq INTEGER PRIMARY KEY AUTOINCREMENT,attempt TEXT UNIQUE,remaining INTEGER CHECK(remaining>=0),blocked TEXT);CREATE INDEX ready_frontier ON pending(seq,attempt) WHERE remaining=0 AND blocked IS NULL;CREATE TABLE active(slot INTEGER PRIMARY KEY CHECK(slot=1),attempt TEXT UNIQUE);''')
 db.executemany('INSERT INTO owners VALUES(?,?,?,?)',[(str(i),'/invented/'+str(i),'history',str(i))for i in range(size)]);db.executemany('INSERT INTO reservations VALUES(?,?,?)',[(str(i),'{}','invented')for i in range(size)]);db.executemany('INSERT INTO releases VALUES(?,?,?)',[(str(i),'{}','invented')for i in range(size)]);db.commit()
 for frontier in ['empty','one_ready_one_active']:
  if frontier!='empty':
   db.execute('INSERT INTO owners VALUES(?,?,?,?)',('ready','/invented/ready','next','ready'));db.execute('INSERT INTO pending(attempt,remaining) VALUES(?,0)',('ready',));db.execute('INSERT INTO active VALUES(1,?)',('invented-active',));db.commit()
  for name,query in queries.items():
   counter=[0]
   def step():counter[0]+=1;return 0
   plan=db.execute('EXPLAIN QUERY PLAN '+query).fetchall();db.set_progress_handler(step,1);before=time.perf_counter();rows=db.execute(query).fetchall();elapsed=time.perf_counter()-before;db.set_progress_handler(None,0)
   if name=='new_active':assert rows==([]if frontier=='empty'else[('invented-active',)])
   if name=='new_ready':assert rows==([]if frontier=='empty'else[('ready',)])
   results.append({'historical_fixture_rows':size,'frontier':frontier,'query':name,'sqlite_vm_instructions':counter[0],'seconds_including_per_instruction_python_hook':elapsed,'result_rows':len(rows),'query_plan':plan})
 db.close()
assert pins=={str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in source_paths}
result={'source_pins':pins,'schema':'DurableFrontierBoundedSQLSelectionV1','status':'passed','scope':'Synthetic SQL rows only, no executor result authority; no dispatcher tick called on these fixture databases. New/old exact query strings and relevant schema/indexes only.','rows':results,'seconds':time.monotonic()-start,'controller_maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'storage':'in-memory fixture databases, released after each size; output JSON only','limitations':['VM counts include SQLite implementation choices and Python hook overhead; timings are not throughput evidence.','Does not measure full executor/source-hash/dependency authentication, admission, physical disk I/O or thousands of real cycles.','History here is deliberately unauthenticated synthetic SQL; production adoption is not justified.']};(HERE/'measurements.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','seconds':result['seconds'],'vm':[{k:r[k]for k in ['historical_fixture_rows','frontier','query','sqlite_vm_instructions']}for r in results]}))
