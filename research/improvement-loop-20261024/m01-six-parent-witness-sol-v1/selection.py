"""Frozen prior externally captured acknowledgments; no new phase completion authority."""
import json
from pathlib import Path
from common import *
def authenticate_predecessors(c):
 for phase,item in c['predecessor_acknowledgments'].items():
  path=Path(item['caller_return']);need(pin(path)==item['caller_return_pin'],'frozen caller acknowledgment changed');outer=json.loads(path.read_text());tool=outer['write_stdin_return'];need(tool['exit_code']==0,'prior controller tool did not complete');ack=json.loads(tool['output']);need(ack['arm']==phase and ack['status']=='completed','prior acknowledgment phase mismatch');root=Path(item['phase_directory']);need(pin(root/'controller-success.json')==ack['controller_success'],'prior retained controller acknowledgment mismatch');r=json.loads((root/'controller-success.json').read_text());need(pin(root/'result.json')==r['result']and pin(root/'terminal.json')==r['terminal'],'prior result/terminal drift');terminal=json.loads((root/'terminal.json').read_text());need(terminal['returncode']==0 and terminal['reason']is None,'prior phase not cleanly reaped')
  for role,expected in r['comparison_artifacts'].items():need(pin(Path(item['comparison_paths'][role]))==expected,'prior comparison artifact drift')
 return True
