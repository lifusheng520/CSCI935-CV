"""Synthetic-only runner acceptance tests; never launches any trainer."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from src import member5_mdb_runner as r
from src import member5_mdb_results as m

class RunnerTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix='member5_synthetic_');self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  self.selected=m.read_json(r.SELECTED_CONFIG)
  self.args=dict(self.selected['args'],data_root=str(self.root/'formal/cohort/Dhan-Shomadhan'),weights_path=str(self.root/'weights.pth'),out_name='member5_mdb_formal_test')
  self.approved=dict(schema_version=1,status='APPROVED',model='MDB-Net',purpose='formal',approval_reference='SYNTHETIC TEST ONLY',approval_source='EXPLICIT_USER_INSTRUCTION',protocol=m.PROTOCOL,args=self.args,expected_seeds=[0,1,2,3,4])
  self.policy=dict(status='APPROVED',approval_reference='SYNTHETIC TEST ONLY',approval_source='EXPLICIT_USER_INSTRUCTION',mask_qc_scope='eligibility invariant under ok,review',test_exposure='historical exposure retained',near_duplicate_policy='NOT VERIFIED',qc_status='NO_EFFECT_ON_TRAINING_ELIGIBILITY_VERIFIED',amp_status='ACCEPTABLE_WITH_MONITORING',monitoring_required=True,implementation_sha256=m.file_hash(r.PROJECT/'src/make_masks.py'))
 def put(self,name,value):
  p=self.root/name;p.write_text(json.dumps(value),encoding='utf-8');return p
 def approved_load(self):
  with patch.object(r,'verify_code',return_value=True):return r.load_approved(self.put('approved.json',self.approved))
 def test_valid_user_approval(self):self.assertEqual(self.approved_load()[1]['seeds'],[0,1,2,3,4])
 def test_missing_weights(self):self.assertEqual(r.weights_check(self.root/'missing')['status'],'MISSING WEIGHTS')
 def test_weight_hash_wrong(self):
  p=self.root/'weights';p.write_bytes(b'SYNTHETIC')
  with self.assertRaises(m.VerificationError):r.weights_check(p,'0'*64,r.WEIGHT_VERSION)
 def test_official_full_hash_not_prefix(self):
  p=self.root/'weights';p.write_bytes(b'SYNTHETIC')
  with patch.object(m,'file_hash',return_value='f37072fd'+'0'*56),self.assertRaises(m.VerificationError):r.weights_check(p,'f37072fd'+'0'*56,r.WEIGHT_VERSION)
 def test_manifest_wrong(self):
  with self.assertRaises(m.VerificationError):r.compare_manifest([dict(image='a',sha256='a')],[dict(image='a',sha256='b')])
 def test_manifest_duplicate(self):
  with self.assertRaises(m.VerificationError):r.compare_manifest([dict(image='a'),dict(image='a')],[dict(image='a')])
 def test_missing_index(self):
  with patch.object(r,'validate_qc_policy'),self.assertRaises(m.VerificationError):r.preprocessing_records(self.root,[],self.args,self.root/'policy')
 def test_four_seeds(self):
  self.approved['expected_seeds']=self.args['seeds']=[0,1,2,3]
  with self.assertRaises(m.VerificationError):self.approved_load()
 def test_duplicate_seeds(self):
  self.approved['expected_seeds']=self.args['seeds']=[0,1,2,3,3]
  with self.assertRaises(m.VerificationError):self.approved_load()
 def test_wrong_five_seeds(self):
  self.approved['expected_seeds']=self.args['seeds']=[1,2,3,4,5]
  with self.assertRaises(m.VerificationError):self.approved_load()
 def test_unapproved(self):
  self.approved['status']='DRAFT'
  with self.assertRaises(m.VerificationError):self.approved_load()
 def test_changed_science(self):
  self.args['batch_size']=8
  with self.assertRaises(m.VerificationError):self.approved_load()
 def test_synthetic_cannot_authorize(self):
  self.approved['synthetic']=True
  with self.assertRaises(m.VerificationError):self.approved_load()
 def test_no_fake_signatures(self):
  self.assertEqual(r.validate_qc_policy(self.put('policy.json',self.policy))['approval_source'],'EXPLICIT_USER_INSTRUCTION')
 def test_qc_gate(self):
  self.policy['qc_status']='NOT_VERIFIED'
  with self.assertRaises(m.VerificationError):r.validate_qc_policy(self.put('policy.json',self.policy))
 def test_amp_gate(self):
  self.policy['amp_status']='BLOCKED_PENDING_MODEL_FIX'
  with self.assertRaises(m.VerificationError):r.validate_qc_policy(self.put('policy.json',self.policy))
 def test_monitor_gate(self):
  self.policy['monitoring_required']=False
  with self.assertRaises(m.VerificationError):r.validate_qc_policy(self.put('policy.json',self.policy))
 def test_execute_does_not_bypass_block(self):
  with patch.object(r,'run_preflight',return_value={'status':'BLOCKED'}),patch.object(r,'execute_plan') as launch:
   self.assertEqual(r.run_request(execute=True)['status'],'BLOCKED');launch.assert_not_called()
 def test_default_dry_run_never_launches(self):
  with patch.object(r,'run_preflight',return_value={'status':'DRY RUN READY'}),patch.object(r,'execute_plan') as launch,patch.object(r.subprocess,'Popen') as process:
   self.assertEqual(r.run_request()['status'],'DRY RUN READY');launch.assert_not_called();process.assert_not_called()
 def test_evidence_never_overwrites(self):
  p=self.root/'log.json';r.write_new(p,{'old':1})
  with self.assertRaises(FileExistsError):r.write_new(p,{'new':2})
  self.assertEqual(m.read_json(p),{'old':1})
 def test_console_never_overwrites(self):
  p=self.root/'console.log';p.write_text('old')
  with patch.object(r.subprocess,'Popen') as proc,self.assertRaises(FileExistsError):r.capture_process([],p)
  proc.assert_not_called();self.assertEqual(p.read_text(),'old')
 def test_pilot_formal_rejected(self):
  p=self.root/'pilot/cohort/Dhan-Shomadhan';p.mkdir(parents=True)
  with self.assertRaises(m.VerificationError):r.isolated_identity(p,self.root,self.root/'original','formal')
 def test_other_member_root_rejected(self):
  p=self.root/'Member4/Dhan-Shomadhan';p.mkdir(parents=True)
  with self.assertRaises(m.VerificationError):r.isolated_identity(p,self.root/'workspace',self.root/'original','formal')
 def test_existing_cohort_rejected(self):
  p=self.root/'workspace/formal/member5_test';p.mkdir(parents=True)
  with self.assertRaises(m.VerificationError):r.prepare_plan(self.root/'original',self.root/'workspace','formal','member5_test')
 def test_existing_training_output_rejected(self):
  data=Path(self.args['data_root']);output=data/'_interim/runs'/self.args['out_name'];output.mkdir(parents=True)
  with patch.object(r,'load_approved',return_value=(self.approved,self.args)),patch.object(r,'validate_qc_policy'),patch.object(r,'environment_info',return_value={'cuda_available':True}),patch.object(r,'weights_check',return_value={'status':'WEIGHT FILE VERIFIED'}),patch.object(r,'isolated_identity',return_value=(data.parent,[])):
   plan=r.run_preflight('approval','policy',self.root,self.root/'original')
  self.assertEqual(plan['status'],'BLOCKED');self.assertTrue(any('OUTPUT EXISTS' in e for e in plan['blockers']))
 def test_internal_observer_cannot_bypass(self):
  with self.assertRaises(m.VerificationError):r.observe_training(self.root)
 def test_windows_path_spaces_shell_safe(self):
  self.args['weights_path']=str(self.root/'space with & shell'/'weights.pth')
  cmd=r.build_command(self.args)
  self.assertIsInstance(cmd,list);self.assertEqual(cmd[cmd.index('--weights_path')+1],self.args['weights_path'])
 def test_path_traversal(self):
  with self.assertRaises(m.VerificationError):r.safe_path(self.root/'..'/'escape')
 def test_windows_reserved_names(self):
  for name in ('CON','NUL','LPT1','bad/name','..'):
   with self.subTest(name=name),self.assertRaises(m.VerificationError):r.safe_name(name)
 def manifest(self):return [dict(image=f'{bg}/{c}/{i}.jpg',label=c,background=bg) for bg in ('white','field') for c in m.CLASSES for i in range(40)]
 def test_split_replay(self):
  rows=self.manifest();a=r.split_evidence(rows,self.args);b=r.split_evidence(rows,self.args);self.assertEqual(a,b)
  for s in a.values():
   self.assertFalse(set(s['train'])&set(s['val']));self.assertFalse(set(s['train'])&set(s['test']));self.assertEqual(len(set(s['train']+s['val']+s['test'])),400)
 def test_swap_train_only(self):
  for s in r.split_evidence(self.manifest(),self.args).values():
   self.assertTrue(set(s['background_swap_images'])<=set(s['train']));self.assertFalse(set(s['background_swap_images'])&set(s['test']+s['val']))
 def events(self,skips):
  p=self.root/'amp.jsonl';p.write_text('\n'.join(json.dumps(dict(kind='batch',skipped=k,loss_finite=True,applied_gradients_finite=True,weights_finite=True)) for k in skips)+'\n'+json.dumps(dict(kind='end',success=True)));return p
 def test_single_skip_acceptable(self):self.assertTrue(r.monitoring_ok(self.events([True]+[False]*95)))
 def test_repeated_amp_requires_review(self):
  with self.assertRaises(m.VerificationError):r.monitoring_ok(self.events([True]*3+[False]*93))
 def test_nonfinite_weight_rejected(self):
  p=self.events([False]*10);text=p.read_text().replace('"weights_finite": true','"weights_finite": false',1);p.write_text(text)
  with self.assertRaises(m.VerificationError):r.monitoring_ok(p)
 def test_incomplete_telemetry_rejected(self):
  p=self.root/'empty';p.write_text('')
  with self.assertRaises(m.VerificationError):r.monitoring_ok(p)
 def test_schema_v1_consumed_by_existing_statistics(self):
  # Separate synthetic acceptance of the existing contract; never production approval.
  from test_member5_mdb_results import make_fixture,TEST_SEEDS
  root,approved,evidence=make_fixture(self.root,full=True)
  out=m.audit_experiment(root,TEST_SEEDS,approved,evidence)
  self.assertEqual(out['status'],'COMPLETE_VERIFIED')
 def test_full_cli(self):
  cmd=r.build_command(self.args)
  for key in m.ARG_KEYS:
   if key not in m.BOOL_KEYS and self.args[key] is not None:self.assertIn('--'+key,cmd)
  self.assertNotIn('--from_scratch',cmd);self.assertIn('--save_models',cmd)

if __name__=='__main__':unittest.main()
