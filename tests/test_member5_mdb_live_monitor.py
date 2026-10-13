"""No training, PID termination or real log writes. Test current-attempt telemetry."""
import copy
import os
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src import member5_mdb_live_monitor as m
from src import member5_mdb_runner as runner

ATTEMPT = 'attempt_synthetic_monitor_test'


def events(count, skips=(), attempt=ATTEMPT):
    out = []
    for i in range(count):
        epoch, batch = divmod(i, 48)
        stage = 1 if epoch < 3 else 2
        epoch = epoch + 1 if stage == 1 else epoch - 2
        skipped = i + 1 in skips
        out.append(dict(kind='batch',time='2026-10-14T00:00:00+00:00',attempt_id=attempt,event_seq=i+1,
                        seed=0,stage=stage,epoch=epoch,batch=batch+1,skipped=skipped,
                        loss_finite=True,applied_gradients_finite=True,weights_finite=True,
                        unscaled_gradients_finite=not skipped,loss_values=[1.0,0.5],
                        scale_before=32768.0,scale_after=32768.0))
    return out


class LiveMonitorTests(unittest.TestCase):
    def monitor(self): return m.AMPMonitor(ATTEMPT,48)
    def decision(self, count, skips=()): return self.monitor().consume(events(count,skips),'historical_snapshot')
    def setup_file(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup)
        folder=Path(t.name)/ATTEMPT;folder.mkdir();p=folder/'amp_events.jsonl';p.write_bytes(b'')
        state=self.monitor();return p,state,m.JSONLFollower(p,state)
    def write(self,p,rows,mode='w'):
        with p.open(mode,encoding='utf-8') as f:
            for e in rows:f.write(json.dumps(e)+'\n')
    def test_first_three_one_skip_does_not_stop(self):
        d=self.decision(3,[3]);self.assertEqual(d['action'],'CONTINUE');self.assertFalse(d['rate_gate_active'])
    def test_48_one_skip(self):
        d=self.decision(48,[3]);self.assertEqual(d['action'],'CONTINUE');self.assertAlmostEqual(d['skip_fraction'],1/48)
    def test_284_one_skip(self):
        d=self.decision(284,[3]);self.assertEqual(d['action'],'CONTINUE');self.assertAlmostEqual(d['skip_fraction'],1/284)
    def test_48_three_nonconsecutive_skips(self):
        d=self.decision(48,[3,20,40]);self.assertEqual(d['action'],'STOP');self.assertEqual(d['skip_fraction'],.0625)
    def test_three_consecutive_immediate_before_warmup(self):
        d=self.decision(3,[1,2,3]);self.assertEqual(d['action'],'STOP');self.assertIn('consecutive',d['reason'])
    def test_isolated_skips_below_threshold(self):self.assertEqual(self.decision(96,[3,51])['action'],'CONTINUE')
    def test_nonfinite_loss_immediate(self):
        es=events(1);es[0]['loss_finite']=False;self.assertEqual(self.monitor().consume(es)['action'],'STOP')
    def test_nonfinite_applied_gradient_immediate(self):
        es=events(1);es[0]['applied_gradients_finite']=False;self.assertEqual(self.monitor().consume(es)['action'],'STOP')
    def test_nonfinite_weight_immediate(self):
        es=events(1);es[0]['weights_finite']=False;self.assertEqual(self.monitor().consume(es)['action'],'STOP')
    def test_normal_loss_updates(self):
        d=self.decision(48);self.assertEqual(d['action'],'CONTINUE');self.assertEqual(d['optimizer_updates'],48)
    def test_history_snapshot_repeated_poll_not_counted_twice(self):
        p,s,f=self.setup_file();self.write(p,events(284,[3]));a=f.poll();b=f.poll()
        self.assertEqual(a['mode'],'historical_snapshot');self.assertEqual(b['new_batches'],0);self.assertEqual(b['batches'],284)
    def test_incremental_append_once(self):
        p,s,f=self.setup_file();es=events(48,[3]);self.write(p,es[:3]);self.assertFalse(f.poll()['rate_gate_active'])
        self.write(p,es[3:],'a');d=f.poll();self.assertEqual(d['new_batches'],45);self.assertEqual(d['batches'],48)
        self.assertEqual(f.poll()['batches'],48)
    def test_foreign_attempt_rejected(self):
        d=self.monitor().consume(events(1,attempt='attempt_other'));self.assertEqual(d['action'],'STOP');self.assertIn('attempt',d['reason'])
    def test_partial_json_waits_for_newline(self):
        p,s,f=self.setup_file();es=events(4,[3]);self.write(p,es[:3]);raw=json.dumps(es[3]).encode()
        with p.open('ab') as handle:handle.write(raw[:30])
        d=f.poll();self.assertEqual(d['batches'],3);self.assertEqual(d['action'],'CONTINUE');self.assertGreater(d['pending_bytes'],0)
        with p.open('ab') as handle:handle.write(raw[30:]+b'\n')
        self.assertEqual(f.poll()['batches'],4)
    def test_duplicate_batch_rejected(self):
        es=events(2);es[1]['batch']=1;self.assertEqual(self.monitor().consume(es)['action'],'STOP')
    def test_duplicate_event_seq_rejected(self):
        es=events(2);es[1]['event_seq']=1;self.assertEqual(self.monitor().consume(es)['action'],'STOP')
    def test_no_pid_termination_interface(self):
        # Monitor cannot select/terminate any process, even an unrelated PID.
        with patch('subprocess.run') as run,patch('subprocess.Popen') as popen,patch('os.kill') as kill:
            es=events(3,[1,2,3]);es[-1]['pid']=999999
            self.assertEqual(self.monitor().consume(es)['action'],'STOP')
            run.assert_not_called();popen.assert_not_called();kill.assert_not_called()
        self.assertFalse(hasattr(m,'stop_process'));self.assertFalse(hasattr(m.AMPMonitor,'terminate'))
    def test_synthetic_never_calls_taskkill(self):
        with patch('subprocess.run') as run:
            self.decision(48,[3,20,40]);run.assert_not_called()
    def test_real_284_read_only_replay(self):
        external=os.environ.get('MEMBER5_HISTORICAL_AMP_LOG')
        if not external:self.skipTest('Private historical evidence not supplied; set MEMBER5_HISTORICAL_AMP_LOG')
        old=Path(external)
        if not old.is_file():self.skipTest('Historical evidence unavailable on this machine')
        before=old.read_bytes();s=m.AMPMonitor(old.parent.name,48,legacy_read_only=True)
        f=m.JSONLFollower(old,s);d=f.poll(final=True)
        self.assertEqual(d['action'],'CONTINUE');self.assertEqual(d['batches'],284);self.assertEqual(d['skips'],1)
        self.assertEqual(d['optimizer_updates'],283);self.assertAlmostEqual(d['skip_fraction'],1/284)
        self.assertFalse(d['training_end_observed']);self.assertEqual(old.read_bytes(),before)
    def test_rate_gate_continues_after_warmup(self):
        s=self.monitor();es=events(55,[3,49,51]);self.assertEqual(s.consume(es[:48])['action'],'CONTINUE')
        self.assertEqual(s.consume(es[48:])['action'],'STOP')
    def test_exact_five_percent_allowed(self):self.assertEqual(self.decision(100,[3,23,43,63,83])['action'],'CONTINUE')
    def test_history_snapshot_rate_uses_current_tail(self):
        # Historical first epoch exceeds5%, but current284 has3/284; no prefix rate stop.
        self.assertEqual(self.decision(284,[3,20,40])['action'],'CONTINUE')
    def test_historical_serious_error_stays_fatal(self):
        es=events(284);es[2]['loss_finite']=False;self.assertEqual(self.monitor().consume(es,'historical_snapshot')['action'],'STOP')
    def test_live_stop_cannot_be_cleared(self):
        s=self.monitor();es=events(284,[3,20,40]);self.assertEqual(s.consume(es[:48])['action'],'STOP')
        self.assertEqual(s.consume(es[48:])['action'],'STOP')
    def test_corrupt_complete_line_stops(self):
        p,s,f=self.setup_file();p.write_text('{bad}\n');self.assertEqual(f.poll()['action'],'STOP')
    def test_incomplete_final_line_stops(self):
        p,s,f=self.setup_file();p.write_text('{');self.assertEqual(f.poll(final=True)['action'],'STOP')
    def test_truncated_log_stops(self):
        p,s,f=self.setup_file();self.write(p,events(1));f.poll();p.write_bytes(b'');self.assertEqual(f.poll()['action'],'STOP')
    def test_wrong_attempt_path_rejected(self):
        with self.assertRaises(m.MonitorError):m.JSONLFollower(Path('attempt_other/amp_events.jsonl'),self.monitor())
    def test_missing_batch_and_epoch_transition_rejected(self):
        for key,value in [('batch',3),('epoch',2),('stage',2),('seed',1)]:
            es=events(2);es[1][key]=value
            with self.subTest(key=key):self.assertEqual(self.monitor().consume(es)['action'],'STOP')
    def test_skipped_nonfinite_unscaled_is_warning(self):
        self.assertEqual(self.decision(48,[3])['action'],'CONTINUE')
    def test_nonfinite_unskipped_gradient_is_fatal(self):
        es=events(1);es[0]['unscaled_gradients_finite']=False;self.assertEqual(self.monitor().consume(es)['action'],'STOP')
    def test_terminal_failure_immediate(self):
        es=events(1);es.append(dict(kind='end',success=False,time=es[0]['time'],attempt_id=ATTEMPT,event_seq=2,error='CUDA out of memory'))
        self.assertEqual(self.monitor().consume(es)['action'],'STOP')
    def test_fake_success_before_all_seeds_rejected(self):
        es=events(48);es.append(dict(kind='end',success=True,time=es[0]['time'],attempt_id=ATTEMPT,event_seq=49))
        self.assertEqual(self.monitor().consume(es)['action'],'STOP')
    def test_numeric_nan_loss_stops(self):
        es=events(1);es[0]['loss_values']=[float('nan')];self.assertEqual(self.monitor().consume(es)['action'],'STOP')
    def test_boolean_string_rejected(self):
        es=events(1);es[0]['skipped']='false';self.assertEqual(self.monitor().consume(es)['action'],'STOP')
    def test_recovery_tool_gate_not_optional(self):
        with self.assertRaises(runner.Error):runner.verify_monitor_tools({})
    def test_changed_tool_hash_rejected(self):
        tools={name:runner.ref(runner.PROJECT/name) for name in ('src/member5_mdb_runner.py','src/member5_mdb_live_monitor.py','scripts/member5_mdb_formal_run.ps1')}
        tools['src/member5_mdb_live_monitor.py']['sha256']='0'*64
        with self.assertRaises(runner.Error):runner.verify_monitor_tools({'monitoring_tools':tools})
    def test_all_five_seeds_success_stream_and_runner_post_check(self):
        p,s,f=self.setup_file();rows=[]
        for seed in range(5):
            for event in events(192,[3] if seed==0 else []):
                event['seed']=seed;event['event_seq']=len(rows)+1;rows.append(event)
        rows.append(dict(kind='end',success=True,time=rows[-1]['time'],attempt_id=ATTEMPT,event_seq=len(rows)+1))
        self.write(p,rows);d=f.poll(final=True)
        self.assertEqual(d['action'],'CONTINUE');self.assertTrue(d['training_end_observed']);self.assertEqual(d['batches'],960)
        self.assertTrue(runner.monitoring_ok(p))
    def test_observer_emit_integration_without_training(self):
        # Execute the actual nested emit function extracted from Runner AST,
        # with in-memory logs and no torch/trainer/process involved.
        import ast,io,os
        source=ast.parse(Path(runner.__file__).read_text(encoding='utf-8-sig'))
        observer=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='observe_training')
        emit=next(n for n in ast.walk(observer) if isinstance(n,ast.FunctionDef) and n.name=='emit')
        emit=copy.deepcopy(emit);emit.body=[n for n in emit.body if not isinstance(n,ast.Nonlocal)]
        emit.body.insert(0,ast.Global(names=['event_seq']))
        log=io.StringIO();decisions=io.StringIO()
        # StringIO has no fileno; give the fixture a harmless descriptor value.
        log.fileno=lambda:0;decisions.fileno=lambda:0
        ns=dict(event_seq=0,log=log,decisions=decisions,attempt=Path(ATTEMPT),
                monitor=self.monitor(),utc_now=m.utc_now,json=json,os=os,require=runner.require)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[emit],type_ignores=[])),'runner_emit_fixture','exec'),ns)
        with patch.object(os,'fsync'),patch('subprocess.run') as kill:
            for event in events(3,[3]):
                event={k:v for k,v in event.items() if k not in ('attempt_id','event_seq','time')};ns['emit'](event)
            records=[json.loads(x) for x in decisions.getvalue().splitlines()]
            self.assertEqual(records[-1]['action'],'CONTINUE');self.assertEqual(records[-1]['batches'],3)
            kill.assert_not_called()


if __name__=='__main__':unittest.main()
