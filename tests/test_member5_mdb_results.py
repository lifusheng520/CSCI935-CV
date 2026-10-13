"""Synthetic-only MDB fixtures in TemporaryDirectory; no real experiments/models.

The arbitrary seeds below are TEST DATA, never approved formal experiment seeds.
No fixture/output is written to member5_outputs or the real dataset.
"""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from src import member5_mdb_results as m

TEST_SEEDS=[11,23,37,51,83]

def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=1),encoding='utf-8')

def reference(path):
    return dict(path=str(path.resolve()),sha256=m.file_hash(path))

def independent_metrics(rows):
    out={}
    for sc,background in (('a_white','white'),('b_field','field'),('c_all',None)):
        rr=[r for r in rows if background is None or r['background']==background]
        y=[m.CLASSES.index(r['true']) for r in rr]; p=[m.CLASSES.index(r['predicted']) for r in rr]
        out[sc]=dict(n=len(rr),acc=float(accuracy_score(y,p)),macro_f1=float(f1_score(y,p,labels=list(range(5)),average='macro',zero_division=0)),
            per_class_f1=f1_score(y,p,labels=list(range(5)),average=None,zero_division=0).tolist(),cm=confusion_matrix(y,p,labels=list(range(5))).tolist())
    return out

def make_fixture(parent,full=False):
    root=parent/'input'; root.mkdir(); data_root=parent/'raw'
    samples=[]
    for bg,count in (('white',20),('field',40)):
        for c in m.CLASSES:
            for i in range(count):
                samples.append(dict(image=f"{bg.title()} Background/{c}/{i:03}.jpg",label=c,background=bg))
    samples.sort(key=lambda r:Path(r['image']))
    strata=np.array([f"{m.CLASSES.index(r['label'])}_{0 if r['background']=='white' else 1}" for r in samples])
    args=dict(img_size=256,cache_size=448,batch_size=16,head_epochs=1,epochs=5,lr_head=.001,
        lr_backbone=.0002,weight_decay=.01,label_smoothing=.1,dropout=.2,patience=1,lambda_seg=1.,pool='mask',
        mil_topk_frac=.2,lambda_supcon=0.,supcon_temp=.1,bg_swap=.5,tf_prob=.5,field_train='cls',mask_status='ok,review',
        class_weights=False,from_scratch=True,allow_missing=False,data_root=str(data_root),seeds=TEST_SEEDS,
        exclude_file=None,weights_path=None,workers=0,save_models=True,out_name='fixture_cohort')
    write_json(root/'args.json',{k:str(v) for k,v in args.items()})
    runs=[]; splits={}
    for run_i,seed in enumerate(TEST_SEEDS):
        idx=np.arange(len(samples)); tv,te=train_test_split(idx,test_size=.15,stratify=strata,random_state=seed)
        tr,va=train_test_split(tv,test_size=.15/.85,stratify=strata[tv],random_state=seed)
        tr,va,te=map(np.sort,(tr,va,te)); rows=[]
        for j,i in enumerate(te):
            r=samples[i]; y=m.CLASSES.index(r['label'])
            p=y if r['background']=='white' or j%(run_i+2)==0 else (y+1)%5
            probs=[.05]*5; probs[p]=.8
            rows.append(dict(image=r['image'],background=r['background'],true=m.CLASSES[y],predicted=m.CLASSES[p],correct=str(int(y==p)),confidence='.800',**{'p_'+c:f'{probs[k]:.3f}' for k,c in enumerate(m.CLASSES)}))
        history=[]
        for stage,values in ((1,[.9]),(2,[.5,.6,.6])):
            for ep,f1 in enumerate(values,1):
                history.append(dict(stage=stage,epoch=ep,loss=1.,cls=1.,seg=0.,supcon=0.,val_f1=f1,val_acc=.7,val_dice=float('nan')))
        metrics=independent_metrics(rows)
        result=dict(seed=seed,best_val_macro_f1=.6,n_train=len(tr),n_train_masked=0,history=history,scenarios=metrics,seg=dict(n=0,dice=None,iou=None))
        write_json(root/f'seed{seed}_result.json',result)
        (root/f'seed{seed}_predictions.csv').write_text(m.csv_text(rows),encoding='utf-8')
        runs.append(result)
        splits[seed]=dict(train=[samples[i]['image'] for i in tr],val=[samples[i]['image'] for i in va],test=[samples[i]['image'] for i in te],
            training_images=[samples[i]['image'] for i in tr],background_swap_images=[samples[i]['image'] for i in tr if samples[i]['background']=='field'])
    summary={}
    for sc in m.SCENARIOS:
        ms=[r['scenarios'][sc] for r in runs]; pc=np.array([r['per_class_f1'] for r in ms])
        summary[sc]=dict(runs=5,acc_mean=float(np.mean([r['acc'] for r in ms])),acc_std=float(np.std([r['acc'] for r in ms],ddof=1)),
            macro_f1_mean=float(np.mean([r['macro_f1'] for r in ms])),macro_f1_std=float(np.std([r['macro_f1'] for r in ms],ddof=1)),
            per_class_f1_mean=dict(zip(m.CLASSES,pc.mean(0).round(4).tolist())),per_class_f1_std=dict(zip(m.CLASSES,pc.std(0,ddof=1).tolist())),cm_sum=np.sum([r['cm'] for r in ms],axis=0).tolist())
    write_json(root/'summary.json',summary)
    if not full:
        return root,None,None
    # Full runner-contract fixture: fake bytes, NOT real images/training/checkpoints.
    evidence_dir=parent/'records'; evidence_dir.mkdir(); cache_rows=[]
    for r in samples:
        p=data_root/r['image']; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(('SYNTHETIC FILE '+r['image']).encode())
        r['sha256']=m.file_hash(p)
        cached=data_root/'_interim/cache_448'/r['image']; cached.parent.mkdir(parents=True,exist_ok=True); cached.write_bytes(('SYNTHETIC CACHE '+r['image']).encode())
        cache_rows.append(dict(image=r['image'],cache=reference(cached),mask=None,status='none'))
    manifest=evidence_dir/'dataset.json'; write_json(manifest,samples)
    cache=evidence_dir/'cache.json'; write_json(cache,cache_rows)
    qc_dir=data_root/'_interim/qc'; qc_dir.mkdir(parents=True)
    mask=qc_dir/'mask_index.csv'; mask.write_text('image,cache_image,mask,class,status\n',encoding='utf-8')
    qc=qc_dir/'qc_report.csv'; qc.write_text('image,cls,status\n',encoding='utf-8')
    exclude=qc_dir/'exclude.txt'; exclude.write_text('',encoding='utf-8')
    policy=evidence_dir/'policy.json'; write_json(policy,dict(status='APPROVED',approval_reference='SYNTHETIC TEST ONLY',mask_qc_scope='fixed test policy',test_exposure='test fixture',near_duplicate_policy='test fixture'))
    prep={k:reference(v) for k,v in dict(cache_manifest=cache,mask_index=mask,qc_report=qc,exclude_file=exclude,policy=policy).items()}
    code={n:reference(Path(m.__file__).parent/n) for n in ('train_multitask.py','train_resnet18.py','make_masks.py','seg_leaf.py')}
    records=[]
    for seed in TEST_SEEDS:
        cfg=evidence_dir/f'config{seed}.json'; write_json(cfg,dict(seed=seed,model='MDB-Net',purpose='formal',protocol=m.PROTOCOL,args=args))
        split=evidence_dir/f'split{seed}.json'; write_json(split,splits[seed])
        log=evidence_dir/f'log{seed}.txt'; log.write_text('SYNTHETIC TEST LOG, NO TRAINING',encoding='utf-8')
        env=evidence_dir/f'env{seed}.json'; write_json(env,{k:'test-only' for k in ('python','torch','torchvision','numpy','scikit-learn','device')})
        cp=root/f'seed{seed}_model.pt'; cp.write_bytes(b'SYNTHETIC CHECKPOINT, NOT A MODEL')
        records.append(dict(seed=seed,model='MDB-Net',purpose='formal',started_at='2026-10-13T00:00:00+00:00',completed_at='2026-10-13T01:00:00+00:00',
            config=reference(cfg),split=reference(split),log=reference(log),environment=reference(env),result=reference(root/f'seed{seed}_result.json'),predictions=reference(root/f'seed{seed}_predictions.csv'),
            checkpoint={**reference(cp),'stage':2,'epoch':2,'selection':m.PROTOCOL['selection']}))
    ev=evidence_dir/'evidence.json'; write_json(ev,dict(schema_version=1,model='MDB-Net',purpose='formal',dataset_manifest=reference(manifest),source_code=code,preprocessing=prep,runs=records))
    approval=evidence_dir/'approval.json'; write_json(approval,dict(schema_version=1,status='APPROVED',model='MDB-Net',purpose='formal',approval_reference='SYNTHETIC TEST ONLY',expected_seeds=TEST_SEEDS,
        protocol=m.PROTOCOL,args=args,dataset_manifest_sha256=m.file_hash(manifest),preprocessing_sha256={k:v['sha256'] for k,v in prep.items()}))
    return root,approval,ev

class MDBTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='member5_mdb_SYNTHETIC_TEST_'); self.addCleanup(self.temp.cleanup)
        self.parent=Path(self.temp.name); self.root,_,_=make_fixture(self.parent)
    def audit(self,**kwargs):
        return m.audit_experiment(self.root,TEST_SEEDS,**kwargs)
    def edit_json(self,name,func):
        p=self.root/name; obj=m.read_json(p); func(obj); write_json(p,obj)
    def edit_csv(self,func,seed=11):
        p=self.root/f'seed{seed}_predictions.csv'
        with p.open(newline='',encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        func(rows); p.write_text(m.csv_text(rows),encoding='utf-8')
    def invalid(self,needle=None,**kwargs):
        a=self.audit(**kwargs); self.assertEqual(a['status'],'INVALID',a)
        self.assertEqual(a['summary'],{})
        if needle:
            self.assertIn(needle,' '.join(a['errors']))
        return a
    def test_complete_five_seed_parse(self):
        a=self.audit(); self.assertEqual(a['validation'],'VERIFIED',a['errors']); self.assertEqual(len(a['runs']),5)
    def test_white_field_mixed_coverage(self):
        r=self.audit()['runs'][0]['metrics']; self.assertEqual(r['a_white']['n'],15); self.assertEqual(r['b_field']['n'],30); self.assertEqual(r['c_all']['n'],45)
    def test_accuracy_independent(self):
        a=self.audit(); r=a['runs'][0]; expected=sum(x['correct'] for x in r['rows'])/45
        self.assertAlmostEqual(r['metrics']['c_all']['acc'],expected)
    def test_macro_f1_fixed_five(self):
        r=self.audit()['runs'][0]; y=[x['true'] for x in r['rows']]; p=[x['predicted'] for x in r['rows']]
        self.assertAlmostEqual(r['metrics']['c_all']['macro_f1'],f1_score(y,p,labels=list(range(5)),average='macro',zero_division=0))
    def test_fixed_class_order(self):
        self.assertEqual(m.CLASSES,('brown_spot','leaf_scald','rice_blast','rice_tungro','sheath_blight'))
        cm=self.audit()['runs'][0]['metrics']['a_white']['cm']; self.assertEqual(cm,np.diag([3]*5).tolist())
    def test_confusion_matrix_recomputed(self):
        r=self.audit()['runs'][0]; expected=confusion_matrix([x['true'] for x in r['rows']],[x['predicted'] for x in r['rows']],labels=list(range(5))).tolist()
        self.assertEqual(r['metrics']['c_all']['cm'],expected)
    def test_sample_sd_ddof_one(self):
        a=self.audit(); v=[r['metrics']['c_all']['acc'] for r in a['runs']]
        self.assertGreater(np.std(v,ddof=1),0); self.assertAlmostEqual(a['summary']['c_all']['acc_std'],np.std(v,ddof=1))
        self.assertNotAlmostEqual(a['summary']['c_all']['acc_std'],np.std(v,ddof=0))
    def test_missing_seed(self):
        (self.root/'seed83_result.json').unlink(); self.invalid('missing/extra')
    def test_duplicate_expected_seed(self):
        a=m.audit_experiment(self.root,[11,23,37,51,51]); self.assertEqual(a['status'],'INVALID'); self.assertIn('Duplicate',' '.join(a['errors']))
    def test_duplicate_result_seed(self):
        self.edit_json('seed23_result.json',lambda x:x.update(seed=11)); self.invalid('identity')
    def test_unexpected_seed(self):
        (self.root/'seed99_result.json').write_text('{}',encoding='utf-8'); self.invalid('extra')
    def test_missing_predictions(self):
        (self.root/'seed11_predictions.csv').unlink(); self.invalid('missing')
    def test_missing_summary(self):
        (self.root/'summary.json').unlink(); self.invalid('unreadable')
    def test_json_invalid(self):
        (self.root/'seed11_result.json').write_text('{bad',encoding='utf-8'); self.invalid('invalid JSON')
    def test_duplicate_json_key(self):
        (self.root/'seed11_result.json').write_text('{"seed":11,"seed":23}',encoding='utf-8'); self.invalid('duplicate key')
    def test_bad_true_label(self):
        self.edit_csv(lambda r:r[0].update(true='healthy')); self.invalid('label')
    def test_bad_predicted_label(self):
        self.edit_csv(lambda r:r[0].update(predicted='MALG')); self.invalid('label')
    def test_bad_background(self):
        self.edit_csv(lambda r:r[0].update(background='White')); self.invalid('background')
    def test_duplicate_path(self):
        self.edit_csv(lambda r:r[1].update(image=r[0]['image'])); self.invalid('duplicate image')
    def test_windows_case_alias_duplicate(self):
        self.edit_csv(lambda r:r[1].update(image=r[0]['image'].upper())); self.invalid('duplicate image')
    def test_metric_mismatch(self):
        self.edit_json('seed11_result.json',lambda r:r['scenarios']['a_white'].update(acc=.4)); self.invalid('recomputed')
    def test_confusion_mismatch(self):
        self.edit_json('seed11_result.json',lambda r:r['scenarios']['a_white']['cm'][0].__setitem__(0,999)); self.invalid('confusion matrix mismatch')
    def test_mixed_not_unweighted_mean(self):
        r=self.audit()['runs'][0]['metrics']; mean=(r['a_white']['acc']+r['b_field']['acc'])/2
        self.assertNotAlmostEqual(r['c_all']['acc'],mean)
        self.edit_json('seed11_result.json',lambda x:x['scenarios']['c_all'].update(acc=mean)); self.invalid('recomputed')
    def test_pilot_args_rejected(self):
        self.edit_json('args.json',lambda x:x.update(out_name='member5_pilot_001')); self.invalid('Pilot')
    def test_pilot_result_rejected(self):
        self.edit_json('seed11_result.json',lambda x:x.update(purpose='pilot')); self.invalid('nonformal')
    def test_rounded_probability_sum_tolerance(self):
        self.edit_csv(lambda r:r[0].update(**{'p_'+m.CLASSES[0]:str(float(r[0]['p_'+m.CLASSES[0]])+.001)}))
        self.assertEqual(self.audit()['validation'],'VERIFIED')
    def test_rounded_probability_tie_not_error(self):
        def change(rows):
            r=rows[0]; p=m.CLASSES.index(r['predicted']); q=(p+1)%5
            r.update({'p_'+c:('0.350' if i in (p,q) else '0.100') for i,c in enumerate(m.CLASSES)}); r['confidence']='.350'
        self.edit_csv(change); a=self.audit(); self.assertEqual(a['validation'],'VERIFIED'); self.assertTrue(any('rounded/near-tied' in x for x in a['warnings']))
    def test_probability_contradiction(self):
        def change(rows):
            r=rows[0]; p=m.CLASSES.index(r['predicted']); q=(p+1)%5
            r.update({'p_'+c:('0.800' if i==q else '0.050') for i,c in enumerate(m.CLASSES)})
        self.edit_csv(change); self.invalid('contradicts')
    def test_probability_nan(self):
        self.edit_csv(lambda r:r[0].update(p_brown_spot='nan')); self.invalid('finite')
    def test_probability_out_of_range(self):
        self.edit_csv(lambda r:r[0].update(p_brown_spot='1.2')); self.invalid('above')
    def test_probability_bad_sum(self):
        self.edit_csv(lambda r:r[0].update(**{'p_'+c:'0.3' for c in m.CLASSES})); self.invalid('sum')
    def test_correct_flag(self):
        self.edit_csv(lambda r:r[0].update(correct=str(1-int(r[0]['correct'])))); self.invalid('correct flag')
    def test_windows_posix_paths(self):
        self.edit_csv(lambda r:[x.update(image=x['image'].replace('/','\\')) for x in r]); a=self.audit()
        self.assertEqual(a['validation'],'VERIFIED'); self.assertTrue(all('\\' not in x['image'] for x in a['runs'][0]['rows']))
    def test_path_traversal(self):
        self.edit_csv(lambda r:r[0].update(image='../raw.jpg')); self.invalid('unsafe')
    def test_absolute_path(self):
        self.edit_csv(lambda r:r[0].update(image='C:\\raw\\a.jpg')); self.invalid('unsafe')
    def test_missing_evidence_not_complete(self):
        a=self.audit(); self.assertEqual(a['status'],'PARTIALLY VERIFIED'); self.assertIn('CHECKPOINT IDENTITY NOT VERIFIED',a['limitations'])
    def test_malg_layout_strict_isolation(self):
        (self.root/'seed_42').mkdir(); self.invalid('legacy MALG')
    def test_malg_args_marker(self):
        self.edit_json('args.json',lambda r:r.update(model='MALG-MobileNetV2')); self.invalid('non-MDB')
    def test_history_best_only_stage_two(self):
        h=self.audit()['runs'][0]['history']; self.assertEqual(h['inferred_best_epoch'],2); self.assertEqual(h['best_val_macro_f1'],.6)
    def test_history_head_incomplete(self):
        self.edit_json('seed11_result.json',lambda x:x['history'].pop(0)); self.invalid('head epochs')
    def test_history_nan_loss(self):
        self.edit_json('seed11_result.json',lambda x:x['history'][1].update(loss=float('nan'))); self.invalid('finite')
    def test_history_nan_validation_f1(self):
        self.edit_json('seed11_result.json',lambda x:x['history'][1].update(val_f1=float('nan'))); self.invalid('finite')
    def test_history_nan_dice_explicit_warning(self):
        self.edit_json('seed11_result.json',lambda x:x['history'][1].update(val_dice=float('nan')))
        a=self.audit(); self.assertEqual(a['validation'],'VERIFIED'); self.assertTrue(any('val_dice NaN' in x for x in a['warnings']))
    def test_history_truncated(self):
        self.edit_json('seed11_result.json',lambda x:x['history'].pop()); self.invalid('truncated')
    def test_history_excess_after_patience(self):
        def change(x):
            h=copy.deepcopy(x['history'][-1]); h['epoch']=4; x['history'].append(h)
        self.edit_json('seed11_result.json',change); self.invalid('continued after')
    def test_history_best_mismatch(self):
        self.edit_json('seed11_result.json',lambda x:x.update(best_val_macro_f1=.9)); self.invalid('best_val_macro_f1')
    def test_summary_sd_mismatch(self):
        self.edit_json('summary.json',lambda x:x['c_all'].update(acc_std=0)); self.invalid('acc_std')
    def test_missing_class_still_uses_fixed_five(self):
        rows=[dict(true=0,predicted=0,background=bg) for bg in ('white','field')]
        metrics=m.compute_metrics(rows)
        self.assertEqual(metrics['c_all']['macro_f1'],.2)
        self.assertEqual(metrics['c_all']['cm'],np.diag([2,0,0,0,0]).tolist())
        present_class_score=copy.deepcopy(metrics['c_all']); present_class_score['macro_f1']=1.
        with self.assertRaises(m.VerificationError):
            m.compare_metric_record(present_class_score,metrics['c_all'],'missing-class convention mismatch')
    def test_segmentation_not_classification(self):
        self.edit_json('seed11_result.json',lambda x:x.update(seg=dict(n=5,dice=.8,iou=.7)))
        a=self.audit(); self.assertEqual(a['validation'],'VERIFIED'); self.assertEqual(a['segmentation']['runs'],1)
        self.assertIn('pseudo',a['segmentation']['interpretation'])
    def test_stale_segmentation_summary(self):
        write_json(self.root/'summary_seg.json',dict(runs=5,dice_mean=.8,dice_std=0,iou_mean=.7,iou_std=0)); self.invalid('stale')
    def test_partial_default_only_diagnostics(self):
        a=self.audit(); out=self.parent/'diagnostics'; names=m.export_results(a,out)
        self.assertEqual(set(names),{'verification_report.md','experiment_provenance.json'}); self.assertFalse((out/'summary_results.csv').exists())
    def test_partial_descriptive_export_and_member6(self):
        out=self.parent/'descriptive'; names=m.export_results(self.audit(),out,True)
        self.assertIn('result_table.tex',names); self.assertEqual(len(list((out/'confusion_matrices').glob('*.csv'))),18)
        with (out/'member6_prediction_cases.csv').open(encoding='utf-8') as f:
            cases=list(csv.DictReader(f))
        self.assertEqual(len(cases),5*45*2); self.assertEqual({r['seed'] for r in cases},set(map(str,TEST_SEEDS)))
        self.assertTrue(all(r['predictions_source'].startswith(str(self.root)) for r in cases))
        self.assertIn('PARTIALLY VERIFIED',(out/'result_table.tex').read_text())
    def test_no_overwrite_even_empty_directory(self):
        out=self.parent/'existing'; out.mkdir()
        with self.assertRaises(m.VerificationError):
            m.export_results(self.audit(),out,True)
        self.assertEqual(list(out.iterdir()),[])
    def test_output_inside_input_refused(self):
        with self.assertRaises(m.VerificationError):
            m.export_results(self.audit(),self.root/'output',True)
    def test_invalid_never_exports_metrics(self):
        self.edit_json('seed11_result.json',lambda x:x.update(seed=23)); a=self.audit(); out=self.parent/'invalid'
        m.export_results(a,out,True); self.assertFalse((out/'summary_results.csv').exists())
    def test_cli_require_complete_exit_two(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code=m.main(['--input',str(self.root),'--output',str(self.parent/'cli'),'--expected-seeds',*map(str,TEST_SEEDS),'--require-complete'])
        self.assertEqual(code,2)
    def test_csv_missing_column(self):
        p=self.root/'seed11_predictions.csv'; p.write_text('image,true\nx,brown_spot\n',encoding='utf-8'); self.invalid('columns')
    def test_count_mismatch(self):
        self.edit_json('seed11_result.json',lambda x:x['scenarios']['c_all'].update(n=46)); self.invalid('count mismatch')
    def test_args_duplicate_seeds(self):
        self.edit_json('args.json',lambda x:x.update(seeds='[11,23,37,51,51]')); self.invalid('Duplicate')
    def test_aggregate_requires_five(self):
        with self.assertRaises(m.VerificationError):
            m.aggregate(self.audit()['runs'][:4])
    def test_confidence_inconsistent(self):
        self.edit_csv(lambda r:r[0].update(confidence='.1')); self.invalid('confidence differs')
    def test_history_epoch_gap(self):
        self.edit_json('seed11_result.json',lambda x:x['history'][1].update(epoch=2)); self.invalid('noncontiguous')
    def test_history_infinite_dice_rejected(self):
        self.edit_json('seed11_result.json',lambda x:x['history'][1].update(val_dice=float('inf'))); self.invalid('finite')
    def test_all_seed_errors_recorded(self):
        for seed in TEST_SEEDS:
            self.edit_json(f'seed{seed}_result.json',lambda x:x.update(best_val_macro_f1=.1))
        a=self.invalid(); self.assertEqual(sum(e.startswith('seed ') for e in a['errors']),5)

    def test_unknown_json_nonfinite_rejected(self):
        self.edit_json('seed11_result.json',lambda x:x.update(extra=float('inf'))); self.invalid('nonfinite')
    def test_saved_best_epoch_conflict(self):
        self.edit_json('seed11_result.json',lambda x:x.update(best_epoch=1)); self.invalid('best_epoch')
    def test_inputs_changed_after_audit_refused(self):
        a=self.audit(); self.edit_json('args.json',lambda x:x.update(batch_size='32'))
        out=self.parent/'changed'
        with self.assertRaises(m.VerificationError):
            m.export_results(a,out,True)
        self.assertFalse(out.exists())
    def test_history_exceeds_maximum(self):
        self.edit_json('args.json',lambda x:x.update(epochs='2')); self.invalid('excess fine-tuning')
    def test_no_seed_default(self):
        with self.assertRaises(SystemExit),contextlib.redirect_stderr(io.StringIO()):
            m.main(['--input',str(self.root),'--output',str(self.parent/'no_seed')])
    def test_probability_inf_rejected(self):
        self.edit_csv(lambda r:r[0].update(p_brown_spot='inf')); self.invalid('finite')

    def test_seg_loss_without_masks_rejected(self):
        self.edit_json('seed11_result.json',lambda x:x['history'][1].update(seg=.3,loss=1.3)); self.invalid('without active mask')
    def test_disabled_contrastive_loss_rejected(self):
        self.edit_json('seed11_result.json',lambda x:x['history'][1].update(supcon=.1)); self.invalid('disabled contrastive')

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='member5_mdb_SYNTHETIC_EVIDENCE_'); self.addCleanup(self.temp.cleanup)
        self.parent=Path(self.temp.name); self.root,self.approval,self.evidence=make_fixture(self.parent,full=True)
    def audit(self,approval=True,evidence=True):
        return m.audit_experiment(self.root,TEST_SEEDS,self.approval if approval else None,self.evidence if evidence else None)
    def mutate_evidence(self,func):
        ev=m.read_json(self.evidence); func(ev); write_json(self.evidence,ev)
    def mutate_record_file(self,key,func):
        ev=m.read_json(self.evidence); ref=ev['runs'][0][key]; p=Path(ref['path']); obj=m.read_json(p); func(obj); write_json(p,obj)
        ev['runs'][0][key]=reference(p); write_json(self.evidence,ev)
    def invalid(self,needle):
        a=self.audit(); self.assertEqual(a['status'],'INVALID',a['errors']); self.assertIn(needle,' '.join(a['errors']))
    def test_contract_complete_in_temp_only(self):
        a=self.audit(); self.assertEqual(a['status'],'COMPLETE_VERIFIED',a['errors'])
        self.assertIn('external runner records',' '.join(a['limitations']))
        out=self.parent/'verified_contract_output'; m.export_results(a,out)
        self.assertTrue((out/'summary_results.csv').exists())
        # This checks a record contract, never claims this fake fixture was trained.
        self.assertIn('Synthetic tests are not real experiments',(out/'verification_report.md').read_text())
    def test_approval_without_execution_evidence_is_partial(self):
        a=self.audit(evidence=False); self.assertEqual(a['status'],'PARTIALLY VERIFIED')
    def test_execution_evidence_without_approval_is_partial(self):
        a=self.audit(approval=False); self.assertEqual(a['status'],'PARTIALLY VERIFIED',a['errors'])
    def test_approved_config_conflict(self):
        a=m.read_json(self.approval); a['args']['batch_size']=32; write_json(self.approval,a); self.invalid('args conflict')
    def test_approved_split_ratio_conflict(self):
        a=m.read_json(self.approval); a['protocol']['split_ratios']=[.8,.1,.1]; write_json(self.approval,a); self.invalid('protocol conflict')
    def test_per_seed_scientific_conflict(self):
        self.mutate_record_file('config',lambda c:c['args'].update(lr_head=.002)); self.invalid('scientific configuration conflict')
    def test_missing_actual_split_manifest(self):
        self.mutate_evidence(lambda ev:ev['runs'][0].pop('split')); self.invalid('file reference')
    def test_split_overlap(self):
        self.mutate_record_file('split',lambda x:x['val'].__setitem__(0,x['train'][0])); self.invalid('overlapping')
    def test_background_pool_test_leakage(self):
        self.mutate_record_file('split',lambda x:x['background_swap_images'].append(x['test'][0])); self.invalid('swap pool')
    def test_checkpoint_wrong_best_epoch(self):
        self.mutate_evidence(lambda ev:ev['runs'][0]['checkpoint'].update(epoch=1)); self.invalid('checkpoint selection')
    def test_checkpoint_tamper(self):
        ev=m.read_json(self.evidence); Path(ev['runs'][0]['checkpoint']['path']).write_bytes(b'changed'); self.invalid('hash mismatch')
    def test_source_image_tamper(self):
        ev=m.read_json(self.evidence); manifest=m.read_json(ev['dataset_manifest']['path']); p=Path(m.read_json(self.root/'args.json')['data_root'])/manifest[0]['image']
        p.write_bytes(b'changed'); self.invalid('source image SHA')
    def test_frozen_manifest_approval_mismatch(self):
        a=m.read_json(self.approval); a['dataset_manifest_sha256']='0'*64; write_json(self.approval,a); self.invalid('manifest hash')
    def test_pilot_evidence_rejected(self):
        self.mutate_evidence(lambda ev:ev['runs'][0].update(purpose='pilot')); self.invalid('Pilot/non-MDB')
    def test_duplicate_evidence_seed(self):
        self.mutate_evidence(lambda ev:ev['runs'][1].update(seed=11)); self.invalid('duplicate')
    def test_missing_environment(self):
        self.mutate_record_file('environment',lambda e:e.pop('torch')); self.invalid('environment incomplete')
    def test_output_file_binding(self):
        self.mutate_evidence(lambda ev:ev['runs'][0].update(result=ev['runs'][1]['result'])); self.invalid('not bound')
    def test_frozen_preprocessing_approval_mismatch(self):
        a=m.read_json(self.approval); a.pop('preprocessing_sha256'); write_json(self.approval,a); self.invalid('preprocessing hashes')

    def test_mask_index_qc_semantic_conflict(self):
        ev=m.read_json(self.evidence); ref=ev['preprocessing']['mask_index']; p=Path(ref['path'])
        manifest=m.read_json(ev['dataset_manifest']['path']); row=next(r for r in manifest if r['background']=='white')
        p.write_text('image,cache_image,mask,class,status\n'+row['image']+',_interim/cache_448/'+row['image']+',_interim/masks_448/test.png,'+row['label']+',ok\n',encoding='utf-8')
        ev['preprocessing']['mask_index']=reference(p); write_json(self.evidence,ev)
        a=m.read_json(self.approval); a['preprocessing_sha256']['mask_index']=m.file_hash(p); write_json(self.approval,a)
        self.invalid('index/QC coverage')

if __name__=='__main__':
    unittest.main()
