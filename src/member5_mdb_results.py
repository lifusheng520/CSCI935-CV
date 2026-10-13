"""MDB-Net prediction verification. Never imports training code or chooses seeds.

Missing approval/execution evidence => PARTIALLY VERIFIED, never COMPLETE_VERIFIED.
CLI: python -B -m src.member5_mdb_results --help
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import io
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

CLASSES = ('brown_spot', 'leaf_scald', 'rice_blast', 'rice_tungro', 'sheath_blight')
SCENARIOS = {'a_white': 'White', 'b_field': 'Field', 'c_all': 'Mixed'}
METRIC_TOL = 1e-8
ROUND_TOL = 0.00050001
SCIENCE_KEYS = ('img_size','cache_size','batch_size','head_epochs','epochs','lr_head',
 'lr_backbone','weight_decay','label_smoothing','dropout','patience','lambda_seg','pool',
 'mil_topk_frac','lambda_supcon','supcon_temp','bg_swap','tf_prob','field_train','mask_status',
 'class_weights','from_scratch','allow_missing')
ARG_KEYS = SCIENCE_KEYS + ('data_root','seeds','exclude_file','weights_path','workers','save_models','out_name')
INT_KEYS = {'img_size','cache_size','batch_size','head_epochs','epochs','patience','workers'}
BOOL_KEYS = {'class_weights','from_scratch','allow_missing','save_models'}
FLOAT_KEYS = set(SCIENCE_KEYS)-INT_KEYS-BOOL_KEYS-{'pool','field_train','mask_status'}
CSV_COLUMNS = ('image','background','true','predicted','correct','confidence') + tuple('p_'+c for c in CLASSES)
PROTOCOL = {'model':'MDB-Net','class_names':list(CLASSES),'split_ratios':[.7,.15,.15],
 'stratification':'class_background','selection':'stage2_validation_macro_f1',
 'macro_f1':'fixed_five_zero_division_0','std_ddof':1}
# External contracts; this module NEVER manufactures runner records.
APPROVAL_SCHEMA = {'schema_version':1,'status':'APPROVED','purpose':'formal','model':'MDB-Net',
 'approval_reference':'external dated approval reference','expected_seeds':'five integers',
 'protocol':PROTOCOL,'args':'all ARG_KEYS with approved native JSON values'}
EVIDENCE_SCHEMA = {'schema_version':1,'model':'MDB-Net','purpose':'formal',
 'dataset_manifest':'file reference to ordered JSON image/label/background/sha256 records',
 'source_code':'four audited script basenames -> file references',
 'preprocessing':'cache_manifest/mask_index/qc_report/exclude_file/policy -> file references',
 'runs':'five seed/model/purpose/timestamps/config/split/log/environment/result/predictions/checkpoint records',
 'pretrained_weights':'file reference when not from_scratch'}

class VerificationError(ValueError):
    pass

def require(ok, message):
    if not ok:
        raise VerificationError(message)

def number(v, ctx, low=None, high=None):
    require(type(v) in (int,float) and math.isfinite(v), f'{ctx}: finite number required')
    require(low is None or v >= low, f'{ctx}: below {low}')
    require(high is None or v <= high, f'{ctx}: above {high}')
    return float(v)

def integer(v, ctx, low=0):
    require(type(v) is int and v >= low, f'{ctx}: integer >= {low} required')
    return v

def close(v, expected, ctx, tolerance=METRIC_TOL):
    number(v,ctx)
    require(abs(v-expected) <= tolerance, f'{ctx}: saved {v!r} != recomputed {expected!r}')

def normalize_path(v):
    require(isinstance(v,str) and v and v == v.strip(), 'image: nonempty unpadded relative path required')
    require(not any(ord(c)<32 for c in v), 'image: control character')
    v=v.replace('\\','/'); parts=v.split('/')
    require(not v.startswith('/') and ':' not in v and all(p not in ('','.','..') for p in parts), f'image: unsafe relative path {v!r}')
    return PurePosixPath(v).as_posix()

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()

def _unique_object(pairs):
    out={}
    for k,v in pairs:
        require(k not in out, f'JSON duplicate key: {k}')
        out[k]=v
    return out

def _check_json_numbers(value, location=()):
    if isinstance(value, dict):
        for key, item in value.items():
            _check_json_numbers(item, location+(key,))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _check_json_numbers(item, location+(i,))
    elif type(value) is float and not math.isfinite(value):
        # Upstream explicitly writes val_dice=NaN when there are no validation masks.
        require(math.isnan(value) and len(location)>=3 and location[-1]=='val_dice' and 'history' in location,
                f'JSON nonfinite value outside documented history/val_dice exception: {location}')

def read_json(path):
    try:
        value=json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=_unique_object)
        _check_json_numbers(value)
        return value
    except (OSError,ValueError) as exc:
        raise VerificationError(f'{path}: invalid JSON/unreadable: {exc}') from exc

def seeds_checked(seeds):
    require(isinstance(seeds,(list,tuple)) and len(seeds)==5,'Exactly five externally supplied expected seeds required')
    require(all(type(s) is int and 0<=s<2**32 for s in seeds),'seeds: integers in [0,2**32) required')
    require(len(set(seeds))==5,'Duplicate expected seeds')
    return list(seeds)

def parse_args_record(raw):
    require(isinstance(raw,dict),'args.json: object required')
    require(set(ARG_KEYS)<=set(raw), f'args.json: missing keys {sorted(set(ARG_KEYS)-set(raw))}')
    for k in ('model','variant','model_architecture'):
        require(k not in raw or raw[k]=='MDB-Net', f'args: non-MDB model marker {k}')
    for k in ('pilot','quick','synthetic'):
        require(raw.get(k) not in (True,'True','true'), f'args: {k} cannot count as formal')
    require(raw.get('purpose','formal')=='formal','args: Pilot/nonformal purpose')
    args={}
    for k in ARG_KEYS:
        v=raw[k]
        if isinstance(v,str) and k in INT_KEYS|FLOAT_KEYS|BOOL_KEYS|{'seeds'}:
            try:
                v=ast.literal_eval(v)
            except (SyntaxError,ValueError) as exc:
                raise VerificationError(f'args/{k}: invalid typed value') from exc
        elif v=='None' and k in ('exclude_file','weights_path','out_name'):
            v=None
        if k in INT_KEYS:
            integer(v,f'args/{k}',0 if k in ('head_epochs','patience','workers') else 1)
        elif k in FLOAT_KEYS:
            v=number(v,f'args/{k}',0)
        elif k in BOOL_KEYS:
            require(type(v) is bool,f'args/{k}: boolean required')
        args[k]=v
    args['seeds']=seeds_checked(args['seeds'])
    require(args['pool'] in ('mask','global','mil','proto'),'args/pool invalid')
    require(args['field_train'] in ('none','cls'),'args/field_train invalid')
    require(args['img_size']%32==0,'args/img_size must be multiple of 32')
    for k in ('label_smoothing','dropout','bg_swap','tf_prob','mil_topk_frac'):
        require(args[k]<=1,f'args/{k}: must be <=1')
    for k in ('lr_head','lr_backbone','supcon_temp','mil_topk_frac'):
        require(args[k]>0,f'args/{k}: positive required')
    require(isinstance(args['mask_status'],str) and args['mask_status'] and set(args['mask_status'].split(','))<={'ok','review','fail','excluded'},'args/mask_status invalid')
    require(isinstance(args['data_root'],str) and bool(args['data_root']),'args/data_root invalid')
    for k in ('exclude_file','weights_path','out_name'):
        require(args[k] is None or isinstance(args[k],str),f'args/{k}: string/null required')
    require(not re.search(r'(^|[^a-z])(pilot|smoke|synthetic|malg)([^a-z]|$)',str(args['out_name']).lower()),'args/out_name: Pilot/synthetic/MALG marker')
    return args

def read_predictions(path,warnings):
    try:
        with Path(path).open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f); cols=reader.fieldnames or []
            require(len(cols)==len(set(cols)) and set(CSV_COLUMNS)<=set(cols),f'{path}: missing/duplicate MDB CSV columns')
            rows=list(reader)
    except (OSError,csv.Error,UnicodeError) as exc:
        raise VerificationError(f'{path}: unreadable CSV: {exc}') from exc
    require(rows,f'{path}: empty predictions')
    seen,out,tied=set(),[],0
    for i,r in enumerate(rows,2):
        ctx=f'{path.name}:row {i}'
        require(None not in r and all(v is not None for v in r.values()),f'{ctx}: malformed CSV row')
        image=normalize_path(r['image'])
        require(image.casefold() not in seen,f'{ctx}: duplicate image path including Windows aliases')
        seen.add(image.casefold())
        require(r['background'] in ('white','field'),f'{ctx}: invalid background')
        require(r['true'] in CLASSES and r['predicted'] in CLASSES,f'{ctx}: invalid five-class label')
        y,p=CLASSES.index(r['true']),CLASSES.index(r['predicted'])
        require(r['correct'] in ('0','1') and int(r['correct'])==int(y==p),f'{ctx}: incorrect correct flag')
        try:
            prob=[float(r['p_'+c]) for c in CLASSES]; confidence=float(r['confidence'])
        except ValueError as exc:
            raise VerificationError(f'{ctx}: invalid probability') from exc
        for v in prob+[confidence]:
            number(v,f'{ctx}/probability',0,1)
        require(abs(sum(prob)-1)<=5*ROUND_TOL,f'{ctx}: probability sum exceeds three-decimal rounding bound')
        require(max(prob)-prob[p]<=2*ROUND_TOL,f'{ctx}: prediction contradicts rounded probabilities')
        require(abs(confidence-max(prob))<=2*ROUND_TOL,f'{ctx}: confidence differs from rounded maximum')
        if sum(abs(v-max(prob))<=2*ROUND_TOL for v in prob)>1:
            tied+=1
        out.append(dict(image=image,background=r['background'],true=y,predicted=p,correct=int(y==p),confidence=confidence,probabilities=prob))
    if tied:
        warnings.append(f'{path.name}: {tied} rounded/near-tied rows; exact logits/argmax cannot be reconstructed')
    return out

def compute_metrics(rows):
    out={}
    for sc in SCENARIOS:
        sub=[r for r in rows if sc=='c_all' or r['background']==('white' if sc=='a_white' else 'field')]
        require(sub,f'{sc}: empty required scenario')
        y,p=[r['true'] for r in sub],[r['predicted'] for r in sub]
        out[sc]=dict(n=len(y),acc=float(accuracy_score(y,p)),macro_f1=float(f1_score(y,p,labels=list(range(5)),average='macro',zero_division=0)),
            per_class_f1=f1_score(y,p,labels=list(range(5)),average=None,zero_division=0).tolist(),cm=confusion_matrix(y,p,labels=list(range(5))).tolist())
    require(out['a_white']['n']+out['b_field']['n']==out['c_all']['n'],'scenario coverage invalid')
    return out

def compare_metric_record(saved,expected,ctx):
    require(isinstance(saved,dict),f'{ctx}: metric object required')
    require(type(saved.get('n')) is int and saved['n']==expected['n'],f'{ctx}: sample count mismatch')
    for k in ('acc','macro_f1'):
        close(saved.get(k),expected[k],f'{ctx}/{k}')
    pc=saved.get('per_class_f1')
    require(isinstance(pc,list) and len(pc)==5,f'{ctx}: five per-class F1 required')
    for i,(v,e) in enumerate(zip(pc,expected['per_class_f1'])):
        close(v,e,f'{ctx}/per_class_f1/{i}')
    cm=saved.get('cm')
    require(isinstance(cm,list) and len(cm)==5 and all(isinstance(r,list) and len(r)==5 and all(type(v) is int and v>=0 for v in r) for r in cm),f'{ctx}: integer 5x5 confusion matrix required')
    require(cm==expected['cm'],f'{ctx}: confusion matrix mismatch')

def verify_history(result,args,warnings):
    hist=result.get('history')
    require(isinstance(hist,list) and hist,'history: nonempty list required')
    counts,ft,seen_ft={1:0,2:0},[],False
    for h in hist:
        require(isinstance(h,dict) and type(h.get('stage')) is int and h['stage'] in (1,2),'history: invalid stage')
        stage=h['stage']; require(not(seen_ft and stage==1),'history: head after fine-tuning')
        seen_ft |= stage==2; counts[stage]+=1
        require(type(h.get('epoch')) is int and h['epoch']==counts[stage],'history: noncontiguous/duplicate epoch')
        for k in ('loss','cls','seg','supcon'):
            number(h.get(k),f'history/{k}',0)
        for k in ('val_f1','val_acc'):
            number(h.get(k),f'history/{k}',0,1)
        vd=h.get('val_dice')
        if type(vd) in (int,float) and math.isnan(vd):
            warnings.append('history val_dice NaN: source can emit this with no trusted validation masks; segmentation NOT VERIFIED')
        else:
            number(vd,'history/val_dice',0,1)
        close(h['loss'],h['cls']+args['lambda_seg']*h['seg']+args['lambda_supcon']*h['supcon'],'history/weighted_loss',1e-6)
        if stage==2:
            ft.append(h)
    require(counts[1]==args['head_epochs'],'history: incomplete/excess head epochs')
    require(0<len(ft)<=args['epochs'],'history: incomplete/excess fine-tuning epochs')
    best,best_epoch,bad,stop=-1.0,None,0,False
    for h in ft:
        require(not stop,'history: continued after patience early stop')
        if h['val_f1']>best:
            best,best_epoch,bad=h['val_f1'],h['epoch'],0
        else:
            bad+=1; stop=bad>=args['patience']
    require(len(ft)==args['epochs'] or stop,'history: truncated without early-stop condition')
    close(result.get('best_val_macro_f1'),best,'history/best_val_macro_f1')
    if 'best_epoch' in result:
        require(type(result['best_epoch']) is int and result['best_epoch']==best_epoch, 'history: saved best_epoch conflicts with inferred selection')
    integer(result.get('n_train'),'n_train',1); integer(result.get('n_train_masked'),'n_train_masked')
    require(result['n_train_masked']<=result['n_train'],'n_train_masked exceeds n_train')
    if result['n_train_masked']==0 or args['lambda_seg']==0:
        require(all(h['seg']==0 for h in hist), 'history: nonzero segmentation loss without active mask supervision')
    if args['lambda_supcon']==0:
        require(all(h['supcon']==0 for h in hist), 'history: nonzero disabled contrastive loss')
    return dict(head_epochs=counts[1],finetune_epochs=len(ft),inferred_best_epoch=best_epoch,
        best_val_macro_f1=best,checkpoint_status='CHECKPOINT IDENTITY NOT VERIFIED')

def aggregate(runs):
    require(len(runs)==5 and len({r['seed'] for r in runs})==5,'aggregate: five unique seeds required')
    out={}
    for sc in SCENARIOS:
        ms=[r['metrics'][sc] for r in runs]; v={'runs':5}
        for k in ('acc','macro_f1'):
            values=[x[k] for x in ms]; v[k+'_mean'],v[k+'_std']=float(np.mean(values)),float(np.std(values,ddof=1))
        pcs=np.array([x['per_class_f1'] for x in ms])
        v['per_class_f1_mean']=dict(zip(CLASSES,pcs.mean(0).round(4).tolist()))
        v['per_class_f1_std']=dict(zip(CLASSES,pcs.std(0,ddof=1).tolist()))
        v['cm_sum']=np.sum([x['cm'] for x in ms],axis=0).tolist(); out[sc]=v
    return out

def compare_summary(saved,computed):
    require(isinstance(saved,dict) and set(saved)==set(SCENARIOS),'summary.json: three exact scenarios required')
    for sc,v in computed.items():
        s=saved[sc]
        require(isinstance(s,dict) and type(s.get('runs')) is int and s['runs']==5,f'summary/{sc}: five runs required')
        for k in ('acc_mean','acc_std','macro_f1_mean','macro_f1_std'):
            close(s.get(k),v[k],f'summary/{sc}/{k}')
        for k in ('per_class_f1_mean','per_class_f1_std'):
            require(isinstance(s.get(k),dict) and set(s[k])==set(CLASSES),f'summary/{sc}/{k}: five classes required')
            for c in CLASSES:
                close(s[k][c],v[k][c],f'summary/{sc}/{k}/{c}')
        require(s.get('cm_sum')==v['cm_sum'],f'summary/{sc}: cm_sum mismatch')

def segmentation_summary(runs,root):
    valid=[]
    for r in runs:
        seg=r['result'].get('seg'); require(isinstance(seg,dict),f"seed {r['seed']}: missing seg record")
        n=integer(seg.get('n'),'seg/n'); require(n<=len(r['rows']),'seg/n exceeds test count')
        if n:
            for k in ('dice','iou'):
                number(seg.get(k),'seg/'+k,0,1)
            valid.append(seg)
        else:
            require(seg.get('dice') is None and seg.get('iou') is None,'seg: zero count requires null dice/iou')
    out={'interpretation':'Agreement with automatic pseudo masks, NOT hand-annotated segmentation accuracy','runs':len(valid)}
    if valid:
        for k in ('dice','iou'):
            values=[s[k] for s in valid]; out[k+'_mean'],out[k+'_std']=float(np.mean(values)),float(np.std(values,ddof=1)) if len(values)>1 else 0.0
    p=root/'summary_seg.json'
    if p.exists():
        saved=read_json(p)
        require(isinstance(saved,dict) and type(saved.get('runs')) is int and saved['runs']==len(valid) and valid,'summary_seg: count mismatch/stale output')
        for k in ('dice_mean','dice_std','iou_mean','iou_std'):
            close(saved.get(k),out[k],'summary_seg/'+k)
    return out

def _file_ref(ref,base,hashes):
    require(isinstance(ref,dict) and isinstance(ref.get('path'),str) and re.fullmatch(r'[0-9a-fA-F]{64}',str(ref.get('sha256',''))),'evidence: file reference requires path and SHA256')
    p=Path(ref['path']); p=(base/p).resolve() if not p.is_absolute() else p.resolve()
    require(p.is_file() and file_hash(p)==ref['sha256'].lower(),f'evidence: missing file/hash mismatch {p}')
    hashes[str(p)]=file_hash(p); return p

def verify_external(approval_path,evidence_path,root,args,runs,seeds,hashes):
    """Check externally supplied evidence bindings, not record authorship/training truth."""
    limitations=[]; approved=None
    if approval_path:
        approved=read_json(approval_path)
        require(isinstance(approved,dict) and approved.get('schema_version')==1 and approved.get('status')=='APPROVED'
            and approved.get('purpose')=='formal' and approved.get('model')=='MDB-Net','approved-config: invalid approval/model/purpose')
        require(isinstance(approved.get('approval_reference'),str) and approved['approval_reference'].strip(),'approved-config: reference missing')
        require(set(seeds_checked(approved.get('expected_seeds')))==set(seeds),'approved-config: seed conflict')
        require(approved.get('protocol')==PROTOCOL,'approved-config: evaluation/split protocol conflict')
        require(parse_args_record(approved.get('args'))==args,'approved-config: actual args conflict')
        hashes[str(Path(approval_path).resolve())]=file_hash(approval_path)
    else:
        limitations.append('Approved MDB-Net configuration NOT VERIFIED; expected-seeds are not proof of approval')
    if not evidence_path:
        return limitations+['Per-seed configurations/model provenance NOT VERIFIED',
            'Actual train/validation/test manifests, hashes and distinct splits NOT VERIFIED',
            'CHECKPOINT IDENTITY NOT VERIFIED','Logs/environment/preprocessing provenance NOT VERIFIED',
            'Pilot exclusion cannot be proved from shared args/file names alone']
    ev=read_json(evidence_path)
    require(isinstance(ev,dict) and ev.get('schema_version')==1 and ev.get('model')=='MDB-Net' and ev.get('purpose')=='formal','evidence: model/schema/purpose invalid (Pilot/MALG rejected)')
    require(args['save_models'], 'evidence: save_models must be enabled for checkpoint verification')
    base=Path(evidence_path).resolve().parent; hashes[str(Path(evidence_path).resolve())]=file_hash(evidence_path)
    code=ev.get('source_code'); require(isinstance(code,dict),'evidence: source_code missing')
    for name in ('train_multitask.py','train_resnet18.py','make_masks.py','seg_leaf.py'):
        p=_file_ref(code.get(name),base,hashes)
        require(file_hash(p)==file_hash(Path(__file__).parent/name),f'evidence: audited source version mismatch {name}')
    prep=ev.get('preprocessing'); require(isinstance(prep,dict),'evidence: preprocessing missing')
    for name in ('cache_manifest','mask_index','qc_report','exclude_file','policy'):
        _file_ref(prep.get(name),base,hashes)
    policy=read_json(_file_ref(prep['policy'],base,hashes))
    require(isinstance(policy,dict) and policy.get('status')=='APPROVED' and policy.get('approval_reference') and
        all(isinstance(policy.get(k),str) and policy[k].strip() for k in ('mask_qc_scope','test_exposure','near_duplicate_policy')),'evidence: leakage/preprocessing policy approval incomplete')
    manifest_path=_file_ref(ev.get('dataset_manifest'),base,hashes)
    if approved:
        require(approved.get('dataset_manifest_sha256')==file_hash(manifest_path), 'approved-config: frozen dataset manifest hash missing/mismatched')
    if not args['from_scratch']:
        weights=_file_ref(ev.get('pretrained_weights'),base,hashes)
        expected_weight=Path(args['weights_path']).resolve() if args['weights_path'] else (Path.home()/'.cache/torch/hub/checkpoints/resnet18-f37072fd.pth').resolve()
        require(weights==expected_weight, 'evidence: pretrained weight not bound to actual configured source')
        if approved:
            require(approved.get('pretrained_weights_sha256')==file_hash(weights), 'approved-config: pretrained weight hash missing/mismatched')
    manifest=read_json(manifest_path)
    require(isinstance(manifest,list) and manifest,'evidence: ordered dataset manifest required')
    dataset,seen,source_hashes={},set(),set(); rawroot=Path(args['data_root']).resolve()
    for key in ('mask_index','qc_report'):
        require(_file_ref(prep[key],base,hashes)==(rawroot/'_interim'/'qc'/(key+'.csv')).resolve(), f'evidence: {key} not bound to training data root')
    excluded_path=Path(args['exclude_file']).resolve() if args['exclude_file'] else (rawroot/'_interim/qc/exclude.txt').resolve()
    require(_file_ref(prep['exclude_file'],base,hashes)==excluded_path, 'evidence: exclusion file not bound to actual args')
    for row in manifest:
        image=normalize_path(row['image']); require(image.casefold() not in seen,'evidence: duplicate manifest path')
        seen.add(image.casefold())
        require(row['label'] in CLASSES and row['background'] in ('white','field'),'evidence: invalid manifest label/background')
        p=(rawroot/image).resolve()
        require(p.is_relative_to(rawroot) and p.is_file(),'evidence: source path missing/escapes root')
        require(re.fullmatch('[0-9a-f]{64}',str(row.get('sha256',''))) and file_hash(p)==row['sha256'],'evidence: source image SHA mismatch')
        require(row['sha256'] not in source_hashes,'evidence: identical images across paths')
        source_hashes.add(row['sha256']); dataset[image]=row
    # Verify full raw membership and discovery order, not just a self-selected Pilot subset.
    raw_paths={p.relative_to(rawroot).as_posix() for p in rawroot.rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff') and not any(x.startswith('_') for x in p.relative_to(rawroot).parts)}
    require(set(dataset)==raw_paths and list(dataset)==[p.as_posix() for p in sorted(Path(x) for x in dataset)], 'evidence: dataset membership/order differs from raw discovery')
    cache_records=read_json(_file_ref(prep['cache_manifest'],base,hashes))
    require(isinstance(cache_records,list) and len(cache_records)==len(dataset), 'evidence: cache/mask manifest must cover all source images')
    index_path=_file_ref(prep['mask_index'],base,hashes)
    with index_path.open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        require(set(('image','cache_image','mask','class','status'))<=set(reader.fieldnames or []), 'evidence: mask index columns missing')
        index={}
        for row in reader:
            image=normalize_path(row['image'])
            require(image in dataset and image not in index and row['class']==dataset[image]['label'], 'evidence: mask index identity/label mismatch')
            index[image]=row
    with _file_ref(prep['qc_report'],base,hashes).open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        require(set(('image','cls','status'))<=set(reader.fieldnames or []), 'evidence: QC columns missing')
        qc={}
        for row in reader:
            image=normalize_path(row['image'])
            require(image in index and image not in qc and row['cls']==index[image]['class'] and row['status']==index[image]['status'], 'evidence: mask index/QC inconsistency')
            qc[image]=row
    require(set(qc)==set(index), 'evidence: mask index/QC coverage mismatch')
    trusted=set(); cache_seen=set()
    for row in cache_records:
        image=normalize_path(row['image']); require(image in dataset and image not in cache_seen, 'evidence: cache manifest identity mismatch')
        cache_seen.add(image)
        cache_path=_file_ref(row.get('cache'),base,hashes)
        require(cache_path==(rawroot/'_interim'/f"cache_{args['cache_size']}"/Path(image).with_suffix('.jpg')).resolve(), 'evidence: cache path not bound to actual loader')
        status=row.get('status'); require(status in ('ok','review','fail','excluded','none'), 'evidence: invalid mask status')
        actual=index.get(image)
        if actual is None:
            require(row.get('mask') is None and status=='none', 'evidence: mask manifest invents unindexed mask')
        else:
            require(status==actual['status'] and row.get('mask') is not None, 'evidence: mask index/cache manifest conflict')
            require((rawroot/normalize_path(actual['cache_image'])).resolve()==cache_path, 'evidence: index cache_image mismatch')
            require(_file_ref(row['mask'],base,hashes)==(rawroot/normalize_path(actual['mask'])).resolve(), 'evidence: index mask path mismatch')
        if row.get('mask') is not None:
            _file_ref(row['mask'],base,hashes)
            if status in args['mask_status'].split(','):
                trusted.add(image)
        else:
            require(status=='none', 'evidence: mask status without mask')
    excluded={normalize_path(x.strip()) for x in _file_ref(prep['exclude_file'],base,hashes).read_text(encoding='utf-8-sig').splitlines() if x.strip()}
    trusted-=excluded
    if approved:
        require(approved.get('preprocessing_sha256')=={k:prep[k]['sha256'].lower() for k in prep}, 'approved-config: frozen preprocessing hashes missing/mismatched')
    records=ev.get('runs')
    require(isinstance(records,list) and len(records)==5 and all(isinstance(r,dict) for r in records),'evidence: five run records required')
    require(all(type(r.get('seed')) is int for r in records) and {r['seed'] for r in records}==set(seeds),'evidence: missing/duplicate/unexpected seed')
    fingerprints,counts=set(),[]; ordered=list(dataset)
    from sklearn.model_selection import train_test_split
    strata=np.array([f"{CLASSES.index(dataset[p]['label'])}_{0 if dataset[p]['background']=='white' else 1}" for p in ordered])
    for record in records:
        seed=record['seed']; run=next(r for r in runs if r['seed']==seed)
        require(record.get('purpose')=='formal' and record.get('model')=='MDB-Net','evidence: Pilot/non-MDB run')
        try:
            start,end=[datetime.fromisoformat(record[k].replace('Z','+00:00')) for k in ('started_at','completed_at')]
            require(start.tzinfo is not None and end.tzinfo is not None and start<end,'evidence: invalid timestamps')
        except (KeyError,ValueError,TypeError,AttributeError) as exc:
            raise VerificationError('evidence: timezone-aware run timestamps required') from exc
        cfg=read_json(_file_ref(record.get('config'),base,hashes))
        require(isinstance(cfg,dict) and type(cfg.get('seed')) is int and cfg['seed']==seed and cfg.get('model')=='MDB-Net' and cfg.get('purpose')=='formal','evidence: per-seed identity mismatch')
        require(cfg.get('protocol')==PROTOCOL and parse_args_record(cfg.get('args'))==args,'evidence: per-seed scientific configuration conflict')
        log=_file_ref(record.get('log'),base,hashes); require(log.stat().st_size>0,'evidence: empty log')
        env=read_json(_file_ref(record.get('environment'),base,hashes))
        require(isinstance(env,dict) and all(isinstance(env.get(k),str) and env[k].strip() for k in ('python','torch','torchvision','numpy','scikit-learn','device')),'evidence: environment incomplete')
        for key,name in (('result',f'seed{seed}_result.json'),('predictions',f'seed{seed}_predictions.csv')):
            require(_file_ref(record.get(key),base,hashes)==(root/name).resolve(),f'evidence: {key} not bound to input')
        split=read_json(_file_ref(record.get('split'),base,hashes))
        require(isinstance(split,dict) and all(isinstance(split.get(k),list) and split[k] for k in ('train','val','test','training_images')) and isinstance(split.get('background_swap_images'),list),'evidence: actual split/training/background lists required')
        parts={k:[normalize_path(p) for p in split[k]] for k in ('train','val','test')}
        for k,v in parts.items():
            require(len(v)==len(set(v)),f'evidence: duplicate split path {k}')
        sets={k:set(v) for k,v in parts.items()}
        require(not(sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test']) and set.union(*sets.values())==set(dataset),'evidence: overlapping/incomplete split')
        idx=np.arange(len(ordered)); tv,te=train_test_split(idx,test_size=.15,stratify=strata,random_state=seed)
        tr,va=train_test_split(tv,test_size=.15/.85,stratify=strata[tv],random_state=seed)
        for k,ids in zip(('train','val','test'),(tr,va,te)):
            require(sets[k]=={ordered[i] for i in ids},f'evidence: split does not reproduce MDB algorithm: {k}')
        train=sets['train'] if args['field_train']=='cls' else {p for p in sets['train'] if dataset[p]['background']=='white'}
        bg={p for p in sets['train'] if dataset[p]['background']=='field'} if args['bg_swap']>0 else set()
        require({normalize_path(p) for p in split['training_images']}==train and {normalize_path(p) for p in split['background_swap_images']}==bg,'evidence: training/swap pool mismatch')
        require(run['result']['n_train']==len(train),'evidence: n_train differs from split')
        require(run['result']['n_train_masked']==len(train&trusted), 'evidence: n_train_masked differs from frozen mask records')
        require(run['result']['seg']['n']==len(sets['test']&trusted), 'evidence: segmentation count differs from frozen test masks')
        require({r['image'] for r in run['rows']}==sets['test'],'evidence: predictions do not cover exact test manifest')
        for r in run['rows']:
            require(CLASSES[r['true']]==dataset[r['image']]['label'] and r['background']==dataset[r['image']]['background'],'evidence: prediction truth differs from manifest')
        cp=record.get('checkpoint')
        require(isinstance(cp,dict) and cp.get('stage')==2 and cp.get('epoch')==run['history']['inferred_best_epoch'] and cp.get('selection')==PROTOCOL['selection'],'evidence: checkpoint selection/epoch mismatch')
        p=_file_ref(cp,base,hashes)
        require(p==(root/f'seed{seed}_model.pt').resolve() and p.stat().st_size>0,'evidence: checkpoint not bound to saved input model')
        run['history']['checkpoint_status']='VERIFIED (runner record, hash and epoch binding; weights not deserialized)'
        fingerprints.add(tuple(tuple(sorted(sets[k])) for k in ('train','val','test')))
        counts.append(tuple(len(sets[k]) for k in ('train','val','test')))
    require(len(fingerprints)==5 and len(set(counts))==1,'evidence: distinct splits with constant counts required')
    if not approved:
        limitations.append('Execution records supplied, but formal approval absent')
    limitations.append('Evidence integrity is relative to external runner records, NOT independent proof of authorship/training or absence of near-duplicate/development leakage')
    return limitations

def audit_experiment(input_dir,expected_seeds,approved_config=None,evidence_manifest=None):
    root=Path(input_dir).resolve()
    audit=dict(status='INVALID',validation='INVALID',errors=[],warnings=[],limitations=[],input=str(root),
        expected_seeds=list(expected_seeds),runs=[],summary={},input_hashes={},protocol=PROTOCOL)
    hashes=audit['input_hashes']
    try:
        seeds=seeds_checked(expected_seeds); require(root.is_dir(),'input: directory missing')
        require(not re.search(r'(^|[^a-z])(pilot|smoke|synthetic|malg)([^a-z]|$)',root.name.lower()),'input: Pilot/synthetic/MALG marker')
        require(not any(p.name.startswith('seed_') for p in root.iterdir()),'input: legacy MALG seed_ layout rejected')
        args=parse_args_record(read_json(root/'args.json')); audit['args']=args
        require(set(args['seeds'])==set(seeds),'args: expected seeds conflict')
        expected={f'seed{s}_{suffix}' for s in seeds for suffix in ('result.json','predictions.csv')}
        present={p.name for p in root.iterdir() if p.name.startswith('seed') and (p.name.endswith('_result.json') or p.name.endswith('_predictions.csv'))}
        require(present==expected,f'input: missing/extra seed files; missing={sorted(expected-present)}, extra={sorted(present-expected)}')
        for p in root.iterdir():
            if p.name.startswith('seed'):
                match=re.fullmatch(r'seed(\d+)_(result\.json|predictions\.csv|pred_masks\.jpg|model\.pt)',p.name)
                require(match is not None and int(match[1]) in seeds,f'input: unexpected/stale seed artifact {p.name}')
        for seed in seeds:
            try:
                rp,pp=root/f'seed{seed}_result.json',root/f'seed{seed}_predictions.csv'
                result=read_json(rp)
                require(isinstance(result,dict) and type(result.get('seed')) is int and result['seed']==seed,f'seed {seed}: identity mismatch')
                require(result.get('model','MDB-Net')=='MDB-Net' and result.get('purpose','formal')=='formal' and not result.get('pilot',False),f'seed {seed}: nonformal/non-MDB result')
                rows=read_predictions(pp,audit['warnings']); metrics=compute_metrics(rows)
                require(isinstance(result.get('scenarios'),dict) and set(result['scenarios'])==set(SCENARIOS),f'seed {seed}: scenarios mismatch')
                for sc in SCENARIOS:
                    compare_metric_record(result['scenarios'][sc],metrics[sc],f'seed {seed}/{sc}')
                history=verify_history(result,args,audit['warnings'])
                audit['runs'].append(dict(seed=seed,result=result,rows=rows,metrics=metrics,history=history,predictions_source=str(pp)))
            except (VerificationError,KeyError,TypeError,ValueError,OSError) as exc:
                audit['errors'].append(f'seed {seed}: {exc}')
        require(not audit['errors'],'One or more seeds INVALID; no aggregate produced')
        audit['summary']=aggregate(audit['runs']); compare_summary(read_json(root/'summary.json'),audit['summary'])
        audit['segmentation']=segmentation_summary(audit['runs'],root)
        for p in root.iterdir():
            if p.is_file() and (p.name in ('args.json','summary.json','summary_seg.json') or p.name.startswith('seed')):
                hashes[str(p)]=file_hash(p)
        audit['limitations']=verify_external(approved_config,evidence_manifest,root,args,audit['runs'],seeds,hashes)
        audit['validation']='VERIFIED'
        audit['status']='COMPLETE_VERIFIED' if approved_config is not None and evidence_manifest is not None else 'PARTIALLY VERIFIED'
        audit['verification_scope']='Recorded artifact integrity, prediction metrics and evidence bindings; not independent training attestation'
    except (VerificationError,KeyError,TypeError,ValueError,OSError) as exc:
        audit['errors'].append(str(exc)); audit['summary']={}; audit['status']=audit['validation']='INVALID'
    return audit

def csv_text(rows,columns=None):
    f=io.StringIO(newline=''); w=csv.DictWriter(f,fieldnames=columns or list(rows[0]))
    w.writeheader(); w.writerows(rows); return f.getvalue()

def export_results(audit,output_dir,allow_partial_export=False):
    """Existing output directories are ALWAYS refused; invalid inputs yield diagnostics only."""
    requested=Path(output_dir)
    require(not requested.exists() and not requested.is_symlink(), 'Refusing existing output directory or symlink')
    output=requested.resolve(); root=Path(audit['input']).resolve()
    require(not output.exists(),'Refusing existing output directory')
    require(not output.is_relative_to(root) and not root.is_relative_to(output),'Output must be isolated from input')
    legacy=Path(__file__).resolve().parents[1]/'member5_outputs'/'formal_results'
    require(not output.is_relative_to(legacy),'Refusing legacy MALG formal_results output')
    require(output.parent.is_dir(),'Output parent must already exist')
    include=audit['validation']=='VERIFIED' and (audit['status']=='COMPLETE_VERIFIED' or allow_partial_export)
    report=['# MDB-Net verification','',f"Status: **{audit['status']}**",f"Metric validation: {audit['validation']}",'',
        'VERIFIED = directly checked; PARTIALLY VERIFIED = evidence gaps; NOT VERIFIED = no evidence; INVALID = rejected.','',
        'Input-derived metrics do not prove executed formal training. Synthetic tests are not real experiments.','','## Errors']
    report+=['- '+s for s in audit['errors']]+['','## Limitations']+['- '+s for s in audit['limitations']]
    report+=['','## Warnings']+['- '+s for s in sorted(set(audit['warnings']))]
    for r in audit['runs']:
        h=r['history']; report+=['',f"Seed {r['seed']}: head={h['head_epochs']}, fine-tune={h['finetune_epochs']}, inferred best epoch={h['inferred_best_epoch']}; {h['checkpoint_status']}"]
    report+=['','Segmentation: pseudo-mask agreement, NOT hand-annotated truth.',f'Metric tables exported: {include}. Partial exports are descriptive, NOT verified formal results.']
    provenance={k:v for k,v in audit.items() if k!='runs'}
    provenance['runs']=[dict(seed=r['seed'],history=r['history'],predictions_source=r['predictions_source']) for r in audit['runs']]
    payloads={'verification_report.md':'\n'.join(report)+'\n',
        'experiment_provenance.json':json.dumps(provenance,indent=2,ensure_ascii=False,allow_nan=False)}
    if include:
        individual,summary,cases=[],[],[]
        for r in audit['runs']:
            for sc,scenario in SCENARIOS.items():
                m=r['metrics'][sc]
                individual.append(dict(model='MDB-Net',seed=r['seed'],scenario=scenario,n=m['n'],accuracy=m['acc'],macro_f1=m['macro_f1'],verification_status=audit['status']))
                payloads[f"confusion_matrices/seed{r['seed']}_{sc}.csv"]=csv_text([{'true_class':c,**dict(zip(CLASSES,m['cm'][i]))} for i,c in enumerate(CLASSES)])
            for row in r['rows']:
                for scenario in (SCENARIOS['a_white' if row['background']=='white' else 'b_field'],'Mixed'):
                    cases.append(dict(seed=r['seed'],scenario=scenario,image=row['image'],true=CLASSES[row['true']],predicted=CLASSES[row['predicted']],correct=row['correct'],confidence=row['confidence'],predictions_source=r['predictions_source']))
        for sc,scenario in SCENARIOS.items():
            s=audit['summary'][sc]
            summary.append(dict(model='MDB-Net',scenario=scenario,number_of_runs=5,accuracy_mean=s['acc_mean'],accuracy_std=s['acc_std'],macro_f1_mean=s['macro_f1_mean'],macro_f1_std=s['macro_f1_std'],std_ddof=1,verification_status=audit['status']))
            payloads[f'confusion_matrices/sum_{sc}.csv']=csv_text([{'true_class':c,**dict(zip(CLASSES,s['cm_sum'][i]))} for i,c in enumerate(CLASSES)])
        payloads['individual_runs.csv']=csv_text(individual); payloads['summary_results.csv']=csv_text(summary)
        payloads['member6_prediction_cases.csv']=csv_text(cases)
        lines=[r'\begin{tabular}{lcc}',r'\hline',r'Scenario & Accuracy & Macro-F1 \\',r'\hline']
        for s in summary:
            lines.append(f"{s['scenario']} & {s['accuracy_mean']:.4f} $\\pm$ {s['accuracy_std']:.4f} & {s['macro_f1_mean']:.4f} $\\pm$ {s['macro_f1_std']:.4f} "+r'\\')
        payloads['result_table.tex']='% '+audit['status']+'; sample SD ddof=1; MDB-Net; partial=descriptive only\n'+'\n'.join(lines+[r'\hline',r'\end{tabular}'])+'\n'
    # Recheck recorded bytes before publication; never bless inputs changed after audit.
    for name, expected in audit['input_hashes'].items():
        require(Path(name).is_file() and file_hash(name)==expected, f'Input/evidence changed after audit: {name}')
    # Validate/serialize everything before creating any output. Never clean on failure.
    output.mkdir()
    for name,content in payloads.items():
        p=output/name; p.parent.mkdir(exist_ok=True)
        with p.open('x',encoding='utf-8',newline='') as f:
            f.write(content)
    return list(payloads)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path,help='MDB-Net result directory, read only')
    p.add_argument('--output',required=True,type=Path,help='New isolated directory; must not exist')
    p.add_argument('--expected-seeds',required=True,nargs=5,type=int,help='Five external approved seeds; no defaults')
    p.add_argument('--approved-config',type=Path,help='External approval JSON schema v1')
    p.add_argument('--evidence-manifest',type=Path,help='Original runner evidence JSON schema v1')
    p.add_argument('--allow-partial-export',action='store_true',help='Descriptive tables despite evidence gaps; NEVER upgrades status')
    p.add_argument('--require-complete',action='store_true',help='Exit 2 unless full evidence verification succeeds')
    a=p.parse_args(argv)
    try:
        audit=audit_experiment(a.input,a.expected_seeds,a.approved_config,a.evidence_manifest)
        export_results(audit,a.output,a.allow_partial_export)
    except (VerificationError,OSError,ValueError) as exc:
        print(f'MDB verification/export rejected: {exc}',file=sys.stderr); return 1
    print(f"MDB-Net: {audit['status']}; metric validation: {audit['validation']}; output: {a.output}")
    for msg in audit['errors']+audit['limitations']:
        print(msg)
    if audit['status']=='INVALID':
        return 1
    return 2 if a.require_complete and audit['status']!='COMPLETE_VERIFIED' else 0

if __name__=='__main__':
    raise SystemExit(main())
