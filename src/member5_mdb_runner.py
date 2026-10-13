"""Member 5 isolated MDB workspace/runner. Default CLI operations are read-only.

No training or mask algorithm is implemented here. Only explicit execute after ALL
approvals/preflights may launch the unchanged shared train_multitask.py once.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import contextlib
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import uuid
import numpy as np
from . import member5_mdb_results as results
from . import member5_mdb_live_monitor as live_monitor

PROJECT = Path(__file__).resolve().parents[1]
# Public path adaptation only; historical executed-tool hashes remain in provenance.
WORKSPACE = Path(os.environ.get('MEMBER5_WORKSPACE', str(PROJECT.parent/'member5_workspace'))).absolute()
SOURCE = Path(os.environ.get('MEMBER5_DATA_SOURCE', str(PROJECT.parent/'Dhan-Shomadhan'))).absolute()
SELECTED_CONFIG = Path(os.environ.get('MEMBER5_SELECTED_CONFIG', str(PROJECT/'member5/scientific_configuration.json'))).absolute()
COUNTS = {'total':1106,'white':769,'field':337}
CODE = ('train_multitask.py','train_resnet18.py','make_masks.py','seg_leaf.py')
WEIGHT_VERSION = 'ResNet18_Weights.IMAGENET1K_V1'
EXTENSIONS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
Error = results.VerificationError
require = results.require


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def write_new(path,value):
    path=Path(path)
    with path.open('xb') as f:
        f.write(encoded(value)); f.flush(); os.fsync(f.fileno())


def safe_path(path):
    """Reject lexical traversal, all ancestor symlinks and Windows reparse points."""
    p=Path(path).expanduser()
    require(p.is_absolute(),'Absolute path required')
    require('..' not in p.parts,'Path traversal rejected')
    for part in p.parts[1:]:
        require(not part.rstrip(' .')!=part and ':' not in part,'Unsafe Windows path component')
    for node in [p,*p.parents]:
        if node.exists() or node.is_symlink():
            st=node.lstat()
            require(not stat.S_ISLNK(st.st_mode) and not (getattr(st,'st_file_attributes',0)&0x400),f'Link/reparse redirection rejected: {node}')
    return p.resolve()


def safe_name(name):
    require(isinstance(name,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',name),'Cohort/tag: safe single name required')
    require(name.upper() not in {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))},'Reserved Windows name')
    return name


def ref(path):
    p=safe_path(Path(path).absolute())
    require(p.is_file(),'Evidence file missing: '+str(p))
    return {'path':str(p),'sha256':results.file_hash(p)}


def pure_data_functions():
    """Extract EXACT shared discovery/split AST; do not import its training module."""
    source=PROJECT/'src/train_resnet18.py'
    names={'CLASS_KEYS','CANON','CLASS_NAMES','DOMAINS','IMG_EXT','match_class','discover_samples','make_split'}
    tree=ast.parse(source.read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name in names) or
        (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets))]
    from sklearn.model_selection import train_test_split
    ns=dict(Path=Path,np=np,re=re,difflib=difflib,Counter=Counter,train_test_split=train_test_split)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    require(all(k in ns for k in names),'Shared discovery/split AST contract changed; re-audit required')
    require(tuple(ns['CLASS_NAMES'])==results.CLASSES,'Shared class order changed')
    return ns


def raw_images(root):
    root=safe_path(root); require(root.is_dir(),'Dataset does not exist: '+str(root))
    images=[]
    for folder,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if not d.startswith('_')]
        for d in dirs:
            safe_path(Path(folder)/d)
        for name in files:
            p=Path(folder)/name
            if not name.startswith('_') and p.suffix.lower() in EXTENSIONS:
                safe_path(p); images.append(p)
    return sorted(images)


def dataset_manifest(root,expected_counts=COUNTS):
    root=safe_path(root); images=raw_images(root); ns=pure_data_functions()
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            samples=ns['discover_samples'](root)
        except SystemExit as exc:
            raise Error(str(exc)) from exc
    require({p.relative_to(root) for p in images}=={s[0] for s in samples},'Unknown/skipped raw images detected; file membership differs from discovery')
    records=[]; seen=set(); paths=set(); cache_keys=set()
    for rel,c,d in samples:
        image=results.normalize_path(rel.as_posix()); p=safe_path(root/rel)
        require(image.casefold() not in paths,'Windows case-alias duplicate image')
        cache_key=rel.with_suffix('.jpg').as_posix().casefold()
        require(cache_key not in cache_keys,'Same-stem cache collision')
        paths.add(image.casefold()); cache_keys.add(cache_key)
        before=p.stat(); sha=results.file_hash(p); after=p.stat()
        require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Source changed while hashing')
        require(sha not in seen,'Identical SHA-256 images require group/deduplication policy')
        seen.add(sha)
        records.append(dict(image=image,label=results.CLASSES[c],background=ns['DOMAINS'][d],sha256=sha,file_size=after.st_size))
    counts={'total':len(records),**{b:sum(r['background']==b for r in records) for b in ('white','field')}}
    require(expected_counts is None or counts==expected_counts,f'Dataset count conflict: {counts} != {expected_counts}')
    return records


def compare_manifest(expected,actual):
    require(isinstance(expected,list) and isinstance(actual,list),'Manifest arrays required')
    left={r['image']:r for r in expected}; right={r['image']:r for r in actual}
    require(len(left)==len(expected) and len(right)==len(actual),'Duplicate manifest image')
    require(left.keys()==right.keys(),f'Missing/added images: missing={sorted(left.keys()-right.keys())}, added={sorted(right.keys()-left.keys())}')
    require(expected==actual,'Dataset hash/label/background/size/order changed')


def nearest_existing(path):
    p=path
    while not p.exists():
        p=p.parent
    return p


def prepare_plan(source,workspace=WORKSPACE,mode='formal',cohort_id=None,baseline=None,expected_counts=COUNTS):
    require(mode in ('pilot','formal'),'Mode must be pilot/formal')
    source=safe_path(source); workspace=safe_path(workspace)
    require(not(workspace.is_relative_to(source) or source.is_relative_to(workspace)),'Workspace must not contain or be inside original dataset')
    require(not(workspace.is_relative_to(PROJECT) or PROJECT.is_relative_to(workspace)),'Workspace must be outside project/old results')
    cohort_id=safe_name(cohort_id or ('mdb_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]))
    cohort=safe_path(workspace/mode/cohort_id)
    require(not cohort.exists(),'Cohort exists: never overwrite/reuse')
    manifest=dataset_manifest(source,expected_counts)
    if baseline:
        compare_manifest(results.read_json(baseline),manifest)
    size=sum(r['file_size'] for r in manifest)
    required=int(size*1.1)+64*1024**2
    free=shutil.disk_usage(nearest_existing(cohort)).free
    require(free>=required,f'Insufficient disk space: need {required}, available {free}')
    return dict(status='DRY RUN',mode=mode,cohort_id=cohort_id,source=str(source),workspace=str(workspace),
        cohort=str(cohort),data_root=str(cohort/'Dhan-Shomadhan'),manifest=manifest,manifest_sha256=digest(manifest),
        total_bytes=size,required_copy_bytes=required,free_bytes=free,
        note='Independent byte copies only; excludes _interim/cache/results. No files created in Dry Run.')


def prepare_data(plan):
    """Future explicit PrepareData only. NEVER invoked against real data in STEP 4."""
    source=safe_path(plan['source']); cohort=safe_path(plan['cohort']); target=safe_path(plan['data_root'])
    require(not cohort.exists() and target==cohort/'Dhan-Shomadhan','Unsafe/existing cohort')
    compare_manifest(plan['manifest'],dataset_manifest(source,None))
    free=shutil.disk_usage(nearest_existing(cohort)).free
    require(free>=plan['required_copy_bytes'],'Disk space changed before copy')
    # Exclusive leaf reservation. On any failure retain it; no deletion/rollback.
    cohort.parent.mkdir(parents=True,exist_ok=True); safe_path(cohort.parent); cohort.mkdir()
    target.mkdir(); (cohort/'evidence').mkdir(); (cohort/'logs').mkdir()
    try:
        for row in plan['manifest']:
            src=safe_path(source/row['image']); dst=target/row['image']
            dst.parent.mkdir(parents=True,exist_ok=True); safe_path(dst.parent)
            with src.open('rb') as a,dst.open('xb') as b:
                shutil.copyfileobj(a,b,1024*1024); b.flush(); os.fsync(b.fileno())
            require(dst.stat().st_nlink==1 and not os.path.samefile(src,dst),'Copy unexpectedly linked to source')
            require(results.file_hash(dst)==row['sha256'],'Copied image SHA mismatch')
        compare_manifest(plan['manifest'],dataset_manifest(target,None))
        compare_manifest(plan['manifest'],dataset_manifest(source,None))
        write_new(cohort/'evidence/dataset_manifest.json',plan['manifest'])
        marker={k:plan[k] for k in ('mode','cohort_id','source','workspace','cohort','data_root','manifest_sha256')}
        marker.update(schema_version=1,owner='Member5',model='MDB-Net',status='DATA_PREPARED',prepared_at=utc_now())
        write_new(cohort/'cohort.json',marker)
        return dict(status='DATA_PREPARED',cohort=str(cohort))
    except BaseException as exc:
        write_new(cohort/'evidence/prepare_failure.json',dict(status='FAILED',time=utc_now(),error=str(exc)))
        raise


def load_approved(path,mode='formal'):
    require(path is not None,'MISSING APPROVED CONFIG')
    raw=results.read_json(path)
    require(isinstance(raw,dict) and raw.get('schema_version')==1 and raw.get('status')=='APPROVED' and
        raw.get('model')=='MDB-Net' and raw.get('purpose')==mode,'Approved schema v1/model/purpose mismatch')
    require(isinstance(raw.get('approval_reference'),str) and raw['approval_reference'].strip(),'Approval reference missing')
    require(raw.get('protocol')==results.PROTOCOL,'Approved split/evaluation protocol incompatible with current shared code')
    args_raw=dict(raw.get('args',{}))
    # Same schema/keys for Pilot, purpose remains pilot and is NEVER formal-verified.
    # Only neutralize the naming restriction for type checking, not any science value.
    original_tag=args_raw.get('out_name')
    if mode=='pilot':
        args_raw['out_name']='pilot_typecheck'
        args_raw['out_name']='typecheck'  # neutral token; no command uses this value
    args=results.parse_args_record(args_raw)
    if mode=='pilot':
        args['out_name']=original_tag
    seeds=results.seeds_checked(raw.get('expected_seeds'))
    require(seeds==args['seeds']==[0,1,2,3,4],'Formal seeds must be exactly ordered 0,1,2,3,4')
    selected=results.read_json(SELECTED_CONFIG)
    require(raw.get('approval_source')=='EXPLICIT_USER_INSTRUCTION','User approval provenance required; no inferred member signatures')
    require(not raw.get('synthetic',False),'Synthetic fixture cannot authorize formal execution')
    require(all(args[k]==selected['args'][k] for k in results.SCIENCE_KEYS),'Selected main science configuration conflict')
    require(args['workers']==2 and args['save_models'],'Runtime configuration conflict')
    verify_code(selected)
    safe_name(args['out_name']); require(args['save_models'],'save_models must be explicitly approved true')
    require(not args['from_scratch'],'MDB pretraining required: from_scratch not approved for this workflow')
    require(not args['allow_missing'],'Incomplete class/background dataset forbidden')
    for key in ('data_root','weights_path'):
        require(args[key] is not None,f'MISSING {key.upper()}')
        safe_path(args[key])
    if raw.get('recovery_reason') is not None:
        require(raw['recovery_reason']=='ENGINEERING_MONITOR_FALSE_POSITIVE','Unknown recovery cause')
        verify_monitor_tools(raw)
    return raw,args


def verify_monitor_tools(approval):
    """Tool provenance gate; separate from immutable scientific source hashes."""
    expected=('src/member5_mdb_runner.py','src/member5_mdb_live_monitor.py','scripts/member5_mdb_formal_run.ps1')
    tools=approval.get('monitoring_tools')
    require(isinstance(tools,dict) and set(tools)==set(expected),'Missing monitoring tool bindings')
    for name in expected:
        require(tools[name]==ref(PROJECT/name),'Monitoring tool changed: '+name)
    record=approval.get('monitoring_validation')
    require(isinstance(record,dict) and ref(record['path'])==record,'Monitor validation receipt changed')
    verified=results.read_json(record['path'])
    require(verified.get('status')=='PASS' and verified.get('monitor_version')==live_monitor.VERSION,
            'Monitoring validation not passed')
    require(verified.get('tools')==tools and verified.get('historical_replay',{}).get('action')=='CONTINUE'
            and verified['historical_replay'].get('batches')==284 and verified['historical_replay'].get('skips')==1,
            'Monitor validation bound to wrong tools/replay')
    return True


def split_evidence(manifest,args):
    ns=pure_data_functions(); y=np.array([results.CLASSES.index(r['label']) for r in manifest]); d=np.array([0 if r['background']=='white' else 1 for r in manifest])
    splits={}; fingerprints=set()
    for seed in results.seeds_checked(args['seeds']):
        ids=ns['make_split'](y,d,seed); replay=ns['make_split'](y,d,seed)
        require(all(np.array_equal(a,b) for a,b in zip(ids,replay)),'Same seed did not replay identically')
        parts={k:[manifest[i]['image'] for i in idx] for k,idx in zip(('train','val','test'),ids)}
        sets=[set(parts[k]) for k in ('train','val','test')]
        require(not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]),'Split overlap')
        tr=ids[0]
        parts['training_images']=[manifest[i]['image'] for i in tr if args['field_train']=='cls' or d[i]==0]
        parts['background_swap_images']=[manifest[i]['image'] for i in tr if d[i]==1] if args['bg_swap']>0 else []
        fingerprints.add(tuple(tuple(parts[k]) for k in ('train','val','test'))); splits[seed]=parts
    require(len(fingerprints)==5,'Five distinct approved splits required')
    return splits

def validate_qc_policy(path):
    require(path is not None,'MISSING QC APPROVAL')
    policy=results.read_json(path)
    require(isinstance(policy,dict) and policy.get('status')=='APPROVED' and policy.get('approval_reference'),'QC approval missing reference/status')
    require(all(isinstance(policy.get(k),str) and policy[k].strip() for k in ('mask_qc_scope','test_exposure','near_duplicate_policy')),'QC policy fields missing')
    require(policy.get('qc_status')=='NO_EFFECT_ON_TRAINING_ELIGIBILITY_VERIFIED','QC eligibility gate not released')
    require(policy.get('amp_status')=='ACCEPTABLE_WITH_MONITORING','AMP gate not released')
    require(policy.get('monitoring_required') is True,'AMP monitoring must be required')
    require(policy.get('approval_source')=='EXPLICIT_USER_INSTRUCTION','Policy approval provenance required')
    require(policy.get('implementation_sha256')==results.file_hash(PROJECT/'src/make_masks.py'),'QC implementation hash conflict')
    return policy


def weights_check(path,approved_sha=None,version=None,load=False):
    if path is None or not Path(path).is_file():
        return dict(status='MISSING WEIGHTS',load_status='NOT VERIFIED')
    p=safe_path(path); actual=results.file_hash(p)
    require(version==WEIGHT_VERSION,'Wrong/missing pretrained ResNet18 version')
    require(approved_sha==actual,'Pretrained weight SHA differs from approval')
    require(actual=='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec','Not the published ResNet18 IMAGENET1K_V1 file (MobileNet/U2-Net forbidden)')
    status='NOT VERIFIED (CPU load deferred until explicit Execute)'
    if load:
        import torch
        from torchvision import models
        state=torch.load(p,map_location='cpu',weights_only=True)
        with torch.random.fork_rng(devices=[]):
            model=models.resnet18(weights=None)  # NEVER downloads; preserve CPU RNG
            model.load_state_dict(state,strict=True)
        del model,state
        status='VERIFIED (strict CPU ResNet18 state-dict load, no training/forward)'
    return dict(status='WEIGHT FILE VERIFIED',path=str(p),sha256=actual,version=version,load_status=status)


def environment_info(probe_cuda=True):
    out={'python':sys.version.split()[0],'executable':sys.executable,'platform':sys.platform,'device':'NOT VERIFIED'}
    for name in ('torch','torchvision','numpy','scikit-learn','opencv-python-headless','matplotlib'):
        try:
            out[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name]='MISSING'
    if probe_cuda:
        import torch
        out.update(cuda_available=torch.cuda.is_available(),cuda_build=str(torch.version.cuda))
        out['device']=torch.cuda.get_device_name(0) if out['cuda_available'] else 'CPU ONLY'
    return out


def isolated_identity(data_root,workspace,source,mode,expected_counts=COUNTS):
    root=safe_path(data_root); workspace=safe_path(workspace); source=safe_path(source)
    require(root.is_dir(),'MISSING DATA')
    require(root.is_relative_to(workspace/mode) and root.name=='Dhan-Shomadhan','Data root outside isolated mode/cohort')
    cohort=root.parent; require(cohort.parent==workspace/mode,'Unexpected nested cohort layout')
    safe_name(cohort.name)
    require(not(root.is_relative_to(source) or source.is_relative_to(root)),'Cannot write original dataset')
    require(not(root.is_relative_to(PROJECT) or PROJECT.is_relative_to(root)),'Cannot use project/teammate run directory')
    marker=results.read_json(cohort/'cohort.json')
    require(isinstance(marker,dict) and marker.get('schema_version')==1 and marker.get('owner')=='Member5' and
        marker.get('model')=='MDB-Net' and marker.get('status')=='DATA_PREPARED' and marker.get('mode')==mode,'Missing/invalid prepared cohort identity')
    for k,p in (('source',source),('workspace',workspace),('cohort',cohort),('data_root',root)):
        require(safe_path(marker.get(k,''))==p,'Cohort marker path conflict: '+k)
    manifest=dataset_manifest(root,expected_counts)
    frozen=results.read_json(cohort/'evidence/dataset_manifest.json'); compare_manifest(frozen,manifest)
    require(digest(manifest)==marker.get('manifest_sha256'),'Prepared dataset hash conflict')
    compare_manifest(manifest,dataset_manifest(source,expected_counts))
    for row in manifest:
        p=root/row['image']
        require(p.stat().st_nlink==1 and not os.path.samefile(p,source/row['image']),'Hardlink/original image alias forbidden')
    return cohort,manifest


def preprocessing_records(data_root,manifest,args,qc_path):
    """Read/freeze existing masks/cache/QC only. NEVER invokes their generators."""
    import csv
    import cv2
    root=safe_path(data_root); validate_qc_policy(qc_path)
    qc_dir=root/'_interim/qc'; index_path=qc_dir/'mask_index.csv'; report_path=qc_dir/'qc_report.csv'
    require(index_path.is_file() and report_path.is_file(),'MISSING MASKS/INDEX/QC')
    exclude=Path(args['exclude_file']) if args['exclude_file'] else qc_dir/'exclude.txt'
    exclude=safe_path(exclude); require(exclude.is_relative_to(root) and exclude.is_file(),'MISSING ISOLATED EXCLUDE FILE (freeze even an empty approved file)')
    images={r['image']:r for r in manifest}; index={}; statuses={'ok','review','fail','excluded'}
    with index_path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        require(set(('image','cache_image','mask','class','status'))<=set(reader.fieldnames or []),'Mask index columns invalid')
        for row in reader:
            image=results.normalize_path(row['image'])
            require(image in images and image not in index and images[image]['background']=='white','Index duplicate/non-white/unknown image')
            require(row['image']==str(Path(image)),'Index path separator differs from actual shared loader on this OS')
            require(row['class']==images[image]['label'] and row['status'] in statuses,'Mask index label/status conflict')
            index[image]=row
    require(set(index)=={r['image'] for r in manifest if r['background']=='white'},'Incomplete White mask index (possible --only replacement)')
    qc={}
    with report_path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        require(set(('image','cls','status'))<=set(reader.fieldnames or []),'QC report columns invalid')
        for row in reader:
            image=results.normalize_path(row['image'])
            require(image in index and image not in qc and row['cls']==index[image]['class'] and row['status']==index[image]['status'],'Index/QC conflict')
            qc[image]=row
    require(set(qc)==set(index),'Index/QC coverage conflict')
    excluded=set()
    for line in exclude.read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            image=results.normalize_path(line.strip())
            require(image in images and line.strip()==str(Path(image)),'Exclude path incompatible with shared loader')
            excluded.add(image)
    rows=[]; trusted=set()
    for item in manifest:
        image=item['image']; cache=root/'_interim'/f"cache_{args['cache_size']}"/Path(image).with_suffix('.jpg')
        cache=safe_path(cache); require(cache.stat().st_nlink==1,'Linked cache forbidden'); require(cache.is_file(),'MISSING CACHE: '+image)
        rgb=cv2.imread(str(cache)); require(rgb is not None,'Unreadable cache: '+image)
        h,w=rgb.shape[:2]; require(max(h,w)<=args['cache_size'],'Cache geometry exceeds approved cache size')
        record=dict(image=image,cache=ref(cache),mask=None,status='none')
        if image in index:
            row=index[image]
            require(safe_path(root/results.normalize_path(row['cache_image']))==cache,'Index cache path conflict')
            mask=safe_path(root/results.normalize_path(row['mask']))
            require(mask.is_relative_to(root/'_interim'/f"masks_{args['cache_size']}") and mask.is_file(),'MISSING/UNSAFE MASK: '+image)
            require(mask.stat().st_nlink==1,'Linked mask forbidden')
            m=cv2.imread(str(mask),cv2.IMREAD_GRAYSCALE)
            require(m is not None and m.shape==(h,w),'Mask/cache shape mismatch')
            require(set(np.unique(m).tolist())<={0,255},'Mask is not binary 0/255')
            record.update(mask=ref(mask),status=row['status'])
            if row['status'] in args['mask_status'].split(',') and image not in excluded:
                trusted.add(image)
        rows.append(record)
    require(trusted,'No trusted masks; segmentation/background-swap prerequisites not met')
    prep={'cache_manifest':None,'mask_index':ref(index_path),'qc_report':ref(report_path),'exclude_file':ref(exclude),'policy':ref(qc_path)}
    return rows,prep,trusted


def build_command(args,python=sys.executable):
    """Argument vector, shell=False. Every science parameter comes from approved args."""
    command=[str(Path(python).resolve()),'-B','-u',str(PROJECT/'src/train_multitask.py')]
    for key in results.ARG_KEYS:
        value=args[key]
        if key in results.BOOL_KEYS:
            if value:
                command.append('--'+key)
        elif value is not None:
            command.append('--'+key)
            command.extend(map(str,value)) if key=='seeds' else command.append(str(value))
    return command


def run_preflight(approved_config=None,qc_approval=None,workspace=WORKSPACE,source=SOURCE,mode='formal',expected_counts=COUNTS):
    plan=dict(status='BLOCKED',execution='NOT STARTED',verification='NOT VERIFIED',mode=mode,blockers=[],checks={})
    checks=plan['checks']; errors=plan['blockers']; approval=None; args=None
    try:
        approval,args=load_approved(approved_config,mode)
        plan.update(approved_config=str(safe_path(approved_config)),approval=approval,args=args,command=build_command(args))
        checks['approval']='APPROVED RECORD PROVIDED (identity not independently authenticated)'
    except (Error,OSError,ValueError,KeyError,TypeError) as exc:
        errors.append(str(exc)); checks['approval']='MISSING/INVALID'
    try:
        validate_qc_policy(qc_approval); checks['qc_policy']='APPROVED RISK POLICY PROVIDED'
    except (Error,OSError,ValueError,KeyError,TypeError) as exc:
        errors.append(str(exc)); checks['qc_policy']='MISSING/INVALID'
    env=environment_info(); plan['environment']=env
    if not env.get('cuda_available'):
        errors.append('CUDA NOT AVAILABLE')
    default_weight=Path.home()/'.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
    try:
        weight=weights_check(args['weights_path'] if args else default_weight,
            approval.get('pretrained_weights_sha256') if approval else None,
            approval.get('pretrained_weights_version') if approval else None)
        checks['weights']=weight
        if weight['status']=='MISSING WEIGHTS':
            errors.append('MISSING WEIGHTS')
    except (Error,OSError,ValueError,KeyError,TypeError) as exc:
        errors.append(str(exc)); checks['weights']={'status':'INVALID','load_status':'NOT VERIFIED'}
    if args:
        try:
            cohort,manifest=isolated_identity(args['data_root'],workspace,source,mode,expected_counts)
            require(not (cohort/'execution_lock.json').exists(),'Existing execution reservation: new cohort required')
            require(mode=='formal' and 'pilot' not in str(args['data_root']).lower(),'Pilot cannot be formal')
            require(args['out_name'].startswith('member5_mdb_formal_'),'Member5 formal tag required')
            tag=safe_name(args['out_name']); output=safe_path(Path(args['data_root'])/'_interim/runs'/tag)
            require(not output.exists(),'OUTPUT EXISTS: new isolated cohort required')
            require(digest(manifest)==approval.get('dataset_manifest_sha256'),'Approved dataset manifest hash differs')
            source_hashes={name:results.file_hash(PROJECT/'src'/name) for name in CODE}
            require(approval.get('source_code_sha256')==source_hashes,'Approved source-code hashes missing/changed')
            splits=split_evidence(manifest,args)
            cache,prep,trusted=preprocessing_records(args['data_root'],manifest,args,qc_approval)
            for seed,split in splits.items():
                require(trusted & set(split['training_images']),f'seed {seed}: no trusted training masks')
            prep_hashes={k:ref['sha256'] for k,ref in prep.items() if ref is not None}
            prep_hashes['cache_manifest']=digest(cache)
            require(approval.get('preprocessing_sha256')==prep_hashes,'Approved preprocessing hashes missing/changed')
            plan.update(cohort=str(cohort),output=str(output),manifest=manifest,splits=splits,cache_manifest=cache,
                preprocessing=prep,source_code_sha256=source_hashes)
            checks['data_split_masks']='VERIFIED BEFORE EXECUTION (not training history)'
        except (Error,OSError,ValueError,KeyError,TypeError) as exc:
            errors.append(str(exc)); checks['data_split_masks']='BLOCKED'
    else:
        checks['data_split_masks']='NOT VERIFIED (no approved isolated data root/seeds)'
        errors.append('MISSING ISOLATED DATA / MASKS / INDEX; no approved root to preflight')
    if not errors:
        plan['status']='DRY RUN READY'
    return plan

def capture_process(command,log_path,on_start=lambda seed,time:None,on_end=lambda seed,time:None):
    """Future execution log capture; tests replace Popen with a fake, never a trainer."""
    env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
    active=None; observed=[]; process=None
    with Path(log_path).open('x',encoding='utf-8',newline='') as log:
        log.write('RUNNER START '+utc_now()+'\nCOMMAND '+json.dumps(command)+'\n'); log.flush(); os.fsync(log.fileno())
        try:
            process=subprocess.Popen(command,cwd=PROJECT,env=env,shell=False,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding='utf-8',errors='replace',bufsize=1)
            for line in process.stdout:
                log.write(line); log.flush(); os.fsync(log.fileno())
                match=re.match(r'^\[seed (\d+)\]',line)
                if match:
                    now=utc_now()
                    if active is not None:
                        on_end(active,now)
                    active=int(match[1]); observed.append(active); on_start(active,now)
            code=process.wait()
            if code==0 and active is not None:
                on_end(active,utc_now())
            log.write(f'RUNNER EXIT {code} {utc_now()}\n'); log.flush(); os.fsync(log.fileno())
            return code,observed
        except BaseException as exc:
            log.write('RUNNER ERROR '+repr(exc)+'\n'); log.flush(); os.fsync(log.fileno())
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            raise


def execute_plan(plan,qc_approval,workspace,source,mode):
    """Explicit future Execute only. Rerun all gates; preserve any failed attempt."""
    require(plan['status']=='DRY RUN READY','BLOCKED: approvals/preflights cannot be bypassed by Execute')
    verify_monitor_tools(plan['approval'])
    fresh=run_preflight(plan['approved_config'],qc_approval,workspace,source,mode)
    require(fresh['status']=='DRY RUN READY','Preflight changed before Execute: '+str(fresh['blockers']))
    require(encoded(plan)==encoded(fresh),'Input/config/environment changed after Dry Run; new plan required')
    args=plan['args']; approval=plan['approval']; cohort=safe_path(plan['cohort']); output=safe_path(plan['output'])
    weights_check(args['weights_path'],approval['pretrained_weights_sha256'],approval['pretrained_weights_version'],load=True)
    require(not output.exists(),'OUTPUT EXISTS: refuse overwrite')
    lock=cohort/'execution_lock.json'
    write_new(lock,dict(status='RESERVED',time=utc_now(),approval=ref(plan['approved_config'])))
    attempt=cohort/'evidence'/('attempt_'+uuid.uuid4().hex); attempt.mkdir()
    (attempt/'source_code').mkdir(); (attempt/'runs').mkdir(); (attempt/'logs').mkdir()
    frozen={}; records={}; completed=[]
    try:
        output.parent.mkdir(parents=True,exist_ok=True); safe_path(output.parent); output.mkdir()
        for name in CODE:
            p=attempt/'source_code'/name
            with (PROJECT/'src'/name).open('rb') as a,p.open('xb') as b:
                shutil.copyfileobj(a,b)
            require(results.file_hash(p)==plan['source_code_sha256'][name],'Source changed while freezing')
            frozen[name]=ref(p)
        cache_path=attempt/'cache_manifest.json'; write_new(cache_path,plan['cache_manifest'])
        prep=dict(plan['preprocessing']); prep['cache_manifest']=ref(cache_path)
        require({k:v['sha256'] for k,v in prep.items()}==approval['preprocessing_sha256'],'Frozen preprocessing hash mismatch')
        manifest_path=cohort/'evidence/dataset_manifest.json'
        environment_path=attempt/'environment.json'; write_new(environment_path,plan['environment'])
        write_new(attempt/'approved_config_snapshot.json',approval)
        for seed in args['seeds']:
            config_path=attempt/'runs'/f'seed{seed}_config.json'
            split_path=attempt/'runs'/f'seed{seed}_split.json'
            write_new(config_path,dict(seed=seed,model='MDB-Net',purpose=mode,args=args,protocol=results.PROTOCOL,
                recorded_at=utc_now(),phase='before_process_launch'))
            write_new(split_path,plan['splits'][seed])
            records[seed]=dict(seed=seed,model='MDB-Net',purpose=mode,config=ref(config_path),split=ref(split_path),environment=ref(environment_path),
                timestamp_basis='Observed unbuffered shared seed-start line to next seed boundary/process completion')
        write_new(attempt/'pre_run_records.json',dict(status='PREPARED_NOT_EXECUTED',schema_version=1,purpose=mode,
            dataset_manifest=ref(manifest_path),source_code=frozen,preprocessing=prep,pretrained_weights=ref(args['weights_path']),
            runs=list(records.values()),command=plan['command'],time=utc_now()))
        log_path=attempt/'logs/training.log'
        def start(seed,time):
            require(seed in records and 'started_at' not in records[seed],'Unexpected/duplicate seed marker')
            records[seed]['started_at']=time
        def finish(seed,time):
            record=records[seed]; rp=output/f'seed{seed}_result.json'; pp=output/f'seed{seed}_predictions.csv'; cp=output/f'seed{seed}_model.pt'
            require(all(p.is_file() for p in (rp,pp,cp)),'Seed incomplete: result/predictions/checkpoint missing')
            result=results.read_json(rp); history=results.verify_history(result,args,[])
            import torch
            state=torch.load(cp,map_location='cpu',weights_only=True)
            require(all(bool(torch.isfinite(t).all()) for t in state.values() if t.is_floating_point()),'Nonfinite saved checkpoint')
            del state
            record.update(completed_at=time,result=ref(rp),predictions=ref(pp),checkpoint={**ref(cp),'stage':2,
                'epoch':history['inferred_best_epoch'],'selection':results.PROTOCOL['selection'],
                'epoch_basis':'Inferred from source selection rule/history, file hash observed after write; no model deserialization'})
            completed.append(seed)
            write_new(attempt/'runs'/f'seed{seed}_observed_completion.json',record)
        write_new(attempt/'monitoring_tool_bindings.json',approval['monitoring_tools'])
        launch=[sys.executable,'-B','-u','-m','src.member5_mdb_runner','observe','--attempt',str(attempt)]
        write_new(attempt/'launch.json',dict(actual_command=launch,logical_shared_command=plan['command'],runner=ref(Path(__file__)),started_at=utc_now()))
        code,observed=capture_process(launch,log_path,start,finish)
        write_new(attempt/'process_exit.json',dict(exit_code=code,completed_at=utc_now()))
        require(monitoring_ok(attempt/'amp_events.jsonl'),'Numerical monitoring requires review; all evidence retained')
        for record in records.values():
            if 'started_at' in record:
                record['log']=ref(log_path)
        require(code==0,f'Training process failed with exit code {code}; all outputs retained')
        require(observed==args['seeds'] and completed==args['seeds'],'Missing/out-of-order actual seed observations')
        require(results.parse_args_record(results.read_json(output/'args.json'))==args,'Actual shared args.json differs from frozen approval')
        compare_manifest(plan['manifest'],dataset_manifest(args['data_root']))
        compare_manifest(plan['manifest'],dataset_manifest(source))
        require({n:results.file_hash(PROJECT/'src'/n) for n in CODE}==plan['source_code_sha256'],'Shared code changed during execution')
        current_cache,current_prep,_=preprocessing_records(args['data_root'],plan['manifest'],args,qc_approval)
        require(digest(current_cache)==digest(plan['cache_manifest']) and current_prep==plan['preprocessing'],'Mask/cache/QC changed during execution')
        require(ref(args['weights_path'])['sha256']==approval['pretrained_weights_sha256'],'Weight changed during execution')
        evidence_path=attempt/'evidence_manifest.json'
        write_new(evidence_path,dict(schema_version=1,model='MDB-Net',purpose=mode,dataset_manifest=ref(manifest_path),
            source_code=frozen,preprocessing=prep,pretrained_weights=ref(args['weights_path']),runs=list(records.values()),
            execution_status='PROCESS_COMPLETED',process_exit_code=code,post_run_verification_at=utc_now()))
        if mode=='formal':
            verification=results.audit_experiment(output,args['seeds'],attempt/'approved_config_snapshot.json',evidence_path)
            results.export_results(verification,attempt/'aggregation')
            require(verification['status']=='COMPLETE_VERIFIED','Post-run verification failed: '+str(verification['errors']))
            status='COMPLETE_VERIFIED (external recorded-artifact scope)'
        else:
            status='PILOT COMPLETED; NOT FORMAL / NOT COMPLETE_VERIFIED'
        write_new(attempt/'success.json',dict(status=status,time=utc_now(),exit_code=code,technical_execution='SUCCESS',metrics_verification=verification['status'],methodology='PASS_WITH_LIMITATIONS',limitations=['Near-duplicate/same-leaf grouping NOT VERIFIED','Historical test exposure','Pseudo-mask agreement is not manual segmentation truth']))
        return dict(status=status,evidence=str(evidence_path),output=str(output))
    except BaseException as exc:
        write_new(attempt/'failure.json',dict(status='FAILED_NOT_COMPLETE_VERIFIED',time=utc_now(),error=str(exc),
            completed_seeds=completed,runs=list(records.values()),note='Preserved; no resume/cleanup/overwrite. Use a new isolated cohort.'))
        raise


def run_request(approved_config=None,qc_approval=None,workspace=WORKSPACE,source=SOURCE,mode='formal',execute=False):
    plan=run_preflight(approved_config,qc_approval,workspace,source,mode)
    if not execute or plan['status']!='DRY RUN READY':
        return plan
    return execute_plan(plan,qc_approval,workspace,source,mode)


def public_plan(plan):
    hidden={'manifest','splits','cache_manifest','approval','preprocessing'}
    out={k:v for k,v in plan.items() if k not in hidden}
    if 'manifest' in plan:
        out['dataset_count']=len(plan['manifest']); out['dataset_manifest_sha256']=digest(plan['manifest'])
    if 'splits' in plan:
        out['split_counts']={str(s):{k:len(v) for k,v in split.items()} for s,split in plan['splits'].items()}
    return out



def verify_code(selected):
    def git(*args):
        return subprocess.check_output(['git','-c','safe.directory='+PROJECT.as_posix(),*args],cwd=PROJECT,text=True).strip()
    require(git('branch','--show-current')==selected.get('required_branch','feat/member5-formal-results'),'Wrong approved Member5 branch')
    require(git('rev-parse','HEAD')==selected['local_git_commit'],'Git HEAD changed; re-audit required')
    require(git('rev-parse','origin/main')==selected['source_main_commit'],'Pinned origin/main changed')
    require(all(results.file_hash(PROJECT/'src'/n)==h for n,h in selected['source_code_sha256'].items()),'Shared source hash changed')
    require(results.file_hash(PROJECT/'README.md')==selected['readme_sha256'],'README changed')
    return True


def selected_plan():
    """Read-only planning against frozen Pilot material; never authorizes training."""
    selected=results.read_json(SELECTED_CONFIG)
    verify_code(selected)
    args=dict(selected['args'],data_root=str(WORKSPACE/'formal/member5_mdb_formal_new/Dhan-Shomadhan'),
              weights_path=str(WORKSPACE/'weights/resnet18-f37072fd.pth'),out_name='member5_mdb_formal_new')
    weights=weights_check(args['weights_path'],'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec',WEIGHT_VERSION)
    require(weights['status']=='WEIGHT FILE VERIFIED','Missing official weight')
    require(args['seeds']==[0,1,2,3,4],'Selected seeds changed')
    require(not Path(args['data_root']).parent.exists(),'Planned formal cohort already exists')
    return dict(status='BLOCKED',execution='NOT STARTED',mode='formal',args=args,command=build_command(args),weights=weights,
        environment=environment_info(),blockers=['Independent formal cohort/data/cache/masks and schema-v1 frozen approval/policy not prepared',
        'Next explicit user execution authorization required'],note='No real dataset copy, mask generation or output creation')


def monitoring_ok(path):
    # Strict current-attempt stream verification for new observer telemetry;
    # legacy synthetic acceptance remains unchanged for existing regression tests.
    lines=Path(path).read_text(encoding='utf-8').splitlines()
    if lines and 'attempt_id' in json.loads(lines[0]):
        monitor=live_monitor.AMPMonitor(Path(path).parent.name,48)
        follower=live_monitor.JSONLFollower(path,monitor)
        decision=follower.poll(final=True)
        require(decision['action']=='CONTINUE' and monitor.ended and monitor.batches,
                'Numerical monitoring requires review: '+str(decision))
        return True
    events=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    end=[e for e in events if e.get('kind')=='end']
    batches=[e for e in events if e.get('kind')=='batch']
    require(end and end[-1].get('success') and batches,'Missing successful numerical monitoring')
    require(all(e['loss_finite'] and e['applied_gradients_finite'] and e['weights_finite'] for e in batches),'Serious numerical failure')
    # Review thresholds do not change calculations or classify one isolated skip as failure.
    streak=0; skipped=0
    for e in batches:
        skipped+=int(e['skipped']); streak=streak+1 if e['skipped'] else 0
        require(streak<3,'Three consecutive AMP skips require review')
    require(skipped/len(batches)<=.05,'AMP skipped-update rate >5% requires review')
    return True


def observe_training(attempt):
    """Future child only: passive wrappers call original operations exactly once.
    No changes to RNG, gradients, scaler policy, loss, model or optimizer arguments.
    """
    attempt=safe_path(attempt)
    require(attempt.is_relative_to(WORKSPACE/'formal') and attempt.parent.name=='evidence' and attempt.name.startswith('attempt_'),'Internal observer restricted to own formal attempt')
    config,args=load_approved(attempt/'approved_config_snapshot.json','formal')
    cohort=attempt.parent.parent
    require(Path(args['data_root']).parent==cohort and (cohort/'execution_lock.json').is_file(),'Execution reservation missing')
    pre=results.read_json(attempt/'pre_run_records.json')
    policy=pre['preprocessing']['policy']; require(ref(policy['path'])==policy,'Policy changed before launch')
    validate_qc_policy(policy['path'])
    weights_check(args['weights_path'],config['pretrained_weights_sha256'],config['pretrained_weights_version'],load=True)
    require(config.get('approval_source')=='EXPLICIT_USER_INSTRUCTION','No user-approved execution configuration')
    require(args['seeds']==[0,1,2,3,4],'Formal seeds required')
    sys.path.insert(0,str(PROJECT/'src'))
    import torch
    import train_multitask as mt
    verify_monitor_tools(config)
    monitor=live_monitor.AMPMonitor(attempt.name,48,head_epochs=args['head_epochs'],max_epochs=args['epochs'])
    context={'seed':None,'stage':None,'epoch':0,'batch':0,'updates':0,'model':None,'loss_finite':True,'loss_values':[]}
    event_seq=0
    original_epoch,original_run=mt.train_epoch,mt.run_one
    original_step=torch.optim.AdamW.step
    original_scaler_step=torch.amp.GradScaler.step
    original_update=torch.amp.GradScaler.update
    original_ce=torch.nn.CrossEntropyLoss.forward
    original_seg=mt.seg_loss_fn
    def finite(t): return bool(torch.isfinite(t).all().item())
    with (attempt/'amp_events.jsonl').open('x',encoding='utf-8') as log, (attempt/'live_monitor.jsonl').open('x',encoding='utf-8') as decisions:
        def emit(event, enforce=True):
            nonlocal event_seq
            event_seq+=1
            event=dict(time=utc_now(),attempt_id=attempt.name,event_seq=event_seq,**event)
            log.write(json.dumps(event,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
            decision=monitor.consume([event],mode='live')
            # No PID API/taskkill: raise only inside THIS authorized observer.
            decision['observer_pid']=os.getpid()
            decisions.write(json.dumps(decision,allow_nan=False)+'\n');decisions.flush();os.fsync(decisions.fileno())
            if enforce:
                require(decision['action']!='STOP','Live monitor requires review: '+str(decision['reason']))
        def loss_check(value):
            context['loss_finite'] &= finite(value)
            if context['loss_finite']: context['loss_values'].append(float(value.detach().item()))
            require(context['loss_finite'],'Nonfinite loss; training stopped with evidence')
            return value
        def ce(self,*a,**kw): return loss_check(original_ce(self,*a,**kw))
        def seg(*a,**kw): return loss_check(original_seg(*a,**kw))
        def opt_step(self,*a,**kw):
            require(all(finite(p.grad) for g in self.param_groups for p in g['params'] if p.grad is not None),'Nonfinite optimizer-applied gradient')
            v=original_step(self,*a,**kw);context['updates']+=1;return v
        def scaler_step(self,opt,*a,**kw):
            context['batch']+=1;before=context['updates'];scale=self.get_scale()
            # Original step performs original unscale and overflow decision.
            value=original_scaler_step(self,opt,*a,**kw)
            gradients_finite=all(finite(p.grad) for g in opt.param_groups for p in g['params'] if p.grad is not None)
            weights_finite=all(finite(p) for p in context['model'].state_dict().values() if p.is_floating_point())
            context['event']=dict(kind='batch',seed=context['seed'],stage=context['stage'],epoch=context['epoch'],batch=context['batch'],
                loss_values=list(context['loss_values']),scale_before=scale,skipped=context['updates']==before,unscaled_gradients_finite=gradients_finite,
                loss_finite=context['loss_finite'],applied_gradients_finite=gradients_finite or context['updates']==before,weights_finite=weights_finite)
            require(weights_finite,'Nonfinite model weight/buffer')
            return value
        def update(self,*a,**kw):
            value=original_update(self,*a,**kw);event=context.pop('event');event['scale_after']=self.get_scale();emit(event)
            context['loss_finite']=True;context['loss_values']=[];return value
        def epoch(model,loader,opt,sched,scaler,device,crit,a,frozen):
            stage=1 if frozen else 2
            context['epoch']=context['epoch']+1 if stage==context['stage'] else 1
            context.update(model=model,stage=stage,batch=0)
            require(device.type=='cuda','CUDA required')
            require(len(loader)==monitor.batches_per_epoch,'Actual loader epoch length conflicts with monitoring warmup')
            return original_epoch(model,loader,opt,sched,scaler,device,crit,a,frozen)
        def run(seed,a,recs,labels,domains,cache,device,out_dir):
            context.update(seed=seed,stage=None,epoch=0)
            actual=mt.make_split(labels,domains,seed)
            frozen=results.read_json(attempt/'runs'/f'seed{seed}_split.json')
            for ids,key in zip(actual,('train','val','test')):
                require([str(recs[i]['rel']).replace('\\','/') for i in ids]==frozen[key],'Actual split differs from pre-run evidence')
            pool=[str(recs[i]['rel']).replace('\\','/') for i in actual[0] if recs[i]['domain']==1]
            require(pool==frozen['background_swap_images'],'Actual swap pool differs')
            return original_run(seed,a,recs,labels,domains,cache,device,out_dir)
        mt.train_epoch,mt.run_one=epoch,run
        torch.optim.AdamW.step=opt_step;torch.amp.GradScaler.step=scaler_step;torch.amp.GradScaler.update=update
        torch.nn.CrossEntropyLoss.forward=ce;mt.seg_loss_fn=seg
        sys.argv=build_command(args)[3:]
        try:
            mt.main();emit(dict(kind='end',success=True))
        except BaseException as exc:
            emit(dict(kind='end',success=False,error=repr(exc),seed=context['seed'],stage=context['stage'],epoch=context['epoch'],batch=context['batch']),enforce=False)
            raise
        finally:
            mt.train_epoch,mt.run_one=original_epoch,original_run
            torch.optim.AdamW.step=original_step;torch.amp.GradScaler.step=original_scaler_step;torch.amp.GradScaler.update=original_update
            torch.nn.CrossEntropyLoss.forward=original_ce;mt.seg_loss_fn=original_seg

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('plan',help='Read-only selected-main formal planning; no files created')
    p=sub.add_parser('observe',help=argparse.SUPPRESS); p.add_argument('--attempt',type=Path,required=True)
    p=sub.add_parser('prepare',help='Default read-only data/copy plan')
    p.add_argument('--source',type=Path,default=SOURCE); p.add_argument('--workspace',type=Path,default=WORKSPACE)
    p.add_argument('--mode',choices=('pilot','formal'),default='pilot'); p.add_argument('--cohort-id')
    p.add_argument('--baseline-manifest',type=Path); p.add_argument('--prepare-data',action='store_true')
    p=sub.add_parser('run',help='Default read-only execution preflight')
    p.add_argument('--approved-config',type=Path); p.add_argument('--qc-approval',type=Path)
    p.add_argument('--source',type=Path,default=SOURCE); p.add_argument('--workspace',type=Path,default=WORKSPACE)
    p.add_argument('--mode',choices=('pilot','formal'),default='formal'); p.add_argument('--execute',action='store_true')
    a=parser.parse_args(argv)
    try:
        if a.action=='plan':
            output=selected_plan()
        elif a.action=='observe':
            observe_training(a.attempt); return 0
        elif a.action=='prepare':
            plan=prepare_plan(a.source,a.workspace,a.mode,a.cohort_id,a.baseline_manifest)
            output=prepare_data(plan) if a.prepare_data else public_plan(plan)
        else:
            output=public_plan(run_request(a.approved_config,a.qc_approval,a.workspace,a.source,a.mode,a.execute))
        print(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False))
        return 2 if output.get('status')=='BLOCKED' else 0
    except (Error,OSError,ValueError,KeyError,TypeError) as exc:
        print(json.dumps(dict(status='BLOCKED',execution='NOT VERIFIED',error=str(exc)),ensure_ascii=False),file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
