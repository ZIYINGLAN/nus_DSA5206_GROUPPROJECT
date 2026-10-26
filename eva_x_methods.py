"""Official EVA-X loading, smoke gate and shared nested evaluation.

No fallback to a randomly initialized backbone is allowed in the experiment.
"""
import importlib.util
import hashlib
import pickle
import types
import json
from pathlib import Path
from time import perf_counter
import urllib.request
import warnings
import numpy as np
import pandas as pd
from critical_methods import sha256_file, fingerprint, TARGETS, exact_splits
from resnet18_methods import (ALPHAS, BASELINE, RIDGE, aligned_features,
    validate_heldout_records, paired_alignment, fit_nested_ridge, regression_metrics)

EVA_MODEL = 'EVA-X-Tiny + Ridge'
OFFICIAL_COMMIT = 'eedd642c85339b25682ceb574b644cb9afe88e62'
OFFICIAL_SOURCE_SHA256 = '5c21eb02cd977c35d4b8cf2984081b88b73db49953c1b4c2235f30df29df8219'
CHECKPOINT_NAME = 'eva_x_tiny_patch16_merged520k_mim.pt'
PREPROCESSING = {'policy':'official CXR14 evaluation (not CheXpert-Tiny)',
    'resize_shorter_edge':256, 'crop':224, 'interpolation':'bicubic', 'antialias':True,
    'mean':[0.49185243]*3, 'std':[0.28509309]*3,
    'image_mode':'official PIL convert RGB (grayscale naturally replicated)', 'input_shape':[3,224,224]}

def download_file(url, destination, expected_sha=None):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.is_file() and expected_sha and sha256_file(destination)==expected_sha:
        return destination
    temp=destination.with_suffix(destination.suffix+'.partial')
    try:
        with urllib.request.urlopen(url,timeout=120) as src,temp.open('wb') as dst:
            while chunk:=src.read(1024*1024):dst.write(chunk)
        if expected_sha and sha256_file(temp)!=expected_sha:
            raise ValueError('Downloaded artifact SHA-256 mismatch')
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)
    return destination

def official_source(asset_dir):
    path=Path(asset_dir)/'official_eva_x.py'
    return download_file(f'https://raw.githubusercontent.com/hustvl/EVA-X/{OFFICIAL_COMMIT}/eva_x.py',
                         path,OFFICIAL_SOURCE_SHA256)

def official_checkpoint(asset_dir, local_path=None, expected_sha=None):
    if local_path is not None:
        path=Path(local_path)
        if not expected_sha or sha256_file(path)!=expected_sha:
            raise ValueError('Local checkpoint requires a matching trusted SHA-256')
        return path, {'source':'user-supplied official checkpoint; expected hash supplied by user',
                     'sha256':expected_sha,'filename':path.name}
    # API supplies immutable revision and LFS SHA; do not rely on mutable main.
    with urllib.request.urlopen('https://huggingface.co/api/models/MapleF/eva_x?blobs=true',timeout=60) as response:
        info=json.load(response)
    file=next(s for s in info['siblings'] if s['rfilename']==CHECKPOINT_NAME)
    expected=file['lfs']['sha256'];revision=info['sha']
    path=download_file(f'https://huggingface.co/MapleF/eva_x/resolve/{revision}/{CHECKPOINT_NAME}',
                       Path(asset_dir)/CHECKPOINT_NAME,expected)
    return path,{'repository':'MapleF/eva_x','revision':revision,'sha256':expected,'filename':CHECKPOINT_NAME}

def load_official_module(source):
    if sha256_file(source)!=OFFICIAL_SOURCE_SHA256:
        raise ValueError('Official source differs from reviewed pinned implementation')
    spec=importlib.util.spec_from_file_location('_duong_official_eva_x',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def evaluation_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256,interpolation=transforms.InterpolationMode.BICUBIC,antialias=True),
        transforms.CenterCrop(224),transforms.ToTensor(),
        transforms.Normalize(PREPROCESSING['mean'],PREPROCESSING['std'])])

def checkpoint_coverage(model, filtered_state):
    """Require every non-classifier backbone state tensor, shapes and values.

    The official filter removes MIM heads/mask token and fixed RoPE buffers.
    Only the unused classification head may be missing/unexpected. Fail closed.
    """
    import torch
    state=model.state_dict()
    required={k:v for k,v in state.items() if not k.startswith('head.') and 'rope' not in k}
    if not required:
        raise ValueError('Empty backbone cannot establish checkpoint coverage')
    missing=[k for k in required if k not in filtered_state]
    unexpected=[k for k in filtered_state if k not in state and not k.startswith('head.')]
    incompatible=[k for k,v in required.items() if k in filtered_state and tuple(v.shape)!=tuple(filtered_state[k].shape)]
    if missing or unexpected or incompatible:
        raise ValueError(f'Incomplete backbone: missing={missing}, unexpected={unexpected}, incompatible={incompatible}')
    for k,v in required.items():
        if not torch.equal(v.cpu(),filtered_state[k].to(dtype=v.dtype).cpu()):
            raise ValueError(f'Checkpoint tensor was not loaded: {k}')
    return {'backbone_state_tensors':len(required),'backbone_numel':sum(v.numel() for v in required.values()),
            'backbone_coverage_fraction':1.0,
            'ignored_missing_head_keys':[k for k in state if k.startswith('head.') and k not in filtered_state]}

def checkpoint_safe_globals():
    """Only the NumPy scalar reconstruction primitives needed for float64 metadata.

    type(dtype(...)) supports both NumPy <1.25 and the newer Float64DType class.
    No ndarray reconstructors, arbitrary metadata classes or global registration.
    """
    return [np.core.multiarray.scalar, np.dtype, type(np.dtype(np.float64))]

def load_checkpoint_safely(checkpoint, expected_sha):
    """Verify the same open file before one restricted deserialization.

    Unsupported types fail closed. There is intentionally no weights_only=False
    retry: an allowlist failure is not permission to execute checkpoint code.
    """
    import torch
    with open(checkpoint, 'rb') as stream:
        digest=hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
        if not expected_sha or digest.hexdigest()!=expected_sha:
            raise ValueError('Checkpoint integrity mismatch')
        stream.seek(0)
        try:
            # Do not remove any caller-preexisting entry when the context exits.
            existing=torch.serialization.get_safe_globals()
            additions=[entry for entry in checkpoint_safe_globals() if entry not in existing]
            with torch.serialization.safe_globals(additions):
                raw=torch.load(stream,map_location='cpu',weights_only=True)
        except pickle.UnpicklingError as exc:
            raise pickle.UnpicklingError(
                'Restricted EVA-X checkpoint loading failed; unsupported metadata '
                'must be inspected before extending the scoped allowlist. '
                'No unrestricted pickle fallback was attempted. Original error: '
                + str(exc)) from exc
    if not isinstance(raw, dict):
        raise ValueError('Expected an EVA-X checkpoint mapping')
    return raw

def _factory_with_preloaded_state(module, checkpoint, raw):
    """Run the official factory bytecode with only its torch.load bound locally.

    The pinned factory otherwise calls torch.load without weights_only, which
    defaults to unrestricted pickle in torch 2.5.1. Bind this one function's
    globals to an in-memory loader; do not patch torch.load or module globals.
    Architecture construction and checkpoint_filter_fn remain official code.
    """
    def preloaded_load(path, map_location='cpu', **kwargs):
        if str(path)!=str(checkpoint) or map_location!='cpu':
            raise ValueError('Unexpected official factory load request')
        return raw
    factory=module.eva_x_tiny_patch16
    namespace=dict(factory.__globals__)
    namespace['torch']=types.SimpleNamespace(load=preloaded_load)
    bound=types.FunctionType(factory.__code__,namespace,factory.__name__,
                             factory.__defaults__,factory.__closure__)
    return bound(pretrained=str(checkpoint))

def checkpoint_metadata_report(raw):
    """Report structure/scalar types, never tensor contents or optimizer values."""
    scalars=[]
    def visit(value, path):
        if isinstance(value,np.generic):
            scalars.append({'path':path,'type':type(value).__name__,'dtype':str(value.dtype)})
        elif isinstance(value,dict):
            for key,child in value.items():visit(child,path+'.'+str(key))
        elif isinstance(value,(list,tuple)):
            for i,child in enumerate(value):visit(child,path+f'[{i}]')
    visit(raw,'checkpoint')
    return {'top_level_keys':[str(k) for k in raw], 'numpy_scalars':scalars}

def load_eva_tiny(source, checkpoint, expected_sha):
    raw=load_checkpoint_safely(checkpoint,expected_sha)
    module=load_official_module(source)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model=_factory_with_preloaded_state(module,checkpoint,raw)
    filtered=module.checkpoint_filter_fn(raw,model)
    audit=checkpoint_coverage(model,filtered)
    if model.num_features!=192 or model.global_pool!='avg':
        raise ValueError('Unexpected architecture/pooling')
    model.eval();model.requires_grad_(False)
    audit['warnings']=[str(w.message) for w in captured]
    audit['pooling']='official forward_head(pre_logits=True): mean patch tokens excluding prefix, fc_norm'
    audit['deserialization']={
        'weights_only':True, 'unrestricted_fallback':False,
        'sha256_verified_before_deserialization':True,
        'safe_globals':[t.__module__+'.'+t.__name__ for t in checkpoint_safe_globals()],
        'official_factory_load':'already validated in-memory mapping; no second deserialization'}
    audit['checkpoint_metadata']=checkpoint_metadata_report(raw)
    return model,audit

def pooled_features(model, x):
    import torch
    if tuple(x.shape[1:])!=(3,224,224) or x.ndim!=4:
        raise ValueError('Expected Bx3x224x224 input')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Backbone must be frozen and in eval mode')
    with torch.inference_mode():
        result=model.forward_head(model.forward_features(x),pre_logits=True)
    if tuple(result.shape)!=(len(x),192) or not torch.isfinite(result).all():
        raise ValueError('Invalid pooled EVA-X feature output')
    return result

def pretrained_smoke(source, checkpoint, expected_sha):
    """Must complete before dataset extraction. No synthetic metrics are exported."""
    import torch,platform,importlib.metadata
    from PIL import Image
    model,audit=load_eva_tiny(source,checkpoint,expected_sha)
    image=Image.fromarray(np.random.default_rng(0).integers(0,256,(280,240),dtype=np.uint8))
    x=evaluation_transform()(image.convert('RGB')).unsqueeze(0)
    first=pooled_features(model,x);second=pooled_features(model,x)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    report={'status':'passed','synthetic_image':True,'input_shape':list(x.shape),
            'feature_shape':list(first.shape),'repeat_bitwise_equal':True,
            'checkpoint_sha256':expected_sha,'device':str(next(model.parameters()).device),
            'python':platform.python_version(),
            'versions':{p:importlib.metadata.version(p) for p in ['torch','torchvision','timm']},
            'loading':audit,'preprocessing':PREPROCESSING}
    return model,report

def verify_prior_identity(previous,current):
    """Hard stop on available identity fields, not merely equal cohort counts."""
    required=['ordered_filenames','ordered_patient_ids','ordered_labels',
              'metadata_sha256','scores_sha256','scored_image_sha256']
    for key in required:
        if key not in previous or previous[key]!=current[key]:
            raise ValueError(f'Previous experiment identity mismatch/missing field: {key}')
    return True

def evaluate_eva(features,metadata,earlier_predictions,alphas=ALPHAS):
    features=aligned_features(features,metadata.filename.tolist(),metadata,192)
    # Require both complete earlier baselines, no accidental optional PCA rows.
    earlier_predictions=earlier_predictions[earlier_predictions.model.isin([BASELINE,RIDGE])].copy()
    for model in [BASELINE,RIDGE]:
        validate_heldout_records(earlier_predictions,metadata,'model',model)
    paired_alignment(earlier_predictions)
    predictions=[earlier_predictions];timings=[];audits=[]
    for seed,tr,te in exact_splits(metadata):
        groups=metadata.iloc[tr].patientid.to_numpy()
        for target,column in TARGETS.items():
            y=metadata[column].to_numpy(float)
            search,folds,seconds=fit_nested_ridge(features[tr],y[tr],groups,False,alphas)
            start=perf_counter();yp=search.predict(features[te]);predict_seconds=perf_counter()-start
            timings.append({'target':target,'seed':seed,'search_fit_seconds':seconds,
                'predict_seconds':predict_seconds,'best_alpha':search.best_params_['ridge__alpha']})
            if target==next(iter(TARGETS)):
                audits.append({'seed':seed,'train_indices':tr.tolist(),'test_indices':te.tolist(),
                    'train_patient_ids':metadata.iloc[tr].patientid.tolist(),
                    'test_patient_ids':metadata.iloc[te].patientid.tolist(),'patient_overlap':0,
                    'inner_folds':[{'train_indices':tr[a].tolist(),'validation_indices':tr[b].tolist()} for a,b in folds]})
            predictions.append(pd.DataFrame({'model':EVA_MODEL,'target':target,'seed':seed,
                'row_index':te,'filename':metadata.iloc[te].filename.to_numpy(),
                'patientid':metadata.iloc[te].patientid.to_numpy(),'y_true':y[te],'y_pred':yp}))
    combined=pd.concat(predictions,ignore_index=True);paired_alignment(combined)
    rows=[]
    for (model,target,seed),part in combined.groupby(['model','target','seed']):
        rows.append({'model':model,'target':target,'seed':seed,**regression_metrics(part.y_true,part.y_pred)})
    return pd.DataFrame(rows),combined,pd.DataFrame(timings),audits
