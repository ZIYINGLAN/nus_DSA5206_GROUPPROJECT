"""Prediction-only descriptive error analysis. No model loading, fitting or features.

Raw identities stay in memory/private_audit. Public exports and plots use study IDs.
"""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = ('DenseNet Lung Opacity + OLS', 'ResNet18 + Ridge', 'EVA-X-Tiny + Ridge')
DENSE, RESNET, EVA = MODELS
TARGETS = ('Geographic Extent', 'Opacity')
LABEL_COLUMNS = {'Geographic Extent': 'geographic_mean', 'Opacity': 'opacity_mean'}
REQUIRED = ['model','target','seed','row_index','filename','patientid','y_true','y_pred']
IMAGE_KEYS = ['target','row_index','filename','patientid']
PAIR_COLUMNS = ['mean_delta_abs_error_eva_dense','mean_delta_abs_error_eva_resnet']
SCORE_LIMITS = {'Geographic Extent': 8., 'Opacity': 6.}
STRATUM_LABELS = ['S1','S2','S3']
ALL_LARGE_AE = 1.0  # score units; descriptive, not a clinical cutoff
SMALL_GROUP = 5


def file_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verified_prediction_filename(notebook_path):
    """Read actual predictions.to_csv export expression; never guess a filename."""
    notebook=json.loads(Path(notebook_path).read_text())
    found=[]
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        try:tree=ast.parse(''.join(cell['source']))
        except SyntaxError:continue  # pip/magic cells contain no export
        for call in ast.walk(tree):
            if (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                and call.func.attr=='to_csv' and isinstance(call.func.value,ast.Name)
                and call.func.value.id=='predictions' and call.args):
                expr=call.args[0]
                if (isinstance(expr,ast.BinOp) and isinstance(expr.op,ast.Div)
                    and isinstance(expr.left,ast.Name) and expr.left.id=='OUTPUT_DIR'
                    and isinstance(expr.right,ast.Constant) and isinstance(expr.right.value,str)):
                    found.append(expr.right.value)
    if len(found)!=1 or Path(found[0]).name!=found[0]:
        raise ValueError('Cannot unambiguously verify the three-model prediction export')
    return found[0]


def _integers(values, name):
    raw=np.asarray(values)
    try:numeric=np.asarray(values,dtype=float)
    except (ValueError,TypeError) as exc:raise ValueError(f'Invalid {name}') from exc
    if (numeric.ndim!=1 or not np.isfinite(numeric).all() or (numeric<0).any()
        or (numeric!=np.floor(numeric)).any() or any(isinstance(v,(bool,np.bool_)) for v in raw)):
        raise ValueError(f'Expected nonnegative integer {name}')
    return numeric.astype(np.int64)


def _identity(value):
    if value is None or pd.isna(value) or not str(value).strip():
        raise ValueError('Missing filename or patient ID')
    return str(value)  # keep exact spelling; no stripping/relabeling identities


def validate_predictions(records, split_audit, manifest=None, expected_cohort=(94,65)):
    """Fail closed on identifiers, labels, pairing and external audited membership.

    Requires all 50 audited splits, not each image in each split. Manifest identity
    is additionally checked when present; absence is recorded, not fabricated.
    """
    if not set(REQUIRED).issubset(records):raise ValueError('Missing required prediction columns')
    frame=records[REQUIRED].copy()
    if frame.empty or frame[REQUIRED].isna().any().any():raise ValueError('Empty/missing prediction records')
    if set(frame.model)!=set(MODELS):raise ValueError('Expected exactly three original model identifiers')
    if set(frame.target)!=set(TARGETS):raise ValueError('Expected Geographic Extent and Opacity targets')
    for col in ['seed','row_index']:frame[col]=_integers(frame[col],col)
    for col in ['filename','patientid']:frame[col]=frame[col].map(_identity)
    for col in ['y_true','y_pred']:
        try:frame[col]=pd.to_numeric(frame[col],errors='raise').astype(float)
        except (ValueError,TypeError) as exc:raise ValueError('Nonnumeric predictions/labels') from exc
        if not np.isfinite(frame[col]).all():raise ValueError('Nonfinite predictions/labels')
    if frame.duplicated(['model','target','seed','row_index']).any():raise ValueError('Duplicate prediction keys')
    for model in MODELS:
        for target in TARGETS:
            if set(frame[(frame.model==model)&(frame.target==target)].seed)!=set(range(50)):
                raise ValueError('Every model/target requires exact seeds 0-49')
    paired=frame.groupby(['target','seed','row_index'],sort=False)
    if not (paired.model.nunique()==3).all():raise ValueError('Missing paired model prediction')
    for col in ['filename','patientid','y_true']:
        if not (paired[col].nunique()==1).all():raise ValueError(f'Model alignment mismatch: {col}')
    image_identity=frame.groupby('row_index')[['filename','patientid']].nunique()
    if not (image_identity==1).all().all():raise ValueError('Inconsistent row/filename/patient identity')
    if not (frame.groupby('filename')[['row_index','patientid']].nunique()==1).all().all():
        raise ValueError('Inconsistent filename-to-row/patient mapping')
    if not (frame.groupby(['target','row_index']).y_true.nunique()==1).all():
        raise ValueError('True labels differ across held-out appearances')
    if not (frame.groupby('row_index').target.nunique()==2).all():raise ValueError('Image lacks one target')
    for target,maximum in SCORE_LIMITS.items():
        if not frame.loc[frame.target==target,'y_true'].between(0,maximum).all():
            raise ValueError('Ground-truth score outside target range')
    identity=frame.drop_duplicates('row_index').set_index('row_index')
    cohort={}
    if manifest is not None:
        if not isinstance(manifest,dict):raise ValueError('Manifest must be a mapping')
        if 'outer_seeds' in manifest and manifest['outer_seeds']!=list(range(50)):
            raise ValueError('Manifest outer seeds differ from audited experiment')
        names=manifest.get('ordered_filenames');patients=manifest.get('ordered_patient_ids')
        labels=manifest.get('ordered_labels')
        if not isinstance(names,list) or not isinstance(patients,list) or not isinstance(labels,dict):
            raise ValueError('Manifest lacks ordered cohort/label identity')
        if len(names)!=len(patients) or len(set(names))!=len(names):raise ValueError('Invalid manifest cohort')
        for target,column in LABEL_COLUMNS.items():
            if column not in labels or len(labels[column])!=len(names):raise ValueError('Invalid manifest labels')
        for idx,(name,patient) in enumerate(zip(names,patients)):
            cohort[idx]={'filename':_identity(name),'patientid':_identity(patient)}
            for target,column in LABEL_COLUMNS.items():
                value=float(labels[column][idx])
                if not np.isfinite(value) or not 0<=value<=SCORE_LIMITS[target]:raise ValueError('Invalid manifest score')
                cohort[idx][target]=value
        if expected_cohort is not None and (len(cohort),len(set(patients)))!=tuple(expected_cohort):
            raise ValueError('Manifest cohort is not the expected 94 images / 65 patients')
        for row in frame.itertuples(index=False):
            expected=cohort.get(row.row_index)
            if (expected is None or expected['filename']!=row.filename or expected['patientid']!=row.patientid
                or expected[row.target]!=row.y_true):raise ValueError('Manifest sample ordering/identity/label mismatch')
    if not isinstance(split_audit,list) or len(split_audit)!=50:raise ValueError('Require 50 previously audited splits')
    audit_by_seed={}
    for item in split_audit:
        seed=int(_integers([item.get('seed')],'audit seed')[0])
        if seed in audit_by_seed:raise ValueError('Duplicate audited seed')
        audit_by_seed[seed]=item
    if set(audit_by_seed)!=set(range(50)):raise ValueError('Audited seeds must be 0-49')
    audit_patient_map={};full_set=None
    for seed in range(50):
        item=audit_by_seed[seed]
        tr=_integers(item.get('train_indices',[]),'audit train indices')
        te=_integers(item.get('test_indices',[]),'audit test indices')
        if not len(tr) or not len(te) or len(set(tr))!=len(tr) or len(set(te))!=len(te) or set(tr)&set(te):
            raise ValueError('Invalid audited train/test partition')
        partition=set(tr)|set(te)
        if full_set is None:full_set=partition
        if partition!=full_set:raise ValueError('Audited cohort changes across seeds')
        if manifest is not None and partition!=set(cohort):raise ValueError('Audit/manifest cohort mismatch')
        group_ids=[]
        for indices,key in [(tr,'train_patient_ids'),(te,'test_patient_ids')]:
            values=item.get(key)
            if values is None:
                if manifest is None:raise ValueError('Cannot verify patient separation without IDs or manifest')
                values=[cohort[int(idx)]['patientid'] for idx in indices]
            if len(values)!=len(indices):raise ValueError('Audit patient list/index length mismatch')
            values=[_identity(v) for v in values];group_ids.append(set(values))
            for idx,patient in zip(indices,values):
                idx=int(idx)
                if idx in audit_patient_map and audit_patient_map[idx]!=patient:raise ValueError('Audited patient mapping changes')
                audit_patient_map[idx]=patient
                if manifest is not None and cohort[idx]['patientid']!=patient:raise ValueError('Audit/manifest patient mismatch')
                if idx in identity.index and identity.loc[idx,'patientid']!=patient:raise ValueError('Audit/prediction patient mismatch')
        if group_ids[0]&group_ids[1]:raise ValueError('Patient overlap in audited train/test split')
        for target in TARGETS:
            observed=frame[(frame.seed==seed)&(frame.target==target)&(frame.model==DENSE)]
            if set(observed.row_index)!=set(te):raise ValueError(f'Observed test membership mismatch: {target} seed {seed}')
        for key,indices in [('train_filenames',tr),('test_filenames',te)]:
            if key in item:
                values=item[key]
                if len(values)!=len(indices):raise ValueError('Audit filename list/index length mismatch')
                for idx,name in zip(indices,values):
                    if idx in identity.index and str(name)!=identity.loc[idx,'filename']:raise ValueError('Audit filename mismatch')
                    if manifest is not None and str(name)!=cohort[int(idx)]['filename']:raise ValueError('Audit filename/manifest mismatch')
    observed_rows=set(identity.index)
    patients=sorted(set(audit_patient_map.values()))
    case_ids={int(idx):f'C{rank:03d}' for rank,idx in enumerate(sorted(full_set),1)}
    patient_ids={patient:f'P{rank:03d}' for rank,patient in enumerate(patients,1)}
    frame['case_id']=frame.row_index.map(case_ids)
    frame['study_patient_id']=frame.patientid.map(patient_ids)
    frame=frame.sort_values(['target','seed','row_index','model']).reset_index(drop=True)
    cohort_rows=[]
    for idx in sorted(full_set):
        cohort_rows.append({'row_index':int(idx),'case_id':case_ids[int(idx)],'patientid':audit_patient_map[int(idx)],
            'study_patient_id':patient_ids[audit_patient_map[int(idx)]],
            'filename':cohort[idx]['filename'] if manifest is not None else (identity.loc[idx,'filename'] if idx in identity.index else None),
            'has_heldout_predictions':idx in observed_rows})
    validation={'manifest_available':manifest is not None,'external_membership_audit_verified':True,
        'audited_images':len(full_set),'audited_patients':len(patients),
        'observed_images':len(observed_rows),'observed_patients':frame.patientid.nunique(),
        'unobserved_images':len(full_set-observed_rows),'records':len(frame),'seeds':list(range(50))}
    return frame,pd.DataFrame(cohort_rows),validation


def add_errors(frame):
    result=frame.copy()
    result['signed_error']=result.y_pred-result.y_true
    result['absolute_error']=result.signed_error.abs()
    result['squared_error']=result.signed_error**2
    if not np.isfinite(result[['signed_error','absolute_error','squared_error']]).all().all():
        raise ValueError('Error calculation overflow')
    pivot=result.pivot(index=['target','seed','row_index'],columns='model',values='absolute_error')
    pairs=pd.DataFrame({'delta_abs_error_eva_dense':pivot[EVA]-pivot[DENSE],
                        'delta_abs_error_eva_resnet':pivot[EVA]-pivot[RESNET]}).reset_index()
    return result.merge(pairs,on=['target','seed','row_index'],validate='many_to_one')


def summarize_images(errors):
    keys=IMAGE_KEYS+['case_id','study_patient_id','model']
    images=errors.groupby(keys,sort=True,dropna=False).agg(
        n_heldout=('seed','size'),mean_signed_error=('signed_error','mean'),
        mean_absolute_error=('absolute_error','mean'),median_absolute_error=('absolute_error','median'),
        std_signed_error=('signed_error','std'),mean_prediction=('y_pred','mean'),
        y_true=('y_true','first'),
        overestimation_rate=('signed_error',lambda x:(x>0).mean()),
        underestimation_rate=('signed_error',lambda x:(x<0).mean()),
        zero_error_prediction_rate=('signed_error',lambda x:(x==0).mean())).reset_index()
    paired=errors[errors.model==EVA].groupby(IMAGE_KEYS+['case_id','study_patient_id'],sort=True).agg(
        n_heldout=('seed','size'),mean_delta_abs_error_eva_dense=('delta_abs_error_eva_dense','mean'),
        mean_delta_abs_error_eva_resnet=('delta_abs_error_eva_resnet','mean')).reset_index()
    images=images.merge(paired.drop(columns='n_heldout'),on=IMAGE_KEYS+['case_id','study_patient_id'],validate='many_to_one')
    return images,paired


def summarize_patients(images,image_pairs,cohort):
    keys=['target','patientid','study_patient_id']
    patients=images.groupby(keys+['model'],sort=True).agg(
        n_analyzed_images=('row_index','nunique'),mean_absolute_error=('mean_absolute_error','mean'),
        mean_signed_error=('mean_signed_error','mean'),mean_prediction=('mean_prediction','mean'),
        overestimation_rate=('overestimation_rate','mean'),underestimation_rate=('underestimation_rate','mean'),
        zero_error_prediction_rate=('zero_error_prediction_rate','mean'),
        mean_true_score=('y_true','mean'),min_image_appearances=('n_heldout','min'),
        max_image_appearances=('n_heldout','max'),mean_image_appearances=('n_heldout','mean')).reset_index()
    totals=cohort.groupby('patientid').row_index.nunique().rename('n_scored_images').reset_index()
    patients=patients.merge(totals,on='patientid',validate='many_to_one')
    counts=images[images.model==EVA].groupby(keys).apply(
        lambda g:json.dumps(dict(zip(g.case_id,g.n_heldout.astype(int))),sort_keys=True),include_groups=False
    ).rename('heldout_appearances_per_image').reset_index()
    patients=patients.merge(counts,on=keys,validate='many_to_one')
    paired=image_pairs.groupby(keys,sort=True).agg(n_analyzed_images=('row_index','nunique'),
        mean_delta_abs_error_eva_dense=(PAIR_COLUMNS[0],'mean'),
        mean_delta_abs_error_eva_resnet=(PAIR_COLUMNS[1],'mean')).reset_index()
    patients=patients.merge(paired.drop(columns='n_analyzed_images'),on=keys,validate='many_to_one')
    rows=[]
    for target,group in paired.groupby('target'):
        for column,comparison in zip(PAIR_COLUMNS,['DenseNet','ResNet18']):
            rows.append({'target':target,'comparison':comparison,'n_analyzed_patients':len(group),
                'n_lower_EVA_error':int((group[column]<0).sum()),
                'proportion_lower_EVA_error':float((group[column]<0).mean()),
                'n_equal_error':int((group[column]==0).sum()),'mean_paired_difference':group[column].mean()})
    return patients,paired,pd.DataFrame(rows)


def signed_error_summary(images,patients):
    rows=[]
    for level,table in [('image_equal_weight',images),('patient_equal_weight',patients)]:
        for (target,model),group in table.groupby(['target','model']):
            error=group.mean_signed_error
            rows.append({'level':level,'target':target,'model':model,'n_units':len(group),
                'mean_signed_error':error.mean(),'median_signed_error':error.median(),
                'overestimation_rate':group.overestimation_rate.mean(),
                'underestimation_rate':group.underestimation_rate.mean(),
                'zero_error_prediction_rate':group.zero_error_prediction_rate.mean(),
                'positive_mean_bias_fraction':(error>0).mean(),'negative_mean_bias_fraction':(error<0).mean(),
                'zero_mean_error_rate':(error==0).mean(),'mean_absolute_error':group.mean_absolute_error.mean()})
    return pd.DataFrame(rows)


def severity_summary(images):
    data=images.copy();data['stratum']=''
    rows=[]
    for target in TARGETS:
        edges=np.linspace(0,SCORE_LIMITS[target],4)
        mask=data.target==target
        data.loc[mask,'stratum']=pd.cut(data.loc[mask,'y_true'],edges,labels=STRATUM_LABELS,
                                      include_lowest=True,right=True).astype(str)
        for model in MODELS:
            for i,label in enumerate(STRATUM_LABELS):
                part=data[mask&(data.model==model)&(data.stratum==label)]
                n_images=part.row_index.nunique();n_patients=part.patientid.nunique()
                rows.append({'target':target,'model':model,'stratum':label,
                    'lower_score':edges[i],'upper_score':edges[i+1],
                    'n_images':n_images,'n_patients':n_patients,
                    'mean_absolute_error':part.mean_absolute_error.mean(),
                    'mean_signed_error':part.mean_signed_error.mean(),
                    'small_stratum':n_images<SMALL_GROUP or n_patients<SMALL_GROUP})
    return data,pd.DataFrame(rows)


def image_rankings(images):
    rows=[]
    for target in TARGETS:
        eva=images[(images.target==target)&(images.model==EVA)]
        for name,column,ascending,eligibility in [
            ('largest_EVA_improvement',PAIR_COLUMNS[0],True,eva[PAIR_COLUMNS[0]]<0),
            ('largest_EVA_deterioration',PAIR_COLUMNS[0],False,eva[PAIR_COLUMNS[0]]>0),
            ('lowest_EVA_MAE','mean_absolute_error',True,np.ones(len(eva),bool)),
            ('highest_EVA_MAE','mean_absolute_error',False,np.ones(len(eva),bool))]:
            ranked=eva.loc[eligibility].sort_values([column,'row_index'],ascending=[ascending,True]).copy()
            ranked['ranking']=name;ranked['rank']=np.arange(1,len(ranked)+1);rows.append(ranked)
    return pd.concat(rows,ignore_index=True)


def representative_cases(images):
    """One extremal eligible case/criterion/target, ties -> smallest row_index.

    Selection is descriptive and post hoc; a case may satisfy several criteria.
    A/B require strict improvement/deterioration; C/D require signed direction.
    E requires all three image-mean AEs >=1 score unit. Missing categories are
    represented explicitly, never filled by a different ineligible case.
    """
    rows=[]
    for target in TARGETS:
        eva=images[(images.target==target)&(images.model==EVA)].copy()
        worst=images[images.target==target].groupby('row_index').mean_absolute_error.min()
        eva['minimum_model_MAE']=eva.row_index.map(worst)
        rules=[('A_improvement',eva[PAIR_COLUMNS[0]]<0,PAIR_COLUMNS[0],True),
               ('B_deterioration',eva[PAIR_COLUMNS[0]]>0,PAIR_COLUMNS[0],False),
               ('C_overestimation',eva.mean_signed_error>0,'mean_signed_error',False),
               ('D_underestimation',eva.mean_signed_error<0,'mean_signed_error',True),
               ('E_all_large_errors',eva.minimum_model_MAE>=ALL_LARGE_AE,'minimum_model_MAE',False)]
        for criterion,eligible,column,ascending in rules:
            choices=eva.loc[eligible].sort_values([column,'row_index'],ascending=[ascending,True])
            if choices.empty:
                rows.append({'target':target,'criterion':criterion,'selection_status':'no eligible case'})
            else:
                chosen=choices.iloc[0]
                for record in images[(images.target==target)&(images.row_index==chosen.row_index)].to_dict('records'):
                    record.update(criterion=criterion,selection_status='selected');rows.append(record)
    return pd.DataFrame(rows).reindex(columns=list(images.columns)+['criterion','selection_status'])


def analyze_predictions(validated,cohort):
    errors=add_errors(validated);images,image_pairs=summarize_images(errors)
    patients,patient_pairs,improvement=summarize_patients(images,image_pairs,cohort)
    stratified_images,strata=severity_summary(images)
    return {'aligned_prediction_errors':errors,'image_level_errors':images,'patient_level_errors':patients,
        'image_level_paired_differences':image_pairs,'patient_level_paired_differences':patient_pairs,
        'patient_improvement_summary':improvement,'signed_error_summary':signed_error_summary(images,patients),
        'severity_stratified_summary':strata,'representative_cases':representative_cases(images),
        'image_rankings':image_rankings(images),'stratified_images':stratified_images}


def presentation_table(table):
    return table.drop(columns=['filename','patientid'],errors='ignore').copy()

FIGURE_NAMES = ('image_level_paired_error.png','patient_level_error_comparison.png',
    'signed_error_bias.png','severity_stratified_error.png','failure_case_comparison.png',
    'error_analysis_summary.png')
COLORS = {DENSE:'#4477AA',RESNET:'#EE7733',EVA:'#228833'}
SHORT = {DENSE:'DenseNet + OLS',RESNET:'ResNet18 + Ridge',EVA:'EVA-X-Tiny + Ridge'}


def make_figures(tables,output_dir,synthetic_fixture=False):
    """All plotted units are unique-image or equal-image patient summaries."""
    import matplotlib.pyplot as plt
    output_dir=Path(output_dir)
    images=tables['image_level_errors'];patients=tables['patient_level_errors']
    pairs=tables['image_level_paired_differences'];bias=tables['signed_error_summary']
    strata=tables['severity_stratified_summary'];cases=tables['representative_cases']
    def save(fig,name,caption):
        fig.text(.5,.015,caption,ha='center',fontsize=9,wrap=True)
        if synthetic_fixture:
            fig.text(.5,.5,'SYNTHETIC TEST FIXTURE — NOT STUDY RESULTS',ha='center',rotation=20,alpha=.25,fontsize=16)
        fig.tight_layout(rect=(0,.08,1,.96));fig.savefig(output_dir/name,dpi=180,bbox_inches='tight');plt.close(fig)
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,target in zip(axes,TARGETS):
            data=pairs[pairs.target==target]
            ax.boxplot([data[c].to_numpy() for c in PAIR_COLUMNS],tick_labels=['EVA − DenseNet','EVA − ResNet18'])
            ax.axhline(0,color='black',ls='--',lw=1)
            ax.set(title=f'{target}: {len(data)} distinct images',ylabel='Mean matched absolute-error difference (score units)')
        save(fig,FIGURE_NAMES[0],'Each image contributes once after averaging its matched held-out differences. Negative favours EVA-X; descriptive, dependent resampling.')
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for ax,target in zip(axes,TARGETS):
            data=patients[patients.target==target]
            ax.boxplot([data[data.model==m].mean_absolute_error for m in MODELS],tick_labels=[SHORT[m] for m in MODELS])
            ax.tick_params(axis='x',rotation=15)
            ax.set(title=f'{target}: {data.study_patient_id.nunique()} analyzed patients',ylabel='Equal-image patient mean absolute error (score units)')
        save(fig,FIGURE_NAMES[1],'Within each image average held-out absolute errors; within each patient average distinct-image values. Patient summaries share overlapping trained models.')
        fig,axes=plt.subplots(2,2,figsize=(14,10))
        for i,target in enumerate(TARGETS):
            for j,level in enumerate(['image_equal_weight','patient_equal_weight']):
                ax=axes[i,j];data=bias[(bias.target==target)&(bias.level==level)].set_index('model').loc[list(MODELS)]
                x=np.arange(3);ax.bar(x,data.mean_signed_error,color=[COLORS[m] for m in MODELS],label='Mean signed error')
                ax.scatter(x,data.median_signed_error,color='black',marker='D',label='Median signed error',zorder=3)
                ax.axhline(0,color='gray',ls='--');ax.set_xticks(x,[SHORT[m] for m in MODELS],rotation=12)
                ax.set(title=f'{target}: {level.replace("_"," ")}',ylabel='Signed error (prediction − truth)')
                rates='; '.join(f'{SHORT[m].split(" +")[0]}: over {data.loc[m,"overestimation_rate"]:.0%}, under {data.loc[m,"underestimation_rate"]:.0%}' for m in MODELS)
                ax.text(.5,1.02,rates,ha='center',transform=ax.transAxes,fontsize=8);ax.legend(fontsize=8)
        save(fig,FIGURE_NAMES[2],'Over/under rates average within-image held-out fractions, then equal-image patient fractions; not pooled records. Bars/diamonds show unit-mean signed-error mean/median. No causal/fairness claims.')
        fig,axes=plt.subplots(2,2,figsize=(14,10))
        for i,target in enumerate(TARGETS):
            ax=axes[i,0];x=np.arange(3)
            for j,model in enumerate(MODELS):
                data=strata[(strata.target==target)&(strata.model==model)].set_index('stratum').loc[STRATUM_LABELS]
                ax.bar(x+(j-1)*.24,data.mean_absolute_error,width=.24,color=COLORS[model],label=SHORT[model])
            count=strata[(strata.target==target)&(strata.model==EVA)].set_index('stratum').loc[STRATUM_LABELS]
            ax.set_xticks(x,[f'{label}\n{int(count.loc[label,"n_images"])} images / {int(count.loc[label,"n_patients"])} patients' for label in STRATUM_LABELS])
            ax.set(title=target+' descriptive score strata',ylabel='Image-equal mean absolute error');ax.legend(fontsize=8)
            ax=axes[i,1]
            for model in MODELS:
                data=images[(images.target==target)&(images.model==model)]
                ax.scatter(data.y_true,data.mean_signed_error,s=18,alpha=.5,color=COLORS[model],label=SHORT[model])
                trend=strata[(strata.target==target)&(strata.model==model)]
                ax.plot((trend.lower_score+trend.upper_score)/2,trend.mean_signed_error,color=COLORS[model],lw=1)
            ax.axhline(0,color='gray',ls='--');ax.set(title=target+' error vs expert score',xlabel='Ground-truth mean score',ylabel='Image mean signed error');ax.legend(fontsize=8)
        save(fig,FIGURE_NAMES[3],'S1/S2/S3 split each target’s score range into fixed thirds; not clinical categories. Lines connect stratum means, not fitted models. Groups <5 images or patients are flagged in CSV; empty strata remain N/A.')
        fig,axes=plt.subplots(2,1,figsize=(14,10))
        criteria=['A_improvement','B_deterioration','C_overestimation','D_underestimation','E_all_large_errors']
        for ax,target in zip(axes,TARGETS):
            data=cases[cases.target==target];x=np.arange(len(criteria));ticks=[]
            for criterion in criteria:
                part=data[(data.criterion==criterion)&(data.selection_status=='selected')]
                ticks.append(criterion.split('_')[0]+': '+(part.case_id.iloc[0] if len(part) else 'none eligible'))
            for j,model in enumerate(MODELS):
                values=[]
                for criterion in criteria:
                    part=data[(data.criterion==criterion)&(data.model==model)&(data.selection_status=='selected')]
                    values.append(part.mean_signed_error.iloc[0] if len(part) else np.nan)
                ax.bar(x+(j-1)*.24,values,width=.24,color=COLORS[model],label=SHORT[model])
            ax.axhline(0,color='gray',ls='--');ax.set_xticks(x,ticks)
            ax.set(title=target+' deliberately selected cases',ylabel='Mean held-out signed error (score units)');ax.legend(fontsize=8)
        save(fig,FIGURE_NAMES[4],'A strongest strict improvement; B strict deterioration; C overestimation; D underestimation; E all model image MAEs ≥1. IDs may repeat. Signed-error means differ from MAE; complete statistics are in the case table.')
        fig,axes=plt.subplots(1,2,figsize=(15,6))
        improvement=tables['patient_improvement_summary']
        for ax,target in zip(axes,TARGETS):
            ax.axis('off');rows=[]
            for model in MODELS:
                im=images[(images.target==target)&(images.model==model)]
                pt=patients[(patients.target==target)&(patients.model==model)]
                rows.append([SHORT[model],f'{im.mean_absolute_error.mean():.3f}',f'{pt.mean_absolute_error.mean():.3f}',f'{pt.mean_signed_error.mean():+.3f}'])
            table=ax.table(cellText=rows,colLabels=['Pipeline','Image MAE','Patient MAE','Patient bias'],cellLoc='center',bbox=[0,.38,1,.4])
            table.auto_set_font_size(False);table.set_fontsize(9)
            stats=improvement[improvement.target==target]
            text='\n'.join(f'EVA-X lower error than {r.comparison}: {r.n_lower_EVA_error}/{r.n_analyzed_patients} patients ({r.proportion_lower_EVA_error:.1%})' for r in stats.itertuples())
            subset=images[(images.target==target)&(images.model==EVA)]
            ax.set_title(target);ax.text(.5,.9,f'{len(subset)} distinct held-out images / {subset.study_patient_id.nunique()} analyzed patients',ha='center',transform=ax.transAxes)
            ax.text(.5,.18,text,ha='center',transform=ax.transAxes,fontsize=10)
        save(fig,FIGURE_NAMES[5],'Prediction-only descriptive analysis. Image means precede patient averaging; patient improvement denominator is analyzed patients. Repeated training/test splits are dependent. No significance or external-validity claim.')


def export_analysis(tables,cohort,validation,output_dir,input_paths,source_manifest=None,synthetic_fixture=False):
    """Create a NEW folder only. Never overwrite a prior run or source artifact."""
    output_dir=Path(output_dir).resolve()
    for value in input_paths.values():
        parent=Path(value).resolve().parent
        if output_dir==parent or output_dir.is_relative_to(parent):
            raise ValueError('Output must be outside prior input/results folders')
    if output_dir.exists():raise FileExistsError('Choose a new output directory; prior runs are never overwritten')
    output_dir.mkdir(parents=True,exist_ok=False)
    private=output_dir/'private_audit';private.mkdir(mode=0o700)
    for name,table in tables.items():
        if name!='stratified_images':presentation_table(table).to_csv(output_dir/(name+'.csv'),index=False)
    cohort.to_csv(private/'identity_mapping.csv',index=False)
    tables['representative_cases'].to_csv(private/'representative_cases.csv',index=False)
    if source_manifest is not None:
        (private/'source_reproducibility_manifest.json').write_text(json.dumps(source_manifest,indent=2))
    make_figures(tables,output_dir,synthetic_fixture)
    import platform,importlib.metadata
    canonical=cohort[['row_index','filename','patientid']].to_dict('records')
    manifest={'analysis_version':'1.0','data_kind':'synthetic_test_fixture' if synthetic_fixture else 'supplied_heldout_predictions',
        'input_files':{key:{'basename':Path(path).name,'sha256':file_sha256(path)} for key,path in input_paths.items()},
        'cohort_identity_sha256':hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest(),
        'validation':validation,'models':list(MODELS),'targets':list(TARGETS),
        'source_provenance':{key:source_manifest.get(key) for key in ['eva_source_commit','eva_checkpoint','packages','baseline_source','scatter_seed']} if source_manifest else None,
        'identity_mapping':'private_audit/identity_mapping.csv (do not publish)',
        'estimands':{'image':'mean of actual held-out errors across seeds; not error of mean prediction',
                    'patient':'equal-weight mean of distinct image mean errors within patient',
                    'paired':'matched seed/image differences first, then equal-image patient means',
                    'bias_rates':'within-image held-out over/under fractions, averaged equally across images or across equal-image patient summaries; unit-mean bias signs reported separately'},
        'case_selection':{'A':'most negative strict image mean EVA-Dense AE difference',
            'B':'most positive strict image mean EVA-Dense AE difference','C':'largest positive EVA mean signed error',
            'D':'most negative EVA mean signed error','E':f'all image model MAEs >= {ALL_LARGE_AE}; maximize minimum model MAE',
            'tie_break':'smallest row_index','cases_per_criterion_target':1,'missing_category':'no eligible case; no substitution'},
        'score_strata':{target:np.linspace(0,maximum,4).tolist() for target,maximum in SCORE_LIMITS.items()},
        'stratum_boundary':'right-closed; first includes zero; descriptive, not clinical',
        'small_stratum_threshold':SMALL_GROUP,
        'dependence':'Repeated appearances are not independent images/patients; overlapping training sets also couple patient summaries. No inferential significance tests.',
        'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','matplotlib']},
        'analysis_code_sha256':file_sha256(Path(__file__)),
        'model_training_provenance_status':'reported from supplied manifest, not revalidated by loading or refitting models'}
    (output_dir/'error_analysis_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest
