"""Paired, nested patient-grouped evaluation for the ResNet18 comparison."""
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from critical_methods import TARGETS, exact_splits, evaluate_features

ALPHAS = (0.01, 0.1, 1., 10., 100., 1000.)
SCATTER_SEED = 0
BASELINE = 'DenseNet Lung Opacity + OLS'
RIDGE = 'ResNet18 + Ridge'
PCA_RIDGE = 'ResNet18 + PCA + Ridge'

def aligned_features(values, ordered_filenames, metadata, dimension):
    values = np.asarray(values)
    if list(ordered_filenames) != metadata.filename.tolist():
        raise ValueError('Feature filename order differs from metadata')
    if values.shape != (len(metadata), dimension) or not np.isfinite(values).all():
        raise ValueError('Feature dimensions or finite-value check failed')
    return values

def validate_heldout_records(records, metadata, feature_column='features',
                             feature_value='lung opacity output'):
    """Audit every seed/target against exact_splits, without a split JSON.

    CSV patient-ID dtype must be read to match metadata. Row order in the CSV
    need not match; all keys and membership must. Never repair a mismatch.
    """
    required = {'target','seed','row_index','filename','patientid','y_true','y_pred',feature_column}
    if not required.issubset(records):
        raise ValueError('Prediction CSV lacks required columns')
    selected = records[records[feature_column] == feature_value].copy()
    if set(selected.target) != set(TARGETS) or set(selected.seed) != set(range(50)):
        raise ValueError('Incomplete baseline targets or split seeds')
    expected_total = 0
    for seed,tr,te in exact_splits(metadata):
        for target,column in TARGETS.items():
            expected_total += len(te)
            part = selected[(selected.seed == seed) & (selected.target == target)]
            if part.row_index.duplicated().any() or set(part.row_index) != set(te):
                raise ValueError(f'Test membership mismatch: {target}, seed {seed}')
            part = part.set_index('row_index').loc[te]
            truth = metadata.iloc[te]
            if part.filename.tolist() != truth.filename.tolist() or part.patientid.tolist() != truth.patientid.tolist():
                raise ValueError('Filename/patient-ID/test-index mismatch')
            if not np.allclose(part.y_true, truth[column].to_numpy(float), rtol=0, atol=1e-12):
                raise ValueError('Severity label mismatch')
            if not np.isfinite(part.y_pred).all():
                raise ValueError('Nonfinite baseline predictions')
    if len(selected) != expected_total:
        raise ValueError('Unexpected baseline prediction rows')
    return selected

def make_inner_splits(X, groups):
    groups=np.asarray(groups)
    count=len(np.unique(groups))
    if count < 2:
        raise ValueError('Need at least two training patients for inner CV')
    folds=list(GroupKFold(n_splits=min(5,count)).split(X,groups=groups))
    for tr,va in folds:
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError('Inner patient leakage')
    return folds

def fit_nested_ridge(X_train, y_train, training_groups, use_pca=False, alphas=ALPHAS):
    """API accepts training data only; scale/PCA are refit in each inner fold.

    Selection minimizes mean fold validation MSE, then refits on outer training
    data. Alpha grid and PCA candidates are prespecified, not test-tuned.
    """
    folds=make_inner_splits(X_train,training_groups)
    steps=[('scale',StandardScaler())]
    grid={'ridge__alpha':list(alphas)}
    if use_pca:
        capacity=min(min(len(tr) for tr,va in folds),X_train.shape[1])
        components=[n for n in (2,5,10,20) if n <= capacity]
        if not components:
            components=[1]
        steps.append(('pca',PCA(svd_solver='full')))
        grid['pca__n_components']=components
    steps.append(('ridge',Ridge(solver='svd')))
    search=GridSearchCV(Pipeline(steps),grid,scoring='neg_mean_squared_error',
                        cv=folds,refit=True,n_jobs=1,error_score='raise')
    start=perf_counter()
    search.fit(X_train,y_train)
    return search,folds,perf_counter()-start

def regression_metrics(truth,pred):
    return {'MAE':mean_absolute_error(truth,pred),
            'MSE':mean_squared_error(truth,pred),'R2':r2_score(truth,pred),
            'Pearson':np.nan if np.ptp(pred)==0 or np.ptp(truth)==0 else pearsonr(pred,truth)[0]}

def generate_baseline(logits,metadata):
    metrics,predictions,audits=evaluate_features({'lung opacity output':logits},metadata)
    predictions=predictions.drop(columns='features').assign(model=BASELINE)
    return predictions

def paired_alignment(predictions):
    keys=['target','seed','row_index']
    if predictions.duplicated(keys+['model']).any():
        raise ValueError('Duplicate model prediction keys')
    base=predictions[predictions.model==BASELINE]
    if base.empty:
        raise ValueError('Missing DenseNet baseline')
    for model,part in predictions.groupby('model'):
        joined=base.merge(part,on=keys,suffixes=('_base','_model'),validate='one_to_one',how='outer',indicator=True)
        if not (joined['_merge']=='both').all():
            raise ValueError('Models use different test observations')
        for col in ['filename','patientid','y_true']:
            if not (joined[col+'_base']==joined[col+'_model']).all():
                raise ValueError('Paired identity/label mismatch')
    return True

def compare_models(features,metadata,baseline_predictions,include_pca=False,alphas=ALPHAS):
    """Use one frozen feature matrix, exact outer splits and nested Ridge CV."""
    if len(features)!=len(metadata) or not np.isfinite(features).all():
        raise ValueError('Bad features')
    validate_heldout_records(baseline_predictions,metadata,'model',BASELINE)
    predictions=[baseline_predictions.copy()]
    timings=[]; inner_audits=[]
    for seed,tr,te in exact_splits(metadata):
        groups=metadata.iloc[tr].patientid.to_numpy()
        for target,column in TARGETS.items():
            y=metadata[column].to_numpy(float)
            for pca in ([False,True] if include_pca else [False]):
                name=PCA_RIDGE if pca else RIDGE
                search,folds,seconds=fit_nested_ridge(features[tr],y[tr],groups,pca,alphas)
                start=perf_counter();pred=search.predict(features[te]);inference_seconds=perf_counter()-start
                timings.append({'model':name,'target':target,'seed':seed,
                  'fit_and_search_seconds':seconds,'regression_predict_seconds':inference_seconds,
                  'best_params':str(search.best_params_), 'inner_best_MSE':-search.best_score_})
                if target==next(iter(TARGETS)) and not pca:
                    inner_audits.append({'seed':seed,'outer_train_indices':tr.tolist(),'outer_test_indices':te.tolist(),
                        'folds':[{'train_indices':tr[a].tolist(),'validation_indices':tr[b].tolist()} for a,b in folds]})
                predictions.append(pd.DataFrame({'model':name,'target':target,'seed':seed,
                    'row_index':te,'filename':metadata.iloc[te].filename.to_numpy(),
                    'patientid':metadata.iloc[te].patientid.to_numpy(),'y_true':y[te],'y_pred':pred}))
    combined=pd.concat(predictions,ignore_index=True)
    paired_alignment(combined)
    rows=[]
    for (model,target,seed),part in combined.groupby(['model','target','seed']):
        rows.append({'model':model,'target':target,'seed':seed,**regression_metrics(part.y_true,part.y_pred)})
    return pd.DataFrame(rows),combined,pd.DataFrame(timings),inner_audits

def create_resnet18(weights=True):
    """No fine-tuning; caller chooses torch hub cache outside repository."""
    import torch
    from torchvision.models import resnet18,ResNet18_Weights
    chosen=ResNet18_Weights.IMAGENET1K_V1 if weights else None
    model=resnet18(weights=chosen)
    model.fc=torch.nn.Identity()
    model.requires_grad_(False)
    model.eval()
    return model, ResNet18_Weights.IMAGENET1K_V1.transforms()
