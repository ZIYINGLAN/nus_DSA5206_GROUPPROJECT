"""Critical-assessment evaluation, independent of notebook execution and plotting."""
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ConstantInputWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

TARGETS = {'Geographic Extent': 'geographic_mean', 'Opacity': 'opacity_mean'}
SPLIT_CONFIG = {'train_size': .5, 'test_size': .5, 'seeds': list(range(50))}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def tolerance_scores(y_true, y_pred):
    truth, prediction = np.asarray(y_true), np.asarray(y_pred)
    if truth.ndim != 1 or prediction.shape != truth.shape:
        raise ValueError('Truth and prediction must have identical one-dimensional shape')
    error = prediction - truth
    if error.ndim != 1 or not np.isfinite(error).all() or not len(error):
        raise ValueError('Expected aligned finite nonempty one-dimensional predictions')
    return {'one_sided_accuracy': float(np.mean(error < .7)),
            'absolute_accuracy': float(np.mean(np.abs(error) < .7)),
            'excess_accuracy': float(np.mean(error <= -.7))}

def exact_splits(metadata):
    """Table 1 split settings; group values and ordered rows are not recoded/sorted."""
    if metadata.filename.duplicated().any() or metadata.patientid.isna().any():
        raise ValueError('Require unique filenames and complete patient IDs')
    for seed in SPLIT_CONFIG['seeds']:
        tr, te = next(GroupShuffleSplit(train_size=.5, test_size=.5,
                                       random_state=seed).split(metadata, groups=metadata.patientid))
        overlap = set(metadata.iloc[tr].patientid) & set(metadata.iloc[te].patientid)
        if overlap:
            raise AssertionError(f'Patient leakage in seed {seed}: {overlap}')
        yield seed, tr, te

def evaluate_features(features, metadata):
    """Fit only training rows; save actual held-out predictions for both targets."""
    for col in TARGETS.values():
        if not np.isfinite(metadata[col].to_numpy(dtype=float)).all():
            raise ValueError('Missing or nonfinite severity labels')
    matrices = {k: np.asarray(v) for k,v in features.items()}
    for X in matrices.values():
        if X.ndim != 2 or len(X) != len(metadata) or not np.isfinite(X).all():
            raise ValueError('Feature/label alignment or finite-value failure')
    rows, predictions, audits = [], [], []
    for seed, tr, te in exact_splits(metadata):
        audits.append({'seed': seed, 'train_indices': tr.tolist(), 'test_indices': te.tolist(),
                       'train_filenames': metadata.iloc[tr].filename.tolist(),
                       'test_filenames': metadata.iloc[te].filename.tolist(),
                       'train_patient_ids': metadata.iloc[tr].patientid.tolist(),
                       'test_patient_ids': metadata.iloc[te].patientid.tolist(),
                       'overlap_count': 0})
        for target, column in TARGETS.items():
            y = metadata[column].to_numpy(dtype=float)
            for name,X in matrices.items():
                model = LinearRegression().fit(X[tr],y[tr])
                yp = model.predict(X[te])
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', ConstantInputWarning)
                    corr = pearsonr(yp,y[te])[0]
                rows.append({'target': target, 'features': name, 'seed': seed,
                             'MAE': mean_absolute_error(y[te],yp),
                             'MSE': mean_squared_error(y[te],yp),
                             'R2': r2_score(y[te],yp), 'Pearson': corr,
                             **tolerance_scores(y[te],yp)})
                for idx,pred in zip(te,yp):
                    predictions.append({'target':target, 'features':name, 'seed':seed,
                                        'row_index':int(idx), 'filename':metadata.iloc[idx].filename,
                                        'patientid':metadata.iloc[idx].patientid,
                                        'y_true':float(y[idx]), 'y_pred':float(pred)})
    return pd.DataFrame(rows), pd.DataFrame(predictions), audits

def stability_summary(metrics):
    """Descriptive variability, not confidence intervals from independent samples."""
    return metrics.groupby(['target','features'])[['MAE','MSE','R2','Pearson']].agg(
        ['count','mean','std','median','min',lambda x:x.quantile(.05),lambda x:x.quantile(.95),'max']
    ).rename(columns={'<lambda_0>':'p05','<lambda_1>':'p95'})
