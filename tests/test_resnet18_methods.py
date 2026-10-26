import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from scipy.stats import pearsonr
from critical_methods import exact_splits
from resnet18_methods import (aligned_features,validate_heldout_records,make_inner_splits,
    fit_nested_ridge,regression_metrics,generate_baseline,paired_alignment,compare_models,
    create_resnet18,BASELINE,RIDGE)

class ResNetComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(30)
        cls.metadata=pd.DataFrame({'filename':[f'cxr-{i}.png' for i in range(30)],
          'patientid':np.repeat(np.arange(15),2),'geographic_mean':rng.uniform(0,8,30),
          'opacity_mean':rng.uniform(0,6,30)})
        cls.X=rng.normal(size=(30,8))
        cls.baseline=generate_baseline(cls.X[:,[0]],cls.metadata)
    def test_ordered_feature_alignment(self):
        aligned_features(self.X,self.metadata.filename,self.metadata,8)
        with self.assertRaises(ValueError):aligned_features(self.X,self.metadata.filename[::-1],self.metadata,8)
        with self.assertRaises(ValueError):aligned_features(self.X[:,:7],self.metadata.filename,self.metadata,8)
        bad=self.X.copy();bad[0,0]=np.inf
        with self.assertRaises(ValueError):aligned_features(bad,self.metadata.filename,self.metadata,8)
    def test_all_outer_splits_match_baseline_csv_records(self):
        audit=validate_heldout_records(self.baseline.sample(frac=1,random_state=4),self.metadata,'model',BASELINE)
        self.assertEqual(len(audit),len(self.baseline))
        self.assertEqual(len(list(exact_splits(self.metadata))),50)
    def test_record_corruption_rejected(self):
        for column,value in [('filename','wrong.png'),('patientid',999),('y_true',999),('row_index',999)]:
            bad=self.baseline.copy();bad.loc[0,column]=value
            with self.assertRaises(ValueError):validate_heldout_records(bad,self.metadata,'model',BASELINE)
        with self.assertRaises(ValueError):validate_heldout_records(self.baseline.iloc[1:],self.metadata,'model',BASELINE)
    def test_inner_folds_are_outer_training_patients_only(self):
        _,tr,te=next(exact_splits(self.metadata))
        groups=self.metadata.iloc[tr].patientid.to_numpy()
        for it,iv in make_inner_splits(self.X[tr],groups):
            self.assertFalse(set(groups[it])&set(groups[iv]))
            self.assertFalse(set(tr[it])&set(te));self.assertFalse(set(tr[iv])&set(te))
    def test_scaler_fitted_on_outer_train_and_inside_fold(self):
        _,tr,te=next(exact_splits(self.metadata))
        search,folds,_=fit_nested_ridge(self.X[tr],self.metadata.geographic_mean.to_numpy()[tr],
            self.metadata.iloc[tr].patientid.to_numpy(),alphas=(.1,10))
        from sklearn.base import clone
        for a,b in folds:
            pipeline=clone(search.best_estimator_).fit(self.X[tr[a]],self.metadata.geographic_mean.to_numpy()[tr[a]])
            np.testing.assert_allclose(pipeline['scale'].mean_,self.X[tr[a]].mean(axis=0))
        np.testing.assert_allclose(search.best_estimator_['scale'].mean_,self.X[tr].mean(axis=0))
        self.assertIsNotNone(search.best_estimator_.named_steps['ridge'])
    def test_outer_test_feature_and_label_changes_do_not_change_selection(self):
        _,tr,te=next(exact_splits(self.metadata))
        y=self.metadata.geographic_mean.to_numpy().copy();X=self.X.copy()
        groups=self.metadata.iloc[tr].patientid.to_numpy()
        first,_,_=fit_nested_ridge(X[tr],y[tr],groups,alphas=(.1,10))
        X[te]+=10000;y[te]-=999
        second,_,_=fit_nested_ridge(X[tr],y[tr],groups,alphas=(.1,10))
        self.assertEqual(first.best_params_,second.best_params_)
        np.testing.assert_allclose(first.best_estimator_['ridge'].coef_,second.best_estimator_['ridge'].coef_)
    def test_optional_pca_fold_capacity_and_train_fit(self):
        _,tr,te=next(exact_splits(self.metadata))
        search,folds,_=fit_nested_ridge(self.X[tr],self.metadata.geographic_mean.to_numpy()[tr],
            self.metadata.iloc[tr].patientid.to_numpy(),use_pca=True,alphas=(1,))
        capacity=min(min(len(a) for a,b in folds),self.X.shape[1])
        self.assertTrue(all(n<=capacity for n in search.param_grid['pca__n_components']))
        scaled=search.best_estimator_['scale'].transform(self.X[tr])
        np.testing.assert_allclose(search.best_estimator_['pca'].mean_,scaled.mean(axis=0),atol=1e-12)
    def test_metrics_independent_reference(self):
        truth=np.array([0.,1.,3.,6.]);pred=np.array([.2,2.,2.5,4.])
        actual=regression_metrics(truth,pred)
        for key,value in [('MAE',mean_absolute_error(truth,pred)),('MSE',mean_squared_error(truth,pred)),
                          ('R2',r2_score(truth,pred)),('Pearson',pearsonr(pred,truth)[0])]:
            self.assertAlmostEqual(actual[key],value)
        self.assertTrue(np.isnan(regression_metrics(truth,np.zeros(4))['Pearson']))
    def test_paired_alignment_rejects_mismatch(self):
        both=pd.concat([self.baseline,self.baseline.assign(model=RIDGE)],ignore_index=True)
        self.assertTrue(paired_alignment(both))
        both.loc[len(self.baseline),'filename']='wrong'
        with self.assertRaises(ValueError):paired_alignment(both)
    def test_frozen_resnet_shape_without_download(self):
        import torch
        model,transform=create_resnet18(weights=False)
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        torch.set_num_threads(2)
        with torch.inference_mode():out=model(torch.zeros(2,3,224,224))
        self.assertEqual(tuple(out.shape),(2,512));self.assertTrue(torch.isfinite(out).all())
    def test_full_50_split_synthetic_comparison(self):
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            metrics,predictions,timings,audits=compare_models(self.X,self.metadata,self.baseline,alphas=(.1,10))
        self.assertEqual(len(metrics),200);self.assertEqual(len(timings),100);self.assertEqual(len(audits),50)
        self.assertTrue(paired_alignment(predictions))
        for a in audits:
            test=set(a['outer_test_indices'])
            for fold in a['folds']:
                self.assertFalse(test&set(fold['train_indices']))
                self.assertFalse(test&set(fold['validation_indices']))
        self.assertTrue(np.isfinite(metrics[['MAE','MSE','R2']]).all().all())

if __name__=='__main__':unittest.main()
