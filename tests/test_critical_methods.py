import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from scipy.stats import pearsonr
from critical_methods import tolerance_scores,exact_splits,evaluate_features,stability_summary,fingerprint

class CriticalAssessmentTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(12)
        self.meta=pd.DataFrame({'filename':[f'image-{i}' for i in range(24)],
          'patientid':np.repeat(np.arange(12),2),
          'geographic_mean':rng.uniform(0,8,24), 'opacity_mean':rng.uniform(0,6,24)})
        self.X=rng.normal(size=(24,4))
    def test_tolerance_directions_and_exact_boundary(self):
        scores=tolerance_scores(np.zeros(6),[-100,100,-.7,.7,-.6,.6])
        self.assertAlmostEqual(scores['one_sided_accuracy'],4/6)
        self.assertAlmostEqual(scores['absolute_accuracy'],2/6)
        self.assertAlmostEqual(scores['excess_accuracy'],2/6)
    def test_all_50_exact_splits_and_disjoint_patients(self):
        splits=list(exact_splits(self.meta));self.assertEqual(len(splits),50)
        for seed,tr,te in splits:
            expected=next(GroupShuffleSplit(train_size=.5,test_size=.5,random_state=seed).split(self.X,groups=self.meta.patientid))
            np.testing.assert_array_equal(tr,expected[0]);np.testing.assert_array_equal(te,expected[1])
            self.assertFalse(set(self.meta.iloc[tr].patientid)&set(self.meta.iloc[te].patientid))
            self.assertEqual(set(tr)|set(te),set(range(24)))
    def test_prediction_alignment_rejected(self):
        with self.assertRaises(ValueError):tolerance_scores([0],[1,2])
        with self.assertRaises(ValueError):tolerance_scores([],[])
        with self.assertRaises(ValueError):tolerance_scores([0],[np.nan])
    def test_group_validation(self):
        missing=self.meta.copy();missing.loc[0,'patientid']=np.nan
        with self.assertRaises(ValueError):list(exact_splits(missing))
        duplicate=self.meta.copy();duplicate.loc[0,'filename']=duplicate.loc[1,'filename']
        with self.assertRaises(ValueError):list(exact_splits(duplicate))
    def test_real_prediction_pipeline_on_synthetic_fixture(self):
        metrics,preds,audits=evaluate_features({'four':self.X,'No data':np.zeros((24,1))},self.meta)
        self.assertEqual(len(metrics),200);self.assertEqual(len(audits),50)
        seed,tr,te=next(exact_splits(self.meta))
        y=self.meta.geographic_mean.to_numpy()
        expected=LinearRegression().fit(self.X[tr],y[tr]).predict(self.X[te])
        actual=preds[(preds.target=='Geographic Extent')&(preds.features=='four')&(preds.seed==0)]
        np.testing.assert_array_equal(actual.row_index,te)
        np.testing.assert_allclose(actual.y_pred,expected)
        np.testing.assert_allclose(actual.y_true,y[te])
        row=metrics[(metrics.target=='Geographic Extent')&(metrics.features=='four')&(metrics.seed==0)].iloc[0]
        for key,value in {'MAE':mean_absolute_error(y[te],expected),'MSE':mean_squared_error(y[te],expected),'R2':r2_score(y[te],expected),'Pearson':pearsonr(expected,y[te])[0]}.items():
            self.assertAlmostEqual(row[key],value)
        self.assertTrue(metrics[metrics.features=='No data'].Pearson.isna().all())
        summary=stability_summary(metrics)
        self.assertEqual(summary.loc[('Opacity','No data'),('Pearson','count')],0)
        self.assertEqual(summary.loc[('Opacity','four'),('MAE','count')],50)
    def test_heldout_labels_do_not_affect_seed_zero_predictions(self):
        _,_,te=next(exact_splits(self.meta))
        first=evaluate_features({'four':self.X},self.meta)[1]
        changed=self.meta.copy();changed.loc[te,'geographic_mean']+=1000
        second=evaluate_features({'four':self.X},changed)[1]
        mask=lambda df:(df.seed==0)&(df.target=='Geographic Extent')
        np.testing.assert_allclose(first.loc[mask(first),'y_pred'],second.loc[mask(second),'y_pred'])
    def test_fingerprint_changes_with_provenance(self):
        self.assertEqual(fingerprint({'a':1,'b':2}),fingerprint({'b':2,'a':1}))
        self.assertNotEqual(fingerprint({'checkpoint':'a'}),fingerprint({'checkpoint':'b'}))
    def test_invalid_feature_alignment_rejected(self):
        with self.assertRaises(ValueError):evaluate_features({'bad':self.X[:-1]},self.meta)

if __name__=='__main__':unittest.main()
