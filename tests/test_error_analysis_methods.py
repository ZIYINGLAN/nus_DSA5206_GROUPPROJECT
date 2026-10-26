"""Synthetic prediction-only fixtures, never clinical experimental results."""
import ast,copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from critical_methods import exact_splits
from error_analysis_methods import (MODELS,DENSE,RESNET,EVA,TARGETS,validate_predictions,
    add_errors,summarize_images,summarize_patients,analyze_predictions,severity_summary,
    representative_cases,export_analysis,FIGURE_NAMES,verified_prediction_filename)


def fixture():
    metadata=pd.DataFrame({'filename':[f'image-{i}.png' for i in range(6)],
        'patientid':['A','A','B','C','D','D'],'geographic_mean':[0.,1.,4.,6.,7.5,8.],
        'opacity_mean':[0.,1.,2.,4.,5.,6.]})
    manifest={'ordered_filenames':metadata.filename.tolist(),'ordered_patient_ids':metadata.patientid.tolist(),
        'ordered_labels':{c:metadata[c].tolist() for c in ['geographic_mean','opacity_mean']}}
    rows=[];audit=[]
    errors={DENSE:[1.8,.5,-1.,2.,-2.5,1.],RESNET:[2.,.8,-1.5,1.5,-1.8,2.],EVA:[.2,-1.5,2.5,-.3,-1.8,-.7]}
    for seed,tr,te in exact_splits(metadata):
        audit.append({'seed':seed,'train_indices':tr.tolist(),'test_indices':te.tolist(),
            'train_patient_ids':metadata.iloc[tr].patientid.tolist(),'test_patient_ids':metadata.iloc[te].patientid.tolist()})
        for target,column in [('Geographic Extent','geographic_mean'),('Opacity','opacity_mean')]:
            for index in te:
                for model in MODELS:
                    true=metadata.iloc[index][column]
                    rows.append({'model':model,'target':target,'seed':seed,'row_index':int(index),
                        'filename':metadata.iloc[index].filename,'patientid':metadata.iloc[index].patientid,
                        'y_true':true,'y_pred':true+errors[model][index]+(seed%3-1)*.05})
    return pd.DataFrame(rows),audit,manifest

class ErrorAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.records,self.audit,self.manifest=fixture()
    def validated(self):
        return validate_predictions(self.records,self.audit,self.manifest,expected_cohort=None)
    def test_exact_three_model_alignment_and_audit(self):
        frame,cohort,report=self.validated()
        self.assertEqual(set(frame.model),set(MODELS));self.assertEqual(len(cohort),6)
        self.assertEqual(report['audited_patients'],4);self.assertTrue(report['external_membership_audit_verified'])
        self.assertTrue((frame.groupby(['target','seed','row_index']).size()==3).all())
    def test_verified_export_name_from_real_notebook(self):
        self.assertEqual(verified_prediction_filename('EVA_X_Comparison.ipynb'),'three_model_heldout_predictions.csv')
    def test_duplicate_and_missing_model_rejected(self):
        for bad in [pd.concat([self.records,self.records.iloc[[0]]]),self.records.iloc[1:]]:
            with self.assertRaises(ValueError):validate_predictions(bad,self.audit,self.manifest,expected_cohort=None)
    def test_wrong_model_target_and_seeds_rejected(self):
        for col,value in [('model','fourth model'),('target','Opacity Score'),('seed',50),('row_index',.5)]:
            bad=self.records.copy()
            if col=='row_index':bad[col]=bad[col].astype(float)
            bad.loc[0,col]=value
            with self.assertRaises(ValueError):validate_predictions(bad,self.audit,self.manifest,expected_cohort=None)
    def test_nan_and_infinite_values_rejected(self):
        for col,value in [('y_pred',np.inf),('y_true',np.nan),('patientid',None)]:
            bad=self.records.copy();bad.loc[0,col]=value
            with self.assertRaises(ValueError):validate_predictions(bad,self.audit,self.manifest,expected_cohort=None)
    def test_changed_labels_across_appearances_rejected(self):
        bad=self.records.copy();mask=(bad.row_index==1)&(bad.seed==int(bad[bad.row_index==1].seed.iloc[0]))&(bad.target=='Opacity')
        bad.loc[mask,'y_true']+=.1
        with self.assertRaisesRegex(ValueError,'labels differ'):
            validate_predictions(bad,self.audit,None)
    def test_bad_patient_or_filename_mapping_rejected(self):
        for col,value in [('patientid','different'),('filename','wrong.png')]:
            bad=self.records.copy();bad.loc[0,col]=value
            with self.assertRaises(ValueError):validate_predictions(bad,self.audit,self.manifest,expected_cohort=None)
    def test_manifest_identity_and_historical_count_validation(self):
        with self.assertRaisesRegex(ValueError,'94 images / 65 patients'):
            validate_predictions(self.records,self.audit,self.manifest)
        for key in ['ordered_filenames','ordered_patient_ids']:
            bad=copy.deepcopy(self.manifest);bad[key][0]='wrong'
            with self.assertRaises(ValueError):validate_predictions(self.records,self.audit,bad,expected_cohort=None)
    def test_audit_membership_required_no_full_image_per_seed_assumption(self):
        with self.assertRaises(ValueError):validate_predictions(self.records,None,self.manifest,expected_cohort=None)
        bad=copy.deepcopy(self.audit);bad[0]['test_indices']=bad[0]['test_indices'][:-1]
        with self.assertRaises(ValueError):validate_predictions(self.records,bad,self.manifest,expected_cohort=None)
        frame,cohort,report=self.validated()
        self.assertTrue((frame[frame.model==EVA].groupby(['target','seed']).row_index.nunique()<6).all())
    def test_patient_leakage_and_changed_audited_ids_rejected(self):
        bad=copy.deepcopy(self.audit);bad[0]['train_patient_ids'][0]=bad[0]['test_patient_ids'][0]
        with self.assertRaises(ValueError):validate_predictions(self.records,bad,self.manifest,expected_cohort=None)
    def test_missing_entire_target_seed_rejected(self):
        bad=self.records[~((self.records.seed==49)&(self.records.target=='Opacity'))]
        with self.assertRaises(ValueError):validate_predictions(bad,self.audit,self.manifest,expected_cohort=None)
    def test_errors_and_paired_difference_direction(self):
        frame,_,_=self.validated();errors=add_errors(frame)
        np.testing.assert_allclose(errors.signed_error,errors.y_pred-errors.y_true)
        np.testing.assert_allclose(errors.absolute_error,np.abs(errors.signed_error))
        np.testing.assert_allclose(errors.squared_error,errors.signed_error**2)
        chosen=errors[(errors.row_index==0)&(errors.model==EVA)].iloc[0]
        self.assertAlmostEqual(chosen.delta_abs_error_eva_dense,-1.6)
    def test_equal_image_patient_estimand_and_undefined_std(self):
        frame,cohort,_=self.validated();rows=[]
        for idx,seed,error in [(0,0,1.),(0,1,-1.),(1,0,3.),(2,0,2.)]:
            source=frame[frame.row_index==idx].drop_duplicates(['target','model'])
            for row in source.to_dict('records'):
                row.update(seed=seed,y_pred=row['y_true']+error);rows.append(row)
        images,pairs=summarize_images(add_errors(pd.DataFrame(rows)))
        patients,_,_=summarize_patients(images,pairs,cohort)
        image=images[(images.target=='Opacity')&(images.model==EVA)&(images.row_index==0)].iloc[0]
        self.assertEqual(image.n_heldout,2);self.assertEqual(image.mean_signed_error,0.)
        self.assertEqual(image.mean_absolute_error,1.);self.assertAlmostEqual(image.std_signed_error,np.sqrt(2))
        self.assertEqual(image.overestimation_rate,.5)
        single=images[(images.row_index==1)&(images.model==EVA)].iloc[0]
        self.assertTrue(np.isnan(single.std_signed_error))
        patient=patients[(patients.target=='Opacity')&(patients.model==EVA)&(patients.patientid=='A')].iloc[0]
        self.assertEqual(patient.n_analyzed_images,2);self.assertEqual(patient.mean_absolute_error,2.)
        self.assertNotAlmostEqual(patient.mean_absolute_error,5/3)
        self.assertEqual(patient.mean_signed_error,1.5)
        self.assertEqual(patient.overestimation_rate,.75)
    def test_patient_pairing_counts_and_unequal_appearances(self):
        frame,cohort,_=self.validated();tables=analyze_predictions(frame,cohort)
        images=tables['image_level_errors'];patients=tables['patient_level_errors']
        self.assertEqual(len(images),6*2*3);self.assertEqual(len(patients),4*2*3)
        self.assertGreater(images.n_heldout.nunique(),1)
        paired=tables['patient_level_paired_differences']
        expected=tables['image_level_paired_differences'].query("patientid=='A' and target=='Opacity'").mean_delta_abs_error_eva_dense.mean()
        actual=paired.query("patientid=='A' and target=='Opacity'").mean_delta_abs_error_eva_dense.iloc[0]
        self.assertAlmostEqual(expected,actual)
        self.assertTrue((tables['patient_improvement_summary'].n_analyzed_patients==4).all())
    def test_case_selection_deterministic_under_input_shuffle(self):
        frame,cohort,_=self.validated();tables=analyze_predictions(frame,cohort)
        first=representative_cases(tables['image_level_errors'])
        second=representative_cases(tables['image_level_errors'].sample(frac=1,random_state=7))
        cols=['target','criterion','model','case_id']
        pd.testing.assert_frame_equal(first[cols].sort_values(cols).reset_index(drop=True),second[cols].sort_values(cols).reset_index(drop=True))
    def test_empty_case_categories_and_small_empty_strata(self):
        frame,cohort,_=self.validated();images=summarize_images(add_errors(frame))[0]
        images['mean_signed_error']=0.;images['mean_absolute_error']=0.
        images['mean_delta_abs_error_eva_dense']=0.;images['mean_delta_abs_error_eva_resnet']=0.
        cases=representative_cases(images)
        self.assertTrue((cases.selection_status=='no eligible case').all())
        subset=images[images.row_index==0]
        _,strata=severity_summary(subset)
        empty=strata[strata.n_images==0]
        self.assertTrue(empty.mean_absolute_error.isna().all())
        self.assertTrue(strata.small_stratum.all())
        self.assertTrue((strata.n_patients<=1).all())
    def test_full_export_synthetic_only_private_identifiers_and_six_figures(self):
        frame,cohort,report=self.validated();tables=analyze_predictions(frame,cohort)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inputs=root/'inputs';inputs.mkdir()
            prediction=inputs/'three_model_heldout_predictions.csv';self.records.to_csv(prediction,index=False)
            audit=inputs/'patient_split_audit.json';audit.write_text(json.dumps(self.audit))
            output=root/'analysis'
            manifest=export_analysis(tables,cohort,report,output,{'predictions':prediction,'audit':audit},self.manifest,synthetic_fixture=True)
            self.assertEqual(manifest['data_kind'],'synthetic_test_fixture')
            for name in FIGURE_NAMES:self.assertGreater((output/name).stat().st_size,1000)
            for name in tables:
                if name!='stratified_images':
                    exported=pd.read_csv(output/(name+'.csv'))
                    self.assertNotIn('patientid',exported);self.assertNotIn('filename',exported)
            self.assertTrue((output/'private_audit/identity_mapping.csv').is_file())
            self.assertTrue((output/'error_analysis_manifest.json').is_file())
            with self.assertRaises(FileExistsError):export_analysis(tables,cohort,report,output,{'predictions':prediction})
    def test_never_heldout_images_are_reported_not_imputed(self):
        template=self.records.drop_duplicates(['model','target','row_index'])
        template=template[template.row_index.isin([2,3,4,5])]
        frame=pd.concat([template.assign(seed=seed) for seed in range(50)],ignore_index=True)
        audit=[{'seed':seed,'train_indices':[0,1],'test_indices':[2,3,4,5],
            'train_patient_ids':['A','A'],'test_patient_ids':['B','C','D','D']} for seed in range(50)]
        validated,cohort,report=validate_predictions(frame,audit,self.manifest,expected_cohort=None)
        self.assertEqual(report['unobserved_images'],2);self.assertEqual(report['observed_patients'],3)
        tables=analyze_predictions(validated,cohort)
        self.assertEqual(tables['image_level_errors'].row_index.nunique(),4)
        self.assertEqual(tables['patient_level_errors'].patientid.nunique(),3)
    def test_bias_rates_are_image_or_patient_weighted_not_pooled(self):
        frame,cohort,_=self.validated();tables=analyze_predictions(frame,cohort)
        for level,key in [('image_equal_weight','image_level_errors'),('patient_equal_weight','patient_level_errors')]:
            summary=tables['signed_error_summary'].query('level==@level')
            for r in summary.itertuples():
                units=tables[key].query('target==@r.target and model==@r.model')
                self.assertAlmostEqual(r.overestimation_rate,units.overestimation_rate.mean())
                self.assertAlmostEqual(r.underestimation_rate,units.underestimation_rate.mean())
                self.assertEqual(r.n_units,len(units))
    def test_case_ties_use_smallest_row_index(self):
        frame,_,_=self.validated();images=summarize_images(add_errors(frame))[0]
        images['mean_delta_abs_error_eva_dense']=-1.
        cases=representative_cases(images)
        selected=cases[cases.criterion=='A_improvement']
        self.assertTrue((selected.row_index==0).all())
    def test_prediction_only_module_has_no_training_or_feature_calls(self):
        source=Path('error_analysis_methods.py').read_text();tree=ast.parse(source)
        forbidden={'fit','fit_transform','load_state_dict','forward_features','backward'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):self.assertNotIn(node.func.attr,forbidden)

if __name__=='__main__':unittest.main()
