import copy,os,pathlib,unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from PIL import Image
from critical_methods import exact_splits
from resnet18_methods import generate_baseline,compare_models,BASELINE,RIDGE,paired_alignment
from eva_x_methods import (checkpoint_coverage,evaluation_transform,PREPROCESSING,pooled_features,
    verify_prior_identity,evaluate_eva,EVA_MODEL,load_official_module,official_checkpoint)

class EVAComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(12)
        cls.metadata=pd.DataFrame({'filename':[f'x-{i}.png' for i in range(30)],
            'patientid':np.repeat(np.arange(15),2),'geographic_mean':rng.uniform(0,8,30),
            'opacity_mean':rng.uniform(0,6,30)})
        cls.eva_features=rng.normal(size=(30,192))
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            cls.earlier=compare_models(rng.normal(size=(30,8)),cls.metadata,
                generate_baseline(rng.normal(size=(30,1)),cls.metadata),alphas=(1,))[1]
    def test_coverage_rejects_missing_shape_and_unloaded_tensors(self):
        model=torch.nn.Sequential(torch.nn.Linear(3,2))
        state=copy.deepcopy(model.state_dict())
        self.assertEqual(checkpoint_coverage(model,state)['backbone_coverage_fraction'],1.)
        bad=dict(state);bad.pop('0.weight')
        with self.assertRaises(ValueError):checkpoint_coverage(model,bad)
        bad=dict(state);bad['0.weight']=torch.ones(3,3)
        with self.assertRaises(ValueError):checkpoint_coverage(model,bad)
        bad=dict(state);bad['0.weight']=state['0.weight']+1
        with self.assertRaises(ValueError):checkpoint_coverage(model,bad)
    def test_coverage_rejects_unexpected_and_empty_backbone(self):
        model=torch.nn.Linear(2,2);state=dict(model.state_dict());state['extra']=torch.zeros(1)
        with self.assertRaises(ValueError):checkpoint_coverage(model,state)
        with self.assertRaises(ValueError):checkpoint_coverage(torch.nn.Identity(),{})
    def test_unused_head_may_be_missing(self):
        model=torch.nn.Module();model.backbone=torch.nn.Linear(3,2);model.head=torch.nn.Linear(2,1)
        state={k:v for k,v in model.state_dict().items() if not k.startswith('head.')}
        audit=checkpoint_coverage(model,state)
        self.assertEqual(set(audit['ignored_missing_head_keys']),{'head.weight','head.bias'})
    def test_CXR_normalization_and_preprocessing_shape(self):
        image=Image.fromarray(np.full((280,240),128,np.uint8))
        tensor=evaluation_transform()(image.convert('RGB'))
        self.assertEqual(tuple(tensor.shape),(3,224,224))
        expected=(128/255-0.49185243)/0.28509309
        torch.testing.assert_close(tensor,torch.full_like(tensor,expected),rtol=1e-6,atol=1e-6)
        self.assertEqual(PREPROCESSING['interpolation'],'bicubic')
    def test_official_RGB_conversion_preserves_source_channels(self):
        rgb=np.zeros((256,256,3),np.uint8);rgb[:,:,1]=127;rgb[:,:,2]=255
        tensor=evaluation_transform()(Image.fromarray(rgb).convert('RGB'))
        expected=(np.array([0,127,255])/255-.49185243)/.28509309
        np.testing.assert_allclose(tensor[:,0,0].numpy(),expected,rtol=1e-6,atol=1e-6)
    def test_identity_mismatch_halts(self):
        d={'ordered_filenames':['x'],'ordered_patient_ids':[1],
           'ordered_labels':{'geographic_mean':[2],'opacity_mean':[1]},'metadata_sha256':'m',
           'scores_sha256':'s','scored_image_sha256':{'x':'i'}}
        self.assertTrue(verify_prior_identity(d,d))
        for key in d:
            other=copy.deepcopy(d);other[key]='wrong'
            with self.assertRaises(ValueError):verify_prior_identity(d,other)
    def test_local_checkpoint_needs_trusted_digest(self):
        with self.assertRaises(ValueError):official_checkpoint('.',local_path='no-such-file',expected_sha=None)
        import tempfile,hashlib
        with tempfile.TemporaryDirectory() as directory:
            path=pathlib.Path(directory)/'checksum-fixture.bin'
            path.write_bytes(b'unit-test checksum fixture, not model weights')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch('eva_x_methods.urllib.request.urlopen',side_effect=AssertionError('No network for local file')):
                loaded,info=official_checkpoint(directory,local_path=path,expected_sha=digest)
                self.assertEqual(loaded,path);self.assertEqual(info['sha256'],digest)
                with self.assertRaises(ValueError):official_checkpoint(directory,local_path=path,expected_sha='0'*64)

    def test_pooled_feature_guard(self):
        class Fixture(torch.nn.Module):
            def forward_features(self,x):return x
            def forward_head(self,x,pre_logits=False):
                assert pre_logits
                return torch.ones(len(x),192)
        model=Fixture().eval()
        a=pooled_features(model,torch.zeros(1,3,224,224));b=pooled_features(model,torch.zeros(1,3,224,224))
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        with self.assertRaises(ValueError):pooled_features(model,torch.zeros(1,1,224,224))
        model.train()
        with self.assertRaises(ValueError):pooled_features(model,torch.zeros(1,3,224,224))
    def test_full_synthetic_three_model_comparison(self):
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            metrics,pred,timing,audit=evaluate_eva(self.eva_features,self.metadata,self.earlier,alphas=(1,))
        self.assertEqual(len(metrics),300);self.assertEqual(len(timing),100);self.assertEqual(len(audit),50)
        self.assertEqual(set(pred.model),{BASELINE,RIDGE,EVA_MODEL});self.assertTrue(paired_alignment(pred))
        for a in audit:
            self.assertFalse(set(a['train_patient_ids'])&set(a['test_patient_ids']))
            for fold in a['inner_folds']:
                self.assertFalse(set(fold['train_indices'])&set(a['test_indices']))
                self.assertFalse(set(fold['validation_indices'])&set(a['test_indices']))
        # Check seed-0 EVA results against a direct call to the shared nested fitter.
        from resnet18_methods import fit_nested_ridge,regression_metrics
        _,tr,te=next(exact_splits(self.metadata))
        fitted,_,_=fit_nested_ridge(self.eva_features[tr],self.metadata.geographic_mean.to_numpy()[tr],
            self.metadata.iloc[tr].patientid.to_numpy(),alphas=(1,))
        expected=regression_metrics(self.metadata.geographic_mean.to_numpy()[te],fitted.predict(self.eva_features[te]))
        actual=metrics[(metrics.model==EVA_MODEL)&(metrics.target=='Geographic Extent')&(metrics.seed==0)].iloc[0]
        for name,value in expected.items():self.assertAlmostEqual(actual[name],value)
    def test_corrupt_earlier_predictions_rejected(self):
        bad=self.earlier.copy();bad.loc[0,'filename']='wrong.png'
        with self.assertRaises(ValueError):evaluate_eva(self.eva_features,self.metadata,bad,alphas=(1,))
    def test_eva_calls_shared_fitter_with_only_outer_training_rows(self):
        from resnet18_methods import fit_nested_ridge
        seen=[]
        def tracked(X,y,groups,*args):
            seen.append((X.copy(),y.copy(),groups.copy()))
            return fit_nested_ridge(X,y,groups,*args)
        with patch('eva_x_methods.fit_nested_ridge',side_effect=tracked):
            evaluate_eva(self.eva_features,self.metadata,self.earlier,alphas=(1,))
        splits=list(exact_splits(self.metadata))
        for i,(X,y,groups) in enumerate(seen):
            _,tr,te=splits[i//2]
            np.testing.assert_array_equal(X,self.eva_features[tr])
            np.testing.assert_array_equal(groups,self.metadata.iloc[tr].patientid)
            column='geographic_mean' if i%2==0 else 'opacity_mean'
            np.testing.assert_array_equal(y,self.metadata[column].to_numpy()[tr])
    @unittest.skipUnless(os.environ.get('DUONG_EVA_OFFICIAL_SOURCE'),'Set official source path for architecture-only test')
    def test_official_architecture_only_NOT_pretrained_smoke(self):
        source=os.environ['DUONG_EVA_OFFICIAL_SOURCE'];module=load_official_module(source)
        # Unit fixture instantiates the exact constructor expression in the official
        # factory. No synthetic checkpoint is passed off as a pretrained model.
        import ast
        tree=ast.parse(pathlib.Path(source).read_text())
        factory=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='eva_x_tiny_patch16')
        constructor=factory.body[0].value
        model=eval(compile(ast.Expression(constructor),'official factory constructor','eval'),vars(module))
        model.eval();model.requires_grad_(False);torch.set_num_threads(2)
        features=pooled_features(model,torch.zeros(1,3,224,224))
        repeated=pooled_features(model,torch.zeros(1,3,224,224))
        torch.testing.assert_close(features,repeated,rtol=0,atol=0)
        self.assertEqual(tuple(features.shape),(1,192))
        self.assertEqual(model.global_pool,'avg')

if __name__=='__main__':unittest.main()
