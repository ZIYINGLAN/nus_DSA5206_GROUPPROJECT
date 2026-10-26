"""Regression fixtures reproduce NumPy checkpoint metadata; not pretrained results."""
import copy
import os
import ast
import pickle
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from critical_methods import sha256_file
from eva_x_methods import (load_checkpoint_safely, load_eva_tiny, checkpoint_metadata_report,
                           _factory_with_preloaded_state)

class UnapprovedMetadata:
    def __reduce__(self):
        return eval, ('1 + 1',)

class TestEVACheckpointLoading(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)/'checkpoint.pt'
    def save(self,raw):
        torch.save(raw,self.path)
        return sha256_file(self.path)
    def test_reproduce_reported_error_and_load_numpy_float64_safely(self):
        sha=self.save({'model':{'weight':torch.ones(2,3)},'best_auc':np.float64(.8)})
        with self.assertRaisesRegex(pickle.UnpicklingError,'numpy.core.multiarray.scalar'):
            torch.load(self.path,map_location='cpu',weights_only=True)
        before=list(torch.serialization.get_safe_globals())
        loaded=load_checkpoint_safely(self.path,sha)
        self.assertIsInstance(loaded['best_auc'],np.float64)
        torch.testing.assert_close(loaded['model']['weight'],torch.ones(2,3))
        self.assertEqual(before,torch.serialization.get_safe_globals())
        report=checkpoint_metadata_report(loaded)
        self.assertEqual(report['numpy_scalars'],[{'path':'checkpoint.best_auc','type':'float64','dtype':'float64'}])
    def test_bad_sha_never_deserializes(self):
        self.save({'best_auc':np.float64(.8)})
        with patch('torch.load',side_effect=AssertionError('Must not deserialize')) as load:
            with self.assertRaisesRegex(ValueError,'integrity mismatch'):
                load_checkpoint_safely(self.path,'0'*64)
            load.assert_not_called()
    def test_no_unrestricted_retry_for_unknown_global(self):
        sha=self.save({'metadata':UnapprovedMetadata()})
        original=torch.load
        before=list(torch.serialization.get_safe_globals())
        with patch('torch.load',wraps=original) as load:
            with self.assertRaisesRegex(pickle.UnpicklingError,'No unrestricted pickle fallback'):
                load_checkpoint_safely(self.path,sha)
            self.assertEqual(load.call_count,1)
            self.assertTrue(load.call_args.kwargs['weights_only'])
        self.assertEqual(before,torch.serialization.get_safe_globals())
    def test_unobserved_numpy_dtypes_fail_closed(self):
        sha=self.save({'metadata':np.float32(.8)})
        with self.assertRaisesRegex(pickle.UnpicklingError,'No unrestricted pickle fallback'):
            load_checkpoint_safely(self.path,sha)
    def test_scope_preserves_preexisting_allowlist_entries(self):
        sha=self.save({'score':np.float64(.5)})
        with torch.serialization.safe_globals([np.dtype]):
            before=list(torch.serialization.get_safe_globals())
            load_checkpoint_safely(self.path,sha)
            self.assertEqual(before,torch.serialization.get_safe_globals())
    def test_non_mapping_checkpoint_rejected(self):
        sha=self.save(torch.ones(2))
        with self.assertRaisesRegex(ValueError,'checkpoint mapping'):
            load_checkpoint_safely(self.path,sha)
    def test_factory_uses_validated_mapping_not_a_second_load(self):
        class TinyFixture(torch.nn.Module):
            def __init__(self):
                super().__init__();self.backbone=torch.nn.Linear(3,2)
                self.num_features=192;self.global_pool='avg'
        # Emulate the official factory's relevant I/O and state-loading structure.
        def factory(pretrained=False):
            model=TinyFixture()
            raw=torch.load(pretrained,map_location='cpu')
            model.load_state_dict(raw['model'],strict=False)
            return model
        module=types.SimpleNamespace(eva_x_tiny_patch16=factory,
            checkpoint_filter_fn=lambda raw,model:raw['model'])
        fixture=TinyFixture();state=copy.deepcopy(fixture.state_dict())
        sha=self.save({'model':state,'best_auc':np.float64(.7)})
        original_load=torch.load
        with patch('eva_x_methods.load_official_module',return_value=module),patch('torch.load',wraps=original_load) as load:
            model,audit=load_eva_tiny('fixture-only',self.path,sha)
            self.assertEqual(load.call_count,1)
            self.assertTrue(load.call_args.kwargs['weights_only'])
        self.assertIs(torch.load,original_load)
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertEqual(audit['backbone_coverage_fraction'],1.)
        self.assertFalse(audit['deserialization']['unrestricted_fallback'])
        self.assertIs(factory.__globals__['torch'],torch)
        for key,value in state.items():torch.testing.assert_close(model.state_dict()[key],value)
    def test_bad_sha_prevents_official_factory_execution(self):
        self.save({'model':{}})
        with patch('eva_x_methods.load_official_module') as loader:
            with self.assertRaises(ValueError):load_eva_tiny('unused',self.path,'0'*64)
            loader.assert_not_called()
    def test_same_factory_body_and_filter_receive_same_mapping(self):
        raw={'model':{'weight':torch.ones(2)}}
        def factory(pretrained=False):
            return torch.load(pretrained,map_location='cpu')
        module=types.SimpleNamespace(eva_x_tiny_patch16=factory)
        self.assertIs(_factory_with_preloaded_state(module,self.path,raw),raw)

    @unittest.skipUnless(os.environ.get('DUONG_EVA_OFFICIAL_SOURCE'),'Official source required for factory integration fixture')
    def test_actual_official_factory_numpy_metadata_and_full_coverage_fixture(self):
        # Real official factory/body, synthetic state mapping. This is NOT an
        # official pretrained checkpoint or a real-data smoke test.
        from eva_x_methods import load_official_module,pooled_features
        source=os.environ['DUONG_EVA_OFFICIAL_SOURCE'];module=load_official_module(source)
        tree=ast.parse(Path(source).read_text())
        factory=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='eva_x_tiny_patch16')
        model=eval(compile(ast.Expression(factory.body[0].value),'constructor fixture','eval'),vars(module))
        sha=self.save({'model':model.state_dict(),'best_auc':np.float64(.7)})
        loaded,audit=load_eva_tiny(source,self.path,sha)
        self.assertEqual(audit['backbone_coverage_fraction'],1.)
        self.assertTrue(audit['deserialization']['weights_only'])
        self.assertEqual(audit['checkpoint_metadata']['numpy_scalars'][0]['dtype'],'float64')
        torch.set_num_threads(2)
        first=pooled_features(loaded,torch.zeros(1,3,224,224))
        second=pooled_features(loaded,torch.zeros(1,3,224,224))
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        self.assertEqual(tuple(first.shape),(1,192))

if __name__=='__main__':unittest.main()
