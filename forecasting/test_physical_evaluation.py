"""Analytical checks for diagnostics; no production data/GPU required."""
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from forecasting.physical_evaluation import autocorrelation, increment_pdf, summarize, normalization_matches


class DiagnosticsTests(unittest.TestCase):
    def test_normalization_roundoff_and_real_mismatch(self):
        original=dict(mean=[3e-8, -1.6e-8, 2e-8], std=[1.3037199143119635, 1.37, 1.25],
                      count_per_channel=100, file_ids=[131,132])
        rounded=dict(original, mean=[3e-8,-1.600000000000001e-8,2e-8],
                     std=[1.3037199143119638,1.37,1.25])
        self.assertTrue(normalization_matches(original,rounded))
        for changed in (dict(original,std=[1.30,1.37,1.25]),
                        dict(original,mean=[1e-5,-1.6e-8,2e-8]),
                        dict(original,count_per_channel=101),
                        dict(original,file_ids=[131,133]),
                        dict(original,std=[float('nan'),1.37,1.25])):
            self.assertFalse(normalization_matches(original,changed))

    def test_frozen_and_sign_reversal(self):
        rng=np.random.default_rng(10)
        field=rng.normal(size=(3,8,8,8))
        frozen=np.repeat(field[None],8,axis=0)
        np.testing.assert_allclose(autocorrelation(frozen,(1,2),5),1,atol=1e-12)
        frozen[1::2]*=-1
        np.testing.assert_allclose(autocorrelation(frozen,(1,2),5),
                                  np.tile([1,-1,1,-1,1,-1],(3,1)),atol=1e-12)

    def test_against_direct_filtered_dot_products(self):
        rng=np.random.default_rng(20); fields=rng.normal(size=(7,3,8,8,8))
        m=np.fft.fftfreq(8)*8
        radius=np.sqrt(m[:,None,None]**2+m[None,:,None]**2+m[None,None,:]**2)
        z=np.fft.ifftn(np.fft.fftn(fields,axes=(-3,-2,-1))*((radius>1)&(radius<=2)),axes=(-3,-2,-1)).real
        direct=[]
        for lag in range(4):
            left=z[:len(z)-lag]; right=z[lag:]
            direct.append(np.sum(left*right)/np.sqrt(np.sum(left**2)*np.sum(right**2)))
        np.testing.assert_allclose(autocorrelation(fields,(1,2),3)[1],direct,atol=1e-12)

    def test_pdf_keeps_outside_probability(self):
        field=np.zeros((3,8,8,8)); field[0,::2]=10
        edges=np.linspace(-1,1,11)
        pdf,tails=increment_pdf(field,[1,2],edges)
        np.testing.assert_allclose((pdf*np.diff(edges)).sum(-1)+tails.sum(-1),1)
        self.assertGreater(tails[0].sum(),0)
        self.assertEqual(tails[1].sum(),0)

    def test_summary_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'spec.json').write_text(json.dumps(dict(starts=[0],horizon=2,separations=[1])))
            arrays=dict(times=np.array([1.,2.]),mse=np.ones(2),normalized_mse=np.ones(2),
                        lag_times=np.array([0.,1.]),pdf_leads=np.array([1,2]),pdf_edges=np.linspace(-1,1,5))
            for label in ('actual','predicted'):
                arrays[label+'_spectrum']=np.ones((2,8))
                arrays[label+'_correlation']=np.ones((3,2))
                arrays[label+'_pdf']=np.ones((2,1,4))/2
                arrays[label+'_speed_slice']=np.ones((2,8,8))
            for name in ('matey','yalamanchi'):
                folder=root/name; folder.mkdir()
                (folder/'complete.json').write_text('{}')
                np.savez(folder/'origin-000.npz',**arrays)
            summarize(SimpleNamespace(output=directory))
            self.assertEqual(len(list(root.glob('*.png'))),5)
            self.assertTrue((root/'mse.csv').exists())
            self.assertTrue((root/'README.md').exists())


if __name__ == '__main__':
    unittest.main()
