"""Physical invariants and proof that prediction never calls the sparse solver."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from flow_network import Lattice
from parametric_pinn.model import Preprocessor,ParametricPINN,stack_prepared,Predictor

torch.set_num_threads(2)


def test_geodesic_and_boundaries():
    pre=Preprocessor(9)
    g=np.ones(pre.grid.edge_count)
    row=pre.prepare(g)
    np.testing.assert_allclose(row.h0,np.tile(np.linspace(1,0,9),9),atol=1e-14)
    batch=stack_prepared([row],dtype=torch.float64)
    model=ParametricPINN(9,depth=16).double()
    h=model(batch)
    np.testing.assert_array_equal(h.detach().numpy()[0,pre.grid.left],1)
    np.testing.assert_array_equal(h.detach().numpy()[0,pre.grid.right],0)
    assert model.energy(batch,h).item()==pytest.approx(9/8,abs=1e-12)


def test_disconnected_zero_is_exact():
    pre=Preprocessor(7)
    g=np.zeros(pre.grid.edge_count)
    g[14]=3
    row=pre.prepare(g)
    model=ParametricPINN(7,depth=8).double()
    batch=stack_prepared([row],dtype=torch.float64)
    assert model.energy(batch,model(batch)).item()==0
    assert not row.spanning and row.floating.any()


@pytest.mark.parametrize('seed',[10,21,39])
def test_variational_bound_and_head_bounds(seed):
    pre=Preprocessor(11)
    g=pre.grid.generate(.65,2,np.random.default_rng(seed))
    row=pre.prepare(g)
    batch=stack_prepared([row],dtype=torch.float64)
    model=ParametricPINN(11,depth=128).double()
    h=model(batch)
    assert h.min()>=0 and h.max()<=1
    exact=pre.grid.solve(g).conductance
    assert model.energy(batch,h).item()>=exact-1e-10


def test_energy_gradient_is_kirchhoff_residual():
    pre=Preprocessor(5)
    g=np.exp(np.random.default_rng(82).normal(size=pre.grid.edge_count))
    row=pre.prepare(g)
    model=ParametricPINN(5,depth=0).double()
    batch=stack_prepared([row],dtype=torch.float64)
    h=torch.tensor(np.random.default_rng(77).random((1,25)),requires_grad=True,dtype=torch.float64)
    energy=model.energy(batch,h).sum()
    gradient=torch.autograd.grad(energy,h)[0].numpy()[0]
    q=g*(h.detach().numpy()[0,pre.grid.u]-h.detach().numpy()[0,pre.grid.v])
    balance=np.bincount(pre.grid.u,weights=q,minlength=25)-np.bincount(pre.grid.v,weights=q,minlength=25)
    np.testing.assert_allclose(gradient,2*balance,atol=2e-14)


def test_transition_and_scale_equivariance():
    pre=Preprocessor(9)
    g=pre.grid.generate(.7,1.5,np.random.default_rng(89))
    a,b=pre.prepare(g),pre.prepare(13*g)
    np.testing.assert_allclose(a.h0,b.h0,atol=1e-14)
    np.testing.assert_allclose(a.features,b.features,atol=1e-14)
    np.testing.assert_allclose(a.transition.sum(1)[a.free],1,atol=1e-14)
    model=ParametricPINN(9,depth=64).double()
    aa,bb=stack_prepared([a],dtype=torch.float64),stack_prepared([b],dtype=torch.float64)
    ha,hb=model(aa),model(bb)
    np.testing.assert_allclose(ha.detach(),hb.detach(),atol=1e-12)
    assert model.energy(bb,hb).item()==pytest.approx(13*model.energy(aa,ha).item(),rel=1e-12)


def test_batch_independence():
    pre=Preprocessor(7)
    rng=np.random.default_rng(932)
    rows=[pre.prepare(pre.grid.generate(.7,1,rng)) for _ in range(3)]
    model=ParametricPINN(7,depth=32).double().eval()
    together=model(stack_prepared(rows,dtype=torch.float64)).detach().numpy()
    separate=np.concatenate([model(stack_prepared([r],dtype=torch.float64)).detach().numpy() for r in rows])
    np.testing.assert_allclose(together,separate,atol=1e-13)


def test_no_sparse_solve_in_predict(monkeypatch,tmp_path):
    config={'n':7,'width':32,'depth':32}
    model=ParametricPINN(**config)
    path=tmp_path/'weights.pt'
    torch.save({'config':config,'state_dict':model.state_dict()},path)
    def forbidden(*args,**kwargs):
        raise AssertionError('prediction invoked the reference linear solver')
    monkeypatch.setattr(Lattice,'solve',forbidden)
    predictor=Predictor(path,depth=64)
    g=np.ones(predictor.preprocessor.grid.edge_count)
    r=predictor.predict(g,head_left=3,head_right=1)
    assert r['geff']==pytest.approx(7/6,abs=1e-10)
    assert r['total_flow']==pytest.approx(2*7/6,abs=1e-10)
    np.testing.assert_allclose(r['heads'][predictor.preprocessor.grid.left],3)
    np.testing.assert_allclose(r['heads'][predictor.preprocessor.grid.right],1)


def test_invalid_fields():
    pre=Preprocessor(5)
    with pytest.raises(ValueError):pre.prepare(np.full(pre.grid.edge_count,-1))
    with pytest.raises(ValueError):pre.prepare(np.ones(3))


def test_training_generator_requires_no_reference_labels(monkeypatch):
    from parametric_pinn.scripts.train import generate
    def forbidden(*args,**kwargs):
        raise AssertionError('training data generation called the reference solver')
    monkeypatch.setattr(Lattice,'solve',forbidden)
    data,labels,info=generate(8,999,n=7,reference=False)
    assert labels is None and len(data['g'])==8 and info['retained_spanning']==8


def test_public_batch_matches_single_predictions(tmp_path):
    config={'n':7,'width':32,'depth':32}
    model=ParametricPINN(**config)
    path=tmp_path/'weights.pt'
    torch.save({'config':config,'state_dict':model.state_dict()},path)
    predictor=Predictor(path,depth=64)
    rng=np.random.default_rng(765)
    fields=np.stack([predictor.preprocessor.grid.generate(.7,1,rng),
                     np.zeros(predictor.preprocessor.grid.edge_count)])
    batch=predictor.predict_batch(fields,p=.7,sigma=1)
    single=np.array([predictor.predict(g,p=.7,sigma=1)['geff'] for g in fields])
    np.testing.assert_allclose(batch,single,atol=1e-13)
    assert batch[1]==0
