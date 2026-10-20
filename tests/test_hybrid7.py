"""Semantic checks with nonzero readout; no learning claims."""
import pytest
import torch
from mapping.models.se2d.hybrid7 import Hybrid7, RegisterLayout
from mapping.models.se2c.hybrid7c import Hybrid7c
from mapping.models.se2d.hybrid4 import GatedDeltaNet, Attention, delta_scan
from mapping.task_se2d import TaskSpec
from mapping.task_se2c import TaskSpec as ContinuousSpec
from mapping.hybrid6_geometry import patch_calibration

@pytest.fixture(autouse=True)
def cpu_reference(monkeypatch):
    old = torch.get_num_threads(); torch.set_num_threads(1)
    import mapping.models.se2d.hybrid4 as backbone
    def dense(q,k,v,mask):
        b,h,n,_=q.shape
        allowed=mask.mask_mod(torch.arange(b)[:,None,None,None],torch.arange(h)[None,:,None,None],torch.arange(n)[None,None,:,None],torch.arange(k.shape[-2])[None,None,None,:])
        return torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=allowed)
    monkeypatch.setattr(backbone,'attend',dense)
    yield
    torch.set_num_threads(old)

def net(continuous=False, odometry_writes=False):
    torch.manual_seed(7)
    model=(Hybrid7c(ContinuousSpec(),128,query_mode='inert',inter_gdn_attention=True,odometry_writes=odometry_writes) if continuous else Hybrid7(TaskSpec(),128,query_mode='inert',inter_gdn_attention=True,odometry_writes=odometry_writes))
    for m in model.modules():
        if isinstance(m,GatedDeltaNet):m.scan=delta_scan
    torch.nn.init.normal_(model.readout.weight,std=.1)
    return model

def inputs(continuous=False):
    torch.manual_seed(9)
    obs=torch.rand(1,3,6,6);odo=torch.tensor([[3,10]]);query=torch.tensor([[[2,-3,1],[-4,2,3]]]);qt=torch.tensor([[0,2]])
    if continuous:odo=torch.tensor([[[1.2,-3.,.4],[-2.,1.,-.7]]]);query=query.float();query[...,2]*=torch.pi/2
    return obs,odo,query,qt

def cameras(model):
    obs,odo,qp,qt=inputs();trans,query,known=model._task_poses(odo,qp)
    from mapping.hybrid6_geometry import patchify
    return [patchify(obs),torch.eye(4).expand(1,3,4,4,4).clone(),patch_calibration(6,6,'cpu').expand(1,3,4,5).clone(),trans,query[:,:,None].expand(1,2,4,4,4).clone(),patch_calibration(6,6,'cpu').expand(1,2,4,5).clone(),qt]

@pytest.mark.parametrize('continuous',[False,True])
@pytest.mark.parametrize('shape',[(9,9),(15,15),(3,12)])
def test_shapes_backward(continuous,shape):
    m=net(continuous);x=list(inputs(continuous));x[0]=torch.rand(1,3,*shape)
    y=m(*x,query_shape=(3,9));assert y.shape==(1,2,3,9)
    y.square().mean().backward()
    for name in ('registers','pose_embed','sensor_embed','patch_embed'):
        g=getattr(m,name).weight.grad;assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0
    assert isinstance(m.blocks[0].mixer,Attention)
    assert isinstance(m.blocks[-1].mixer,Attention)
    assert len(m.blocks)==5 and isinstance(m.blocks[2].mixer,Attention)
    assert all(isinstance(m.blocks[i].mixer,GatedDeltaNet) for i in (1,3))


def test_mask_and_rope_phase_semantics():
    m=net();seen=[]
    handle=m.blocks[0].register_forward_pre_hook(lambda mod,a:seen.append(a))
    m(*inputs());handle.remove();_,cos,sin,layout=seen[0]
    phase=layout.phases[0];qid=layout.qid[0];L=len(phase);i=torch.arange(L)
    actual=layout.mask_mod()(0,0,i[:,None],i[None,:])
    expected=torch.zeros(L,L,dtype=torch.bool)
    for qi in range(L):
        for ki in range(L):
            expected[qi,ki]=((qid[ki]==0 and phase[ki]<=phase[qi]) or (qid[qi]>0 and qid[ki]==qid[qi]))
    assert torch.equal(actual,expected)
    for t in range(3):
        ix=(phase//3==t).nonzero().flatten();torch.testing.assert_close(cos[:,ix],cos[:,ix[:1]].expand(-1,len(ix),-1),rtol=0,atol=0)
    # Register0 cannot read obs0; obs patches see each other both ways.
    regs=((phase==0)&(qid==0)).nonzero().flatten();obs=((phase==1)&(qid==0)).nonzero().flatten()
    assert actual[regs[:,None],regs].all() and not actual[regs[:,None],obs].any()
    assert actual[obs[:,None],obs].all()


@pytest.mark.parametrize('odometry_writes',[False,True])
def test_history_permutation_and_query_equivariance(odometry_writes):
    m=net(odometry_writes=odometry_writes);a=cameras(m);perm=torch.tensor([3,0,2,1])
    with torch.no_grad():
        base=m.forward_cameras(*a)
        changed=list(a)
        for j in (0,1,2):changed[j]=a[j][:,:,perm]
        y=m.forward_cameras(*changed)
        torch.testing.assert_close(y,base,atol=5e-6,rtol=2e-5)
        changed=list(a)
        for j in (4,5):changed[j]=a[j][:,:,perm]
        y=m.forward_cameras(*changed)
        torch.testing.assert_close(y[:,:,perm.argsort()],base,atol=5e-6,rtol=2e-5)


@pytest.mark.parametrize('odometry_writes',[False,True])
def test_causality_and_inert_queries(odometry_writes):
    m=net(odometry_writes=odometry_writes);obs,odo,qp,qt=inputs()
    with torch.no_grad():
        base=m(obs,odo,qp,qt);future=obs.clone();future[:,1:]=1-future[:,1:]
        y=m(future,(odo+1)%16,qp,qt)
        torch.testing.assert_close(y[:,0],base[:,0],atol=3e-6,rtol=1e-5)
        altered=obs.clone();altered[:,0]=1-altered[:,0]
        assert (m(altered,odo,qp,qt)[:,0]-base[:,0]).abs().max()>1e-5
        alone=m(obs,odo,qp[:,1:],qt[:,1:]);torch.testing.assert_close(alone,base[:,1:],atol=5e-6,rtol=2e-5)
        empty=m(obs,odo,qp[:,:0],qt[:,:0]);assert empty.shape==(1,0,6,6)


@pytest.mark.parametrize('odometry_writes',[False,True])
def test_only_registers_convolve_write_and_decay(odometry_writes):
    m=net(odometry_writes=odometry_writes);counts=[];scans=[];handles=[]
    for mix in [b.mixer for b in m.blocks if isinstance(b.mixer,GatedDeltaNet)]:
        handles.append(mix.short.register_forward_pre_hook(lambda mod,a:counts.append(a[0].shape[-1])))
        old=mix.scan
        def scan(q,k,v,g,beta,old=old):scans.append((g.detach(),beta.detach()));return old(q,k,v,g,beta)
        mix.scan=scan
    layouts=[];handles.append(m.blocks[0].register_forward_pre_hook(lambda mod,a:layouts.append(a[-1])))
    m(*inputs())
    for h in handles:h.remove()
    layout=layouts[0];assert layout.hist.sum()==3*5+2*odometry_writes
    assert counts==[3*5+2*odometry_writes+3]*2  # convolution padding, no frame resets
    for g,beta in scans:
        assert torch.count_nonzero(g[~layout.hist])==0 and torch.count_nonzero(beta[~layout.hist])==0
        assert (beta[layout.hist]>0).all()


def test_cross_task_full_weight_compatibility():
    a=net();b=net(True);b.load_state_dict(a.state_dict(),strict=True)
    obs,odo,qp,qt=inputs();moves=torch.tensor([[-1,0],[0,1],[1,0],[0,-1]])
    continuous_odo=torch.cat((moves[odo%4]*9,(odo//4)[...,None]*torch.pi/2),-1).float();continuous_q=qp.float();continuous_q[...,2]*=torch.pi/2
    with torch.no_grad():torch.testing.assert_close(a(obs,odo,qp,qt),b(obs,continuous_odo,continuous_q,qt),atol=3e-6,rtol=1e-5)

@pytest.mark.parametrize('name',['se2d/hybrid7','se2c/hybrid7c','se2d/hybrid7_odom'])
def test_harness_and_checkpoint(name):
    import run
    from mapping.checkpoint import build_from_arch
    e=run.build(run.resolved(run.load([f'model={name}', 'model.query_mode=inert', 'model.scan_precision=ieee'])))
    spec=ContinuousSpec() if name.startswith('se2c') else TaskSpec()
    m=e.model.build(spec);restored=build_from_arch(e.model.record(),spec);restored.load_state_dict(m.state_dict(),strict=True)
    assert m.register_count==5 and m.query_mode=='inert'
    assert all(x.heads==6 and x.q.weight.shape[0]==192 for x in m.modules() if isinstance(x,GatedDeltaNet))
    assert all(x.scan_precision=='ieee' for x in m.modules() if isinstance(x,GatedDeltaNet))


def test_final_attention_requires_depth():
    with pytest.raises(ValueError, match='size S or larger'):
        Hybrid7(TaskSpec(),64,query_mode='inert')


def test_legacy_four_block_reconstruction():
    # Older architecture records omit the added option; preserve their graph.
    m=Hybrid7(TaskSpec(),128,query_mode='inert',final_attention=True)
    assert len(m.blocks)==4
    assert all(isinstance(m.blocks[i].mixer,GatedDeltaNet) for i in (1,2))


def test_odometry_variant_full_state_compatibility_and_transition():
    import copy
    from types import SimpleNamespace
    import run
    from train import validate_branch_transition
    source=run.build(run.resolved(run.load(['+experiment=hybrid7_initial'])))
    target=run.build(run.resolved(run.load(['+experiment=hybrid7_odometry'])))
    a=source.model.build(TaskSpec());b=target.model.build(TaskSpec())
    assert list(a.state_dict())==list(b.state_dict())
    b.load_state_dict(a.state_dict(),strict=True)
    assert [(n,p.shape) for n,p in a.named_parameters()]==[(n,p.shape) for n,p in b.named_parameters()]
    ck={'args':{'model':'hybrid7','task':'se2d'},'arch':source.model.record(),'step':30000,'opt':{}}
    args=SimpleNamespace(model='hybrid7_odom',task='se2d')
    validate_branch_transition(ck,target.model.record(),'hybrid7_odometry_writes',args)
    wrong=copy.deepcopy(target.model.record());wrong['kwargs']['gdn_heads']=8
    with pytest.raises(SystemExit,match='ONLY'):
        validate_branch_transition(ck,wrong,'hybrid7_odometry_writes',args)
    with pytest.raises(SystemExit,match='requires SE2D'):
        validate_branch_transition(ck,target.model.record(),'hybrid7_odometry_writes',SimpleNamespace(model='hybrid7',task='se2d'))


def test_whitelisted_adapter_transfer():
    import copy
    import run
    from train import hybrid7_adapter_transfer
    source=run.build(run.resolved(run.load(['+experiment=hybrid7_initial']))).model.record()
    target=run.build(run.resolved(run.load(['+experiment=hybrid7_se2c_transfer']))).model.record()
    assert hybrid7_adapter_transfer(source,target,'se2d','se2c')
    wrong=copy.deepcopy(target);wrong['kwargs']['gdn_heads']=8
    assert not hybrid7_adapter_transfer(source,wrong,'se2d','se2c')
    assert not hybrid7_adapter_transfer(source,target,'se2c','se2d')
    wrong=copy.deepcopy(target);wrong['kwargs']['odometry_writes']=True
    assert not hybrid7_adapter_transfer(source,wrong,'se2d','se2c')
