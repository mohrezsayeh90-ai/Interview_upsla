"""Physics-informed parametric neural operator for a discrete resistor lattice.

Architecture: learned nodewise geodesic encoder, parametric coefficient MLP,
and a fixed-depth unrolled Kirchhoff-residual graph network. This is an
amortized learned iterative architecture, not a continuum-coordinate PINN.
Inference performs no sparse linear solve or per-instance weight fitting.
The fixed-depth layers do perform iterative updates of head for every input.
Connectivity and resistance-geodesic preprocessing remain explicit costs.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from flow_network import Lattice


@dataclass
class Prepared:
    g: np.ndarray
    transition: np.ndarray
    h0: np.ndarray
    free: np.ndarray
    features: np.ndarray
    parameters: np.ndarray
    floating: np.ndarray
    spanning: bool


class Preprocessor:
    """Combinatorial components and shortest resistance distances, never Lh=b."""
    def __init__(self,n=25):
        self.grid=Lattice(n)
        self.n=n
        self.nodes=n*n
        self.neighbor=np.tile(np.arange(self.nodes)[:,None],(1,4))
        self.edge_index=np.zeros((self.nodes,4),int)
        self.valid=np.zeros((self.nodes,4),bool)
        for e,(u,v) in enumerate(zip(self.grid.u,self.grid.v)):
            direction=1 if v==u+1 else 3
            self.neighbor[u,direction]=v
            self.neighbor[v,direction-1]=u
            self.edge_index[u,direction]=self.edge_index[v,direction-1]=e
            self.valid[u,direction]=self.valid[v,direction-1]=True

    def prepare(self,g,delta=0.25,p=0.5,sigma=1.0):
        g=np.asarray(g,dtype=np.float64)
        if g.shape!=(self.grid.edge_count,) or np.any(g<0) or not np.isfinite(g).all():
            raise ValueError('invalid finite nonnegative edge-conductance array')
        if not (0<=delta<=.5 and 0<=p<=1 and sigma>=0) or not np.isfinite([delta,p,sigma]).all():
            raise ValueError('invalid physical distribution parameters')
        on=g>0
        u,v=self.grid.u[on],self.grid.v[on]
        adjacency=coo_matrix((np.ones(2*len(u)),(np.r_[u,v],np.r_[v,u])),shape=(self.nodes,self.nodes)).tocsr()
        nc,labels=connected_components(adjacency,directed=False)
        left,right=np.zeros(nc,bool),np.zeros(nc,bool)
        left[labels[self.grid.left]]=True
        right[labels[self.grid.right]]=True
        active=(left&right)[labels]
        free=active & ~self.grid.boundary
        floating=~(left|right)[labels]
        h0=np.zeros(self.nodes)
        h0[left[labels]]=1
        h0[self.grid.left],h0[self.grid.right]=1.,0.
        # Canonicalize uniform conductance scale: exact head/flow equivariance.
        scale=np.median(g[on]) if np.any(on) else 1.
        canonical=g/scale
        directed=canonical[self.edge_index]*self.valid
        degree=directed.sum(-1)
        transition=directed/np.maximum(degree[:,None],1e-300)
        if np.any(active):
            resistance=1/canonical[on]
            radj=coo_matrix((np.r_[resistance,resistance],(np.r_[u,v],np.r_[v,u])),shape=(self.nodes,self.nodes)).tocsr()
            dl=dijkstra(radj,directed=False,indices=self.grid.left,min_only=True)
            dr=dijkstra(radj,directed=False,indices=self.grid.right,min_only=True)
            h0[active]=dr[active]/(dl[active]+dr[active])
            total=np.zeros(self.nodes)
            total[active]=np.log1p(dl[active]+dr[active])/8
        else:
            total=np.zeros(self.nodes)
        x=np.arange(self.nodes)%self.n/(self.n-1)
        y=np.arange(self.nodes)//self.n/(self.n-1)
        features=np.column_stack([h0,x,y,total,np.log1p(degree)/4,
            np.log1p(directed),directed>0,active,self.grid.boundary])
        # Input distribution parameters condition approximation, while g fixes physics.
        parameters=np.array([delta/.25,(p-.5)/.25,sigma/2,
            np.mean(on),np.mean(active),np.mean(np.log1p(degree))/4])
        return Prepared(g,transition,h0,free,features,parameters,floating,bool(np.any(active)))


def stack_prepared(rows,dtype=torch.float32):
    return {name:torch.as_tensor(np.stack([getattr(r,name) for r in rows]),
                                dtype=torch.bool if name=='free' else dtype)
            for name in ['g','transition','h0','free','features','parameters']}


class ParametricPINN(nn.Module):
    """A trained, fixed-depth graph network with a Dirichlet energy loss.

The last K layers are weighted graph message-passing layers driven by the
Kirchhoff residual. They resemble heavy-ball iteration by design. Their
coefficients and initial-head correction are learned once. Always disclose
depth and preprocessing; this is not a single-shot dense MLP or a closed form.
"""
    def __init__(self,n=25,width=32,depth=256):
        super().__init__()
        prep=Preprocessor(n)
        self.n,self.width,self.depth=n,width,depth
        self.register_buffer('neighbor',torch.as_tensor(prep.neighbor))
        self.register_buffer('edge_u',torch.as_tensor(prep.grid.u))
        self.register_buffer('edge_v',torch.as_tensor(prep.grid.v))
        self.encoder=nn.Sequential(nn.Linear(21,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,1))
        self.conditioner=nn.Sequential(nn.Linear(6,16),nn.Tanh(),nn.Linear(16,2))
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)
        nn.init.zeros_(self.conditioner[-1].weight)
        # beta=.96; alpha=1.90 at initialization, stable for normalized spectrum [0,2].
        b=.96/.999
        a=(1.90/1.96)/.999
        nn.init.constant_(self.conditioner[-1].bias[0],float(np.log(b/(1-b))))
        nn.init.constant_(self.conditioner[-1].bias[1],float(np.log(a/(1-a))))

    def coefficients(self,parameters):
        c=self.conditioner(parameters)
        beta=.999*torch.sigmoid(c[:,0:1])
        alpha=(1+beta)*(.999*torch.sigmoid(c[:,1:2]))
        return alpha,beta

    def forward(self,batch,depth=None,learned=True):
        h0=batch['h0']
        fixed=h0
        if learned:
            params=batch['parameters'][:,None,:].expand(-1,h0.shape[1],-1)
            correction=torch.tanh(self.encoder(torch.cat([batch['features'],params],dim=-1)).squeeze(-1))
            h=h0+h0*(1-h0)*correction
            alpha,beta=self.coefficients(batch['parameters'])
        else:
            h=h0
            alpha,beta=1.90,.96
        h=torch.where(batch['free'],h,fixed)
        previous=h
        for _ in range(self.depth if depth is None else depth):
            average=(batch['transition']*h[:,self.neighbor]).sum(-1)
            residual=h-average
            nxt=torch.where(batch['free'],h-alpha*residual+beta*(h-previous),fixed)
            previous,h=h,nxt
        # Clipping preserves Dirichlet values and cannot increase graph energy.
        h=h.clamp(0,1)
        return h

    def energy(self,batch,h):
        return (batch['g']*(h[:,self.edge_u]-h[:,self.edge_v]).square()).sum(-1)

    def residual(self,batch,h):
        r=h-(batch['transition']*h[:,self.neighbor]).sum(-1)
        return torch.where(batch['free'],r,torch.zeros_like(r))


class Predictor:
    """Re-usable public API. Physical heads and conductance scale are exact factors."""
    def __init__(self,checkpoint,depth=2048,threads=2):
        torch.set_num_threads(threads)
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        config=state['config']
        self.preprocessor=Preprocessor(config['n'])
        self.model=ParametricPINN(config['n'],config['width'],config['depth']).double().eval()
        self.model.load_state_dict(state['state_dict'])
        self.depth=depth

    @torch.inference_mode()
    def predict(self,conductances,delta=.25,p=.5,sigma=1.,head_left=1.,head_right=0.):
        row=self.preprocessor.prepare(conductances,delta,p,sigma)
        batch=stack_prepared([row],dtype=torch.float64)
        h=self.model(batch,depth=self.depth)
        geff=float(self.model.energy(batch,h)[0])
        unit=h[0].numpy()
        result_heads=head_right+(head_left-head_right)*unit
        result_heads[row.floating]=np.nan
        grid=self.preprocessor.grid
        q=row.g*(unit[grid.u]-unit[grid.v])
        balance=np.bincount(grid.u,weights=q,minlength=grid.n**2)-np.bincount(grid.v,weights=q,minlength=grid.n**2)
        return {'geff':geff,'total_flow':geff*(head_left-head_right),
                'heads':result_heads,'spanning':row.spanning,
                'geff_estimator':'Dirichlet energy / unit head difference squared',
                'inlet_flow_from_heads':float(balance[grid.left].sum())*(head_left-head_right),
                'outlet_flow_from_heads':float(-balance[grid.right].sum())*(head_left-head_right),
                'max_interior_residual':float(np.abs(balance[~grid.boundary]).max())*abs(head_left-head_right),
                'in_training_domain':bool(self.model.n==25 and 0<=delta<=.25 and abs(p-.5)<=delta+1e-12 and 0<=sigma<=2),
                'depth':self.depth}

    @torch.inference_mode()
    def predict_batch(self,conductances,delta=.25,p=.5,sigma=1.,batch_size=64):
        fields=np.asarray(conductances)
        if fields.ndim!=2 or fields.shape[1]!=self.preprocessor.grid.edge_count:
            raise ValueError('conductances must have shape (batch, number_of_edges)')
        count=len(fields)
        delta,p,sigma=[np.broadcast_to(np.asarray(x),(count,)) for x in [delta,p,sigma]]
        values=[]
        for start in range(0,count,batch_size):
            stop=min(start+batch_size,count)
            rows=[self.preprocessor.prepare(fields[i],delta[i],p[i],sigma[i]) for i in range(start,stop)]
            result=np.zeros(len(rows))
            active=[i for i,r in enumerate(rows) if r.spanning]
            if active:
                batch=stack_prepared([rows[i] for i in active],dtype=torch.float64)
                h=self.model(batch,depth=self.depth)
                result[active]=self.model.energy(batch,h).numpy()
            values.append(result)
        return np.concatenate(values) if values else np.empty(0)
