"""CVIU fair input calibration. No changes to legacy architecture/checkpoints."""
import itertools
import torch
import torch.nn.functional as F
from dataset_loader.dataloader import _gaussian_blur_torch


def degrade(x, size):
    return F.interpolate(_gaussian_blur_torch(x[0],2)[None],size=size,mode='bicubic',align_corners=False)


def design(rgb, basis, centers=None):
    r,g,b=rgb.unbind(1)
    if basis=='s13':
        terms=[r,g,b,r*r,g*g,b*b,r**3,g**3,b**3,r*g,r*b,g*b]
    elif basis in ('affine','q2','c3'):
        degree={'affine':1,'q2':2,'c3':3}[basis]
        terms=[]
        for d in range(1,degree+1):
            for i in range(d,-1,-1):
                for j in range(d-i,-1,-1):
                    k=d-i-j;terms.append(r**i*g**j*b**k)
    elif basis=='root2':
        # Signed homogeneous extension only if source values are negative.
        root=lambda t:t.sign()*t.abs().sqrt()
        terms=[r,g,b,root(r*g),root(r*b),root(g*b)]
    elif basis=='rbf':
        terms=[r,g,b]
        # 16 deterministic farthest-point centers selected from degraded RGB.
        for c in centers:
            terms.append(torch.exp(-((rgb-c[None,:,None,None])**2).sum(1)/(2*0.2**2)))
    else:raise ValueError(basis)
    return torch.stack(terms+[torch.ones_like(r)],1)


def color_centers(rgb,n=16):
    pts=rgb[0].flatten(1).T
    inds=torch.linspace(0,len(pts)-1,min(4096,len(pts)),device=pts.device).long()
    pts=pts[inds]; chosen=[];dist=((pts-pts.mean(0))**2).sum(1)
    for _ in range(n):
        p=pts[dist.argmax()];chosen.append(p)
        d=((pts-p)**2).sum(1)
        dist=d if len(chosen)==1 else torch.minimum(dist,d)
    return torch.stack(chosen)


@torch.no_grad()
def canonicalize(rgb,lq,srf,method,lam=1e-6):
    """Returns corrected RGB and diagnostics. Single scene, original training S0."""
    anchor=torch.einsum('kc,bchw->bkhw',srf,lq)
    size=lq.shape[-2:]
    if method=='raw':return rgb,{'terms':0}
    if method=='anchor':return F.interpolate(anchor,size=rgb.shape[-2:],mode='bicubic',align_corners=False).clamp(0,1),{'terms':0}
    if method=='bp':
        result=rgb+F.interpolate(anchor-degrade(rgb,size),size=rgb.shape[-2:],mode='bicubic',align_corners=False)
        return result.clamp(0,1),{'terms':0}
    # e.g. s13_forward_bp1
    basis,fit,bp=method.split('_')
    centers=color_centers(degrade(rgb,size)) if basis=='rbf' else None
    phi=design(rgb,basis,centers)
    a=(degrade(phi,size) if fit=='forward' else design(degrade(rgb,size),basis,centers)).flatten(2).transpose(1,2)[0].double()
    target=anchor.flatten(2).transpose(1,2)[0].double()
    scale=a.square().mean(0).sqrt().clamp_min(1e-8);scale[-1]=1
    a=a/scale
    reg=torch.eye(a.shape[1],device=a.device,dtype=a.dtype);reg[-1,-1]=0
    gram=a.T@a/len(a);rhs=a.T@target/len(a)
    coef=torch.linalg.solve(gram+lam*reg,rhs)
    # Avoid a full-resolution float64 feature cube.
    weights=(coef/scale[:,None]).float()
    result=torch.einsum('bkhw,kc->bchw',phi,weights)
    if bp=='bp1':result+=F.interpolate(anchor-degrade(result,size),size=rgb.shape[-2:],mode='bicubic',align_corners=False)
    elif bp!='bp0':raise ValueError(bp)
    diag={'terms':a.shape[1], 'condition_number':float(torch.linalg.cond(a)), 'clipped_fraction':float(((result<0)|(result>1)).float().mean())}
    return result.clamp(0,1),diag
