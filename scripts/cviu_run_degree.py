"""Scene-streaming evaluation with frozen reconstruction backbones."""
import argparse,csv,json,time,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from omegaconf import OmegaConf
from torchmetrics.image import StructuralSimilarityIndexMeasure,MultiScaleStructuralSimilarityIndexMeasure,ErrorRelativeGlobalDimensionlessSynthesis
from model.ResShift_model import ResShiftTrainer, Edge, safe_sam
from arch.BAFUnet import FixedSRFRGBCanonicalizer
from scripts.cviu_canonicalizer_degree import canonicalize,degrade
from scripts.rgb_filters_citable import apply_rgb_filter
from scripts.composite_pipeline import build_profile


def parse():
 p=argparse.ArgumentParser()
 p.add_argument('--dataset',default='Harvard',choices=['Harvard','ICVL'])
 p.add_argument('--split',default='train',choices=['train','test'])
 p.add_argument('--max-scenes',type=int,default=3)
 p.add_argument('--crop',type=int,default=256)
 p.add_argument('--conditions',nargs='+',default=['clean','gamma_encode','white_balance_fluorescent','phase_one','A_indoor','B_outdoor','C_stress'])
 p.add_argument('--methods',nargs='+',default=['raw','anchor','bp','s13_lr_bp0','s13_lr_bp1','s13_forward_bp0','s13_forward_bp1','legacy'])
 p.add_argument('--lambdas',nargs='+',type=float,default=[1e-6])
 p.add_argument('--name',required=True)
 p.add_argument('--seed',type=int,default=20260721)
 p.add_argument('--save-scenes',type=int,default=1)
 return p.parse_args()


@torch.no_grad()
def infer(tr,rgb_original,rgb_corrected,lq):
 up=F.interpolate(lq,scale_factor=8,mode='bicubic',align_corners=False)
 edge=Edge()(rgb_corrected) if tr.use_rgb_edge else None
 base=torch.cat([up,rgb_original],1)  # audited legacy integration, frozen for all methods
 xt=tr.EMRDIFF.prior_sample(base,torch.randn_like(base),edge_map=edge)
 for t in reversed(range(tr.num_timesteps)):
  tt=torch.tensor([t],device=base.device)
  xp,_=tr.Net(xt,rgb_corrected,up,tt,apply_nikon_rgb_restorer=False)
  xp=xp+base
  xt=tr.EMRDIFF.inverse_denoise(x_start=xp,x_t=xt,t=tt,noise=torch.randn_like(xp),edge_map=edge)
 return xp[:,:31].clamp(-1,1)


def metrics(pred,gt):
 mse=float((pred-gt).square().mean())
 norms=torch.linalg.vector_norm(gt,dim=1)*torch.linalg.vector_norm(pred,dim=1)
 valid=norms>1e-8
 sam=float(torch.acos(((pred*gt).sum(1)[valid]/norms[valid]).clamp(-1,1)).mean()) if valid.any() else float('nan')
 out=dict(psnr=-10*np.log10(max(mse,1e-30)),sam=sam,sam_coverage=float(valid.float().mean()))
 for name,cls,kw in [('ssim',StructuralSimilarityIndexMeasure,{'data_range':1.0}),('ms_ssim',MultiScaleStructuralSimilarityIndexMeasure,{'data_range':1.0}),('ergas',ErrorRelativeGlobalDimensionlessSynthesis,{'ratio':8})]:
  out[name]=float(cls(**kw).to(gt.device)(pred,gt))
 return out


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
 args=parse();out=ROOT/'output/CVIU'/args.name;out.mkdir(parents=True,exist_ok=True)
 ck=ROOT/('output/Harvard/h_change_dataset_split_and_dataprocess_basline/checkpoints/best.pth.tar' if args.dataset=='Harvard' else 'output/ICVL/baseline/checkpoints/best.pth.tar')
 paths=sorted((ROOT/'dataset'/args.dataset/args.split).glob('*.mat'))[:args.max_scenes]
 cfg=OmegaConf.load(ROOT/'config/harvard_baseline.yaml')
 cfg.data.dataset_name='CVIU/'+args.name
 cfg.train.pretrained=str(ck);cfg.train.resume=None
 cfg.test={'profile_checkpoint':None,'test_only':True,'skip_flops_profile':True,'use_amp':False,'save_mat':False}
 cfg.model.params.use_pseudo_msi=True;cfg.model.params.use_nikon_rgb_restorer=False
 manifest=dict(args=vars(args),checkpoint_sha256=sha(ck),scenes=[str(p.relative_to(ROOT)) for p in paths],sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),ROOT/'scripts/cviu_canonicalizer_degree.py',ROOT/'arch/BAFUnet.py',ROOT/'model/ResShift_model.py']})
 mf=out/'run.json'
 if mf.exists() and json.loads(mf.read_text())!=manifest:raise RuntimeError('Existing run has different manifest; choose a new --name')
 mf.write_text(json.dumps(manifest,indent=2))
 tr=ResShiftTrainer(cfg);tr.Net.eval();tr.Net.requires_grad_(False)
 # Refuse silently adapted or missing backbone weights.
 original=torch.load(ck,map_location='cpu',weights_only=False)['model']
 for k,v in tr.Net.state_dict().items():
  if k not in original or not torch.equal(v.cpu(),original[k]):raise RuntimeError('Backbone differs from checkpoint: '+k)
 del original
 dev=tr.device;srf=torch.tensor(sio.loadmat(ROOT/'SRF/P_N_V2.mat')['P_20N'],dtype=torch.float32,device=dev)
 legacy=FixedSRFRGBCanonicalizer(rgb_channels=3,hsi_channels=31,scale=8,mode='global_forward_polynomial',normalize_target_srf=True,allow_signed_target_srf=True,clamp_output=True).to(dev);legacy.set_target_srf(srf)
 rows=list(csv.DictReader((out/'per_scene.csv').open())) if (out/'per_scene.csv').exists() else []
 done={(r['scene'],r['condition'],r['method'],float(r['lambda'])) for r in rows}
 for idx,path in enumerate(paths):
  x=torch.tensor(sio.loadmat(path)['ref'],device=dev,dtype=torch.float32).permute(2,0,1)[None]
  if args.crop:x=x[...,:args.crop,:args.crop]
  x=(x-x.min())/(x.max()-x.min()+1e-8)
  h,w=x.shape[-2:];assert h%8==0 and w%8==0
  lq=degrade(x,(h//8,w//8));canonical=torch.einsum('kc,bchw->bkhw',srf,x)
  for cond in args.conditions:
   if cond in ['A_indoor','B_outdoor','C_stress']:
    if args.dataset!='Harvard':raise RuntimeError('ICVL composite must use wavelength-aligned profile; not yet implemented')
    rgb=build_profile(cond)(x[0]).unsqueeze(0)
   elif cond=='phase_one':
    if args.dataset!='Harvard':raise RuntimeError('ICVL source SRF must be wavelength aligned')
    sr=torch.tensor(sio.loadmat(ROOT/'SRF/test/phase_one_harvard.mat')['P_20N'],device=dev,dtype=torch.float32)
    rgb=torch.einsum('kc,bchw->bkhw',sr,x)
   else:rgb=apply_rgb_filter(canonical[0],cond).unsqueeze(0)
   for method in args.methods:
    for lam in (args.lambdas if method not in ['raw','anchor','bp','legacy'] else [0.]):
     key=(path.stem,cond,method,lam)
     if key in done:continue
     torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
     if method=='legacy':corrected=legacy(rgb,lq)[0];diag={'terms':13}
     else:corrected,diag=canonicalize(rgb,lq,srf,method,lam)
     torch.cuda.synchronize();canon_ms=(time.perf_counter()-t0)*1000
     torch.manual_seed(args.seed+idx);torch.cuda.manual_seed_all(args.seed+idx)
     t0=time.perf_counter()
     with torch.no_grad():pred=infer(tr,rgb,corrected,lq)
     torch.cuda.synchronize();infer_s=time.perf_counter()-t0
     row=dict(scene=path.stem,condition=cond,method=method,**{'lambda':lam},seed=args.seed+idx,height=h,width=w,**metrics(pred,x),rgb_mae=float((corrected-canonical).abs().mean()),canon_ms=canon_ms,infer_s=infer_s,peak_mb=torch.cuda.max_memory_allocated()/1048576,terms=diag['terms'])
     exists=(out/'per_scene.csv').exists()
     with (out/'per_scene.csv').open('a') as f:
      cw=csv.DictWriter(f,fieldnames=list(row));
      if not exists:cw.writeheader()
      cw.writerow(row)
     done.add(key);rows.append(row)
     if idx<args.save_scenes and cond in ['clean','gamma_encode','C_stress']:
      np.savez_compressed(out/f'{path.stem}_{cond}_{method}_{lam}.npz',prediction=pred[0].cpu().numpy(),rgb=corrected[0].cpu().numpy())
     print(json.dumps(row),flush=True)
 tr.writer.close()
 print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
