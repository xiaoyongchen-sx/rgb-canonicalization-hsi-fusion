"""Frozen Harvard model on independently captured BUSIFusion RGB+HSI.
Geometry/ROI fixed without reconstruction outcomes; LR-HSI is synthetic.
"""
import sys,json,csv,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,cv2,scipy.io as sio
from skimage.metrics import structural_similarity
from omegaconf import OmegaConf
from scripts.cviu_run import infer,metrics,sha
from scripts.cviu_canonicalizer_degree import canonicalize,degrade
from model.ResShift_model import ResShiftTrainer

def roi_metrics(pred,gt,mask):
 p=pred[0].permute(1,2,0).cpu().numpy();g=gt[0].permute(1,2,0).cpu().numpy();a=p[mask];b=g[mask]
 mse=np.mean((a-b)**2);norm=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1);valid=norm>1e-8
 sam=np.arccos(np.clip(np.sum(a*b,axis=1)[valid]/norm[valid],-1,1)).mean()
 ergas=100/8*np.sqrt(np.mean(np.mean((a-b)**2,axis=0)/(np.mean(b,axis=0)**2+1e-12)))
 _,smap=structural_similarity(g,p,channel_axis=-1,data_range=1,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,full=True)
 inner=cv2.erode(mask.astype(np.uint8),np.ones((11,11),np.uint8)).astype(bool);assert inner.any()
 return dict(roi_psnr=float(-10*np.log10(max(float(mse),1e-30))),roi_sam=float(sam),roi_sam_coverage=float(valid.mean()),roi_ssim=float(smap.mean(2)[inner].mean()),roi_ergas=float(ergas),roi_coverage=float(mask.mean()),ssim_roi_coverage=float(inner.mean()))

@torch.no_grad()
def main():
 config=json.loads((ROOT/'config/canonicalization.json').read_text());assert config['method']=='c5_forward_bp1' and config['lambda']==1e-6
 out=ROOT/'output/CVIU/e05_real_v1';out.mkdir(parents=True,exist_ok=True)
 prep=json.loads((ROOT/'dataset/CVIU_real/prepared_v2/refinement.json').read_text());ids=[r['scene'] for r in prep['results'] if r['accepted']]
 paths=[ROOT/f'dataset/CVIU_real/prepared_v2/{s:03d}.npz' for s in ids];assert len(ids)==16
 ck=ROOT/'output/Harvard/h_change_dataset_split_and_dataprocess_basline/checkpoints/best.pth.tar'
 manifest=dict(checkpoint_sha256=sha(ck),method=config,scene_ids=ids,input_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),ROOT/'scripts/cviu_canonicalizer_degree.py',ROOT/'scripts/cviu_run.py',ROOT/'scripts/cviu_refine_real.py']},seed=20260721,protocol='real independent RGB and HSI; spectral resampling and offline GT-assisted geometry; synthetic LR-HSI; fixed matched ROI plus full crop metrics; no HR-HSI in map fit')
 mf=out/'run.json'
 if mf.exists():assert json.loads(mf.read_text())==manifest
 mf.write_text(json.dumps(manifest,indent=2))
 cfg=OmegaConf.load(ROOT/'config/harvard_baseline.yaml');cfg.data.dataset_name='CVIU/e05_real_v1';cfg.train.pretrained=str(ck);cfg.train.resume=None
 cfg.test={'profile_checkpoint':None,'test_only':True,'skip_flops_profile':True,'use_amp':False,'save_mat':False};cfg.model.params.use_pseudo_msi=True;cfg.model.params.use_nikon_rgb_restorer=False
 tr=ResShiftTrainer(cfg);tr.Net.eval();tr.Net.requires_grad_(False)
 weights=torch.load(ck,map_location='cpu',weights_only=False)['model']
 assert all(k in weights and torch.equal(v.cpu(),weights[k]) for k,v in tr.Net.state_dict().items());del weights
 srf=torch.tensor(sio.loadmat(ROOT/'SRF/P_N_V2.mat')['P_20N'],dtype=torch.float32,device=tr.device)
 csvpath=out/'per_scene.csv';done={(r['scene'],r['method']) for r in csv.DictReader(csvpath.open())} if csvpath.exists() else set()
 for scene,path in zip(ids,paths):
  z=np.load(path);x=torch.tensor(z['hsi'],device=tr.device).permute(2,0,1)[None];rgb=torch.tensor(z['rgb'],device=tr.device).permute(2,0,1)[None];mask=z['valid'].astype(bool)
  lq=degrade(x,(48,48))
  for method in ['raw',config['method']]:
   if (str(scene),method) in done:continue
   torch.cuda.synchronize();t=time.perf_counter();corrected,diag=canonicalize(rgb,lq,srf,method,config['lambda']);torch.cuda.synchronize();ms=(time.perf_counter()-t)*1000
   torch.manual_seed(20260721+scene);torch.cuda.manual_seed_all(20260721+scene);t=time.perf_counter();pred=infer(tr,rgb,corrected,lq);torch.cuda.synchronize();sec=time.perf_counter()-t
   full=metrics(pred,x);row=dict(scene=scene,condition='real',method=method,**{'lambda':0 if method=='raw' else config['lambda']},seed=20260721+scene,**full,**roi_metrics(pred,x,mask),canon_ms=ms,infer_s=sec)
   np.savez_compressed(out/f'{scene:03d}_{method}.npz',prediction=pred[0].cpu().numpy(),rgb=corrected[0].cpu().numpy())
   exists=csvpath.exists()
   with csvpath.open('a') as f:
    w=csv.DictWriter(f,fieldnames=list(row));
    if not exists:w.writeheader()
    w.writerow(row)
   print(json.dumps(row),flush=True)
 tr.writer.close();print('COMPLETE',flush=True)
if __name__=='__main__':main()
