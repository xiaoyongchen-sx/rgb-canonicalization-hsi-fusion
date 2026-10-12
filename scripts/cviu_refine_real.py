"""Fixed geometry-only refinement and validity; no reconstruction outcomes used."""
from pathlib import Path
import json,hashlib
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'dataset/CVIU_real/prepared_v1';OUT=ROOT/'dataset/CVIU_real/prepared_v2';OUT.mkdir(parents=True,exist_ok=True)
cv2.setNumThreads(2);cv2.setRNGSeed(20260916)
def gray(rgb):
 return cv2.createCLAHE(2,(8,8)).apply(cv2.cvtColor(np.uint8(np.clip(rgb,0,1)*255),cv2.COLOR_RGB2GRAY))
def remap(x,flow):
 yy,xx=np.indices(flow.shape[:2],dtype=np.float32);return cv2.remap(x,xx+flow[...,0],yy+flow[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
def process(scene):
 z=np.load(SRC/f'{scene:03d}.npz');x=z['hsi'];rgb=z['rgb'];h,w=x.shape[:2]
 show=x[:,:,[23,13,3]];lo=np.percentile(show,.5,(0,1));hi=np.percentile(show,99.5,(0,1));show=np.clip((show-lo)/(hi-lo+1e-8),0,1)**(1/2.2)
 a=gray(show);b=gray(rgb)
 dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
 f=dis.calc(a,b,None);g=dis.calc(b,a,None)
 warped=remap(rgb,f);bw=remap(b.astype(np.float32),f);fb=np.linalg.norm(f+remap(g,f),axis=2)
 def avg(v):return cv2.boxFilter(v,-1,(15,15),normalize=True,borderType=cv2.BORDER_REFLECT)
 af=a.astype(np.float32);ma=avg(af);mb=avg(bw);va=np.maximum(avg(af*af)-ma*ma,0);vb=np.maximum(avg(bw*bw)-mb*mb,0)
 ncc=(avg(af*bw)-ma*mb)/np.sqrt(va*vb+1e-6)
 inside=remap(np.ones((h,w),np.float32),f)>.999
 mask=(fb<1)&(np.linalg.norm(f,axis=2)<32)&(ncc>.6)&(va>9)&(vb>9)&inside
 mask=cv2.erode(mask.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
 mask[:12]=False;mask[-12:]=False;mask[:,:12]=False;mask[:,-12:]=False
 coverage=float(mask.mean());accepted=coverage>=.25
 result=dict(scene=scene,accepted=accepted,coverage=coverage,median_fb=float(np.median(fb[mask])) if mask.any() else None,median_ncc=float(np.median(ncc[mask])) if mask.any() else None,median_displacement=float(np.median(np.linalg.norm(f,axis=2)[mask])) if mask.any() else None)
 if accepted:np.savez_compressed(OUT/f'{scene:03d}.npz',hsi=x,rgb=warped,valid=mask,wavelength=z['wavelength'])
 return result
if __name__=='__main__':
 prior=json.loads((SRC/'preparation.json').read_text());results=[]
 for r in prior['results']:
  if r.get('accepted'):result=process(r['scene'])
  else:result=dict(scene=r['scene'],accepted=False,reason='initial homography screen failed')
  results.append(result);print(json.dumps(result),flush=True)
  (OUT/'refinement.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol='DIS-medium bidirectional; FB<1px, displacement<32px, localNCC15>.6, variance>9; erode7; border12; coverage>=.25; all fixed before reconstruction',results=results),indent=2))
 print('COMPLETE',flush=True)
