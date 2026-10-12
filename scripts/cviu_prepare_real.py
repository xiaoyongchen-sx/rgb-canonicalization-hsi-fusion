"""BUSIFusion author-compatible radiometry and offline geometric preparation.
No model, corrector, reconstruction scores, or outcome-based sample selection.
"""
import json,re,hashlib
from pathlib import Path
import numpy as np
import cv2,h5py
from scipy.interpolate import interp1d
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'dataset/CVIU_real/inspection';OUT=ROOT/'dataset/CVIU_real/prepared_v1';OUT.mkdir(parents=True,exist_ok=True)
METADATA=OUT;cv2.setNumThreads(2);cv2.setRNGSeed(20260910)

def load_hsi(scene):
    base=SRC/f'rgbrybspec/{scene}';hdr=base.with_suffix('.hdr').read_text()
    def field(key):return re.search(r'^'+re.escape(key)+r'\s*=\s*(.*)$',hdr,re.M|re.I).group(1).strip()
    assert (int(field('samples')),int(field('lines')),int(field('bands')))==(512,512,204)
    assert field('interleave').lower()=='bil' and field('byte order')=='0' and field('data type')=='12'
    wave=np.array([float(v) for v in re.search(r'wavelength\s*=\s*\{(.*?)\}',hdr,re.S).group(1).split(',') if v.strip()])
    with h5py.File(SRC/'wavelength.mat') as f:author_wave=f['wavelength'][()].ravel()
    assert np.max(np.abs(wave-author_wave))<.011
    # Equivalent to author's rdbil16 + MATLAB reshape(order F)/permute.
    raw=np.fromfile(base.with_suffix('.raw'),dtype='<u2').reshape(512,204,512)
    cube=np.maximum(raw.transpose(2,0,1).astype(np.float32)-256,0)
    target=np.arange(420,721,10);assert wave[0]<=target[0] and target[-1]<=wave[-1]
    cube=interp1d(wave,cube,axis=2,kind='linear',bounds_error=True)(target).astype(np.float32)
    tint=float(field('tint'));assert tint>0
    cube=(cube/tint)[:,::-1,:].copy()
    return cube,dict(tint=tint,wavelength_min=float(wave[0]),wavelength_max=float(wave[-1]),black_level=256,orientation='author MATLAB permute and horizontal flip',units='black-level corrected digital counts / tint; not asserted reflectance')

def display(cube):
    rgb=cube[:,:,[23,13,3]] # 650,550,450nm only for geometric features
    lo=np.percentile(rgb,.5,axis=(0,1),keepdims=True);hi=np.percentile(rgb,99.5,axis=(0,1),keepdims=True)
    return np.uint8(255*np.clip((rgb-lo)/(hi-lo+1e-8),0,1)**(1/2.2))

def gray(rgb):
    return cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))

def prepare(scene):
    x,meta=load_hsi(scene);target=display(x)
    phone=cv2.cvtColor(cv2.imread(str(SRC/f'rgbrybspec/{scene}rgb.jpg')),cv2.COLOR_BGR2RGB)
    scale=min(1,1536/max(phone.shape[:2]));phone=cv2.resize(phone,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    sift=cv2.SIFT_create(nfeatures=6000)
    k1,d1=sift.detectAndCompute(gray(phone),None);k2,d2=sift.detectAndCompute(gray(target),None)
    result=dict(scene=scene,**meta,keypoints_phone=len(k1),keypoints_hsi=len(k2),status='rejected_geometry')
    if d1 is None or d2 is None:return result
    matches=cv2.BFMatcher().knnMatch(d1,d2,k=2)
    good=[a for pair in matches if len(pair)==2 for a,b in [pair] if a.distance<.75*b.distance]
    result['matches']=len(good)
    if len(good)<12:return result
    a=np.float32([k1[m.queryIdx].pt for m in good]);b=np.float32([k2[m.trainIdx].pt for m in good])
    H,mask=cv2.findHomography(a,b,cv2.RANSAC,3,maxIters=10000,confidence=.999)
    if H is None:return result
    ins=mask.ravel().astype(bool);projected=cv2.perspectiveTransform(a[:,None,:],H)[:,0,:]
    error=np.linalg.norm(projected-b,axis=1);hull=cv2.contourArea(cv2.convexHull(b[ins]))/(512*512)
    result.update(inliers=int(ins.sum()),inlier_fraction=float(ins.mean()),median_inlier_error_px=float(np.median(error[ins])),inlier_hull_fraction=float(hull),homography_phone_resized_to_hsi=H.tolist(),phone_resize_scale=scale)
    warped=cv2.warpPerspective(phone.astype(np.float32)/255,H,(512,512),flags=cv2.INTER_LINEAR)
    valid=cv2.warpPerspective(np.ones(phone.shape[:2],np.uint8),H,(512,512),flags=cv2.INTER_NEAREST)
    valid=cv2.erode(valid,np.ones((9,9),np.uint8))
    # Fixed central 384 native pixels, chosen without reconstruction metrics.
    box=(64,64,448,448);crop=valid[64:448,64:448]
    result['valid_crop_fraction']=float(crop.mean());result['crop_xyxy']=box
    result['accepted']=bool(ins.sum()>=30 and ins.mean()>=.25 and hull>=.2 and np.median(error[ins])<=2 and crop.min()==1)
    result['status']='geometry_pass' if result['accepted'] else 'rejected_geometry'
    if result['accepted']:
        # Identical per-scene HSI scale for both future methods; RGB fixed /255.
        norm_min=float(x.min());norm_max=float(x.max());x=(x-norm_min)/(norm_max-norm_min+1e-8)
        np.savez_compressed(OUT/f'{scene:03d}.npz',hsi=x[64:448,64:448],rgb=warped[64:448,64:448],valid=crop,wavelength=np.arange(420,721,10))
        result.update(hsi_min=norm_min,hsi_max=norm_max)
    return result

if __name__=='__main__':
    ids=json.loads((ROOT/'config/real_scenes.json').read_text())['ids'];results=[]
    for scene in ids:
        result=prepare(scene);results.append(result);print(json.dumps(result),flush=True)
        (METADATA/'preparation.json').write_text(json.dumps(dict(protocol='v1: author radiometry; fixed SIFT homography; center384; no reconstruction evaluation',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=results),indent=2))
    print('COMPLETE',len(results),sum(r.get('accepted',False) for r in results),flush=True)
