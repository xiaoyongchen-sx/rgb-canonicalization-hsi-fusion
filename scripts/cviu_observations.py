"""Declared observations; ICVL camera/illuminant curves use400--700nm."""
from functools import lru_cache
from pathlib import Path
import torch,numpy as np,colour
import scipy.io as sio
from scripts.composite_pipeline import build_profile,PROFILES
from scripts.convert_jiang_camera_srfs import load_database,safe_name
from scripts.rgb_filters_citable import apply_rgb_filter
ROOT=Path(__file__).resolve().parents[1]
@lru_cache(None)
def camera(name,dataset):
 if dataset=='Harvard':return torch.tensor(sio.loadmat(ROOT/'SRF/test'/f'{name}_harvard.mat')['P_20N'],dtype=torch.float32)
 db={safe_name(k):v for k,v in load_database(ROOT/'SRF/test/source/camspec_database.txt')}
 v=db[name][:,:31];v=v/v.sum(1,keepdims=True)
 return torch.tensor(v,dtype=torch.float32)
@lru_cache(None)
def profile(name,dataset):
 p=build_profile(name)
 if dataset=='ICVL':
  p.srf=camera(PROFILES[name]['srf'],dataset)
  sd=colour.SDS_ILLUMINANTS[PROFILES[name]['illuminant']].copy().align(colour.SpectralShape(400,700,10))
  v=torch.tensor(np.asarray(sd.values),dtype=torch.float32);p.illuminant=v/v.mean()
 return p

def observation(x,canonical,condition,dataset):
 if condition in PROFILES:return profile(condition,dataset)(x[0]).unsqueeze(0)
 if condition=='phase_one' or condition.startswith('camera:'):
  name='phase_one' if condition=='phase_one' else condition.split(':',1)[1]
  return torch.einsum('kc,bchw->bkhw',camera(name,dataset).to(x.device),x)
 return apply_rgb_filter(canonical[0],condition).unsqueeze(0)
