#!/usr/bin/env python3
"""
Gate 3: blind temporal source separation learned by a nonlinear neural circuit.

Two independent Gaussian AR sources have different temporal autocorrelation. They are
mixed by a fixed 2x2 matrix. No source labels are used in training.

A small tanh encoder + linear decoder is trained on:
  1) reconstruction of the observed mixture;
  2) zero-lag decorrelation of the two latent outputs;
  3) off-diagonal lag-covariance penalties at several lags (SOBI/AMUSE-like signal).

Question: does the neural circuit learn a fuzzy implementation whose collective
input->latent map can be decoded back into a compact demixing matrix?
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import itertools
import numpy as np
import torch
from torch import nn

A=torch.tensor([[1.0,0.8],[0.4,1.2]],dtype=torch.float32)
A_INV=torch.linalg.inv(A)

class BlindDemixer(nn.Module):
    def __init__(self,hidden=16):
        super().__init__(); self.fc1=nn.Linear(2,hidden); self.fc2=nn.Linear(hidden,2); self.dec=nn.Linear(2,2)
    def encode(self,x): return self.fc2(torch.tanh(self.fc1(x)))
    def forward(self,x):
        y=self.encode(x); return y,self.dec(y)

def make_ar(T,seed,phis=(0.92,-0.35),scale=0.35):
    g=torch.Generator().manual_seed(seed)
    s=torch.zeros(T,2)
    eps=torch.randn(T,2,generator=g)
    sig=torch.tensor([(1-p*p)**0.5 for p in phis])
    for t in range(1,T): s[t]=torch.tensor(phis)*s[t-1]+sig*eps[t]
    s=s*scale
    x=s@A.T
    return x,s

def standardize(y): return (y-y.mean(0,keepdim=True))/(y.std(0,keepdim=True)+1e-6)
def offdiag2(M): return M[0,1]**2+M[1,0]**2

def temporal_loss(y,lags=(1,2,5,10)):
    # Keep both latent channels alive: unlike per-channel standardization, an explicit
    # covariance-to-identity penalty makes collapse expensive.
    z=y-y.mean(0,keepdim=True); n=len(z)
    C0=z.T@z/n
    diag=torch.diag(C0)
    loss=torch.sum((diag-1.0)**2)+offdiag2(C0)
    for lag in lags:
        C=z[:-lag].T@z[lag:]/(n-lag)
        loss=loss+offdiag2(C)
    return loss/ (1+len(lags))

def fit_affine(x,y):
    X=torch.cat([x,torch.ones(len(x),1)],dim=1); th=torch.linalg.lstsq(X,y).solution; return th[:2].T,th[2]

def nmse(a,b): return float(torch.mean((a-b)**2)/(torch.var(b)+1e-12))

def best_source_corr(y,s):
    ys=standardize(y); ss=standardize(s); C=(ys.T@ss)/len(y); C=C.detach().cpu().numpy()
    best=(-1,None)
    for perm in itertools.permutations(range(2)):
        score=np.mean([abs(C[i,perm[i]]) for i in range(2)])
        if score>best[0]: best=(score,perm)
    return float(best[0]),list(best[1]),C.tolist()

def row_direction_error(B):
    # Projective row comparison to exact inverse: sign/scale/permutation are irrelevant.
    b=B.detach().cpu().numpy(); w=A_INV.detach().cpu().numpy()
    b=b/(np.linalg.norm(b,axis=1,keepdims=True)+1e-12); w=w/(np.linalg.norm(w,axis=1,keepdims=True)+1e-12)
    C=np.abs(b@w.T); best=None
    for perm in itertools.permutations(range(2)):
        cs=[C[i,perm[i]] for i in range(2)]; err=np.mean([np.arccos(np.clip(c,-1,1)) for c in cs])*180/np.pi
        if best is None or err<best[0]: best=(err,perm,cs)
    return float(best[0]),list(best[1]),[float(x) for x in best[2]]

def analytic_amuse(x,s,lag=5):
    X=x-x.mean(0,keepdim=True); C0=(X.T@X)/len(X)
    e,V=torch.linalg.eigh(C0); Q=V@torch.diag(torch.rsqrt(torch.clamp(e,min=1e-8)))@V.T
    z=X@Q
    C=(z[:-lag].T@z[lag:])/(len(z)-lag); C=0.5*(C+C.T)
    _,R=torch.linalg.eigh(C)
    y=z@R
    return best_source_corr(y,s)[0]

def train(seed=0,steps=2000,temporal_weight=2.0):
    torch.manual_seed(seed); np.random.seed(seed)
    x,s=make_ar(24000,seed+1)
    model=BlindDemixer(16); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-5)
    window=768
    checkpoints=[]; cp={0,100,200,500,1000,2000,4000}
    for step in range(steps+1):
        if step in cp:
            with torch.no_grad(): y,_=model(x[:6000]); corr,perm,C=best_source_corr(y,s[:6000]); B,b=fit_affine(x[:6000],y); derr,_,_=row_direction_error(B); fid=nmse(x[:6000]@B.T+b,y)
            checkpoints.append({'step':step,'source_corr':corr,'decoded_row_direction_error_deg':derr,'affine_surrogate_vs_neural_nmse':fid})
        if step==steps: break
        start=int(torch.randint(0,len(x)-window,(1,))); xb=x[start:start+window]
        y,xhat=model(xb)
        recon=torch.mean((xhat-xb)**2)/(torch.var(xb)+1e-8)
        tl=temporal_loss(y)
        loss=recon+temporal_weight*tl
        opt.zero_grad(); loss.backward(); opt.step()
    return model,x,s,checkpoints

def evaluate(model,x,s,seed):
    with torch.no_grad(): y,xhat=model(x)
    corr,perm,C=best_source_corr(y,s); B,b=fit_affine(x,y); derr,dperm,cos=row_direction_error(B); surrogate=x@B.T+b
    x2,s2=make_ar(8000,seed+99,scale=0.70)
    with torch.no_grad(): y2,_=model(x2)
    math2=x2@B.T+b
    ncorr=best_source_corr(y2,s2)[0]; mcorr=best_source_corr(math2,s2)[0]
    return {'source_corr':corr,'corr_matrix':C,'reconstruction_nmse':nmse(xhat,x),'decoded_matrix':B.tolist(),'decoded_bias':b.tolist(),'decoded_row_direction_error_deg':derr,'decoded_row_assignment':dperm,'decoded_row_cosines':cos,'affine_surrogate_vs_neural_nmse':nmse(surrogate,y),'ood2x_neural_source_corr':ncorr,'ood2x_decoded_math_source_corr':mcorr}

def run(seed=0,steps=2000):
    model,x,s,cps=train(seed,steps,2.0); main=evaluate(model,x[:8000],s[:8000],seed)
    attacker,xa,sa,_=train(seed+1000,steps,0.0); rec_only=evaluate(attacker,xa[:8000],sa[:8000],seed+1000)
    amuse=analytic_amuse(x[:8000],s[:8000],lag=5)
    return {'seed':seed,'training':'no source labels; reconstruction + lagged decorrelation','main':main,'reconstruction_only_attacker':rec_only,'analytic_amuse_source_corr':amuse,'checkpoints':cps}

def main_cli():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=2000);p.add_argument('--out',default='results/gate3_seed0.json');a=p.parse_args();r=run(a.seed,a.steps);Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(r,indent=2));print(json.dumps({'main':r['main'],'recon_only_corr':r['reconstruction_only_attacker']['source_corr'],'amuse_corr':r['analytic_amuse_source_corr']},indent=2))
if __name__=='__main__':main_cli()
