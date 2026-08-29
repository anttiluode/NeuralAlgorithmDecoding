#!/usr/bin/env python3
"""
Gate 2: the decoder is no longer told "fit a matrix".

It gets only x -> neural-output queries and a small candidate language of polynomial
programs (degrees 0..3). It must select the simplest family that reproduces the
black box below a fixed fidelity tolerance.

Positive controls:
  L: a neural network trained to approximate a linear demixing law -> choose degree 1.
  Q: a neural network trained to approximate a genuinely quadratic 2-D law -> choose degree 2.

Ground-truth equations are used only after selection for scoring, never by the decoder.
This gate tests automatic model-class selection, not open-ended symbolic discovery.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from torch import nn

A=torch.tensor([[1.0,0.8],[0.4,1.2]],dtype=torch.float32); A_INV=torch.linalg.inv(A)

class Net(nn.Module):
    def __init__(self,hidden=32):
        super().__init__(); self.fc1=nn.Linear(2,hidden); self.fc2=nn.Linear(hidden,2)
    def forward(self,x): return self.fc2(torch.tanh(self.fc1(x)))

def nmse(a,b): return float(torch.mean((a-b)**2)/(torch.var(b)+1e-12))

def source_x(n,seed,scale=0.35):
    g=torch.Generator().manual_seed(seed); u=torch.rand(n,generator=g)-0.5
    s1=-torch.sign(u)*torch.log1p(-2*torch.abs(u)+1e-7)/math.sqrt(2.0)
    s2=(torch.rand(n,generator=g)*2-1)*math.sqrt(3.0)
    s=torch.stack([s1,s2],dim=1)*scale; return s@A.T

def target_linear(x): return x@A_INV.T

def target_quadratic(x):
    x1,x2=x[:,0],x[:,1]
    y1=1.10*x1-0.70*x2+0.75*x1*x2
    y2=-0.35*x1+0.90*x2+0.55*x1*x1-0.40*x2*x2
    return torch.stack([y1,y2],dim=1)

def exponents_2d(degree):
    exps=[]
    for total in range(degree+1):
        for a in range(total+1): exps.append((a,total-a))
    return exps

def features(x,degree):
    cols=[]
    for a,b in exponents_2d(degree): cols.append((x[:,0]**a)*(x[:,1]**b))
    return torch.stack(cols,dim=1)

def fit_poly(x,y,degree):
    F=features(x,degree); coef=torch.linalg.lstsq(F,y).solution; return coef

def predict_poly(x,coef,degree): return features(x,degree)@coef

def train_organism(seed,target_fn,steps=5000):
    torch.manual_seed(seed); np.random.seed(seed); model=Net(32); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-5)
    scale = 0.35 if target_fn is target_linear else 0.20
    x=source_x(12000,seed+10,scale); y=target_fn(x)
    for _ in range(steps):
        idx=torch.randint(0,len(x),(256,)); pred=model(x[idx]); loss=torch.mean((pred-y[idx])**2)
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def decode(model,seed,tol=0.01):
    # Decoder is not told the family, but probes the same training-domain scale.
    # Quadratic controls use seed >=1000 and the smaller training-domain scale.
    scale = 0.35 if seed < 1000 else 0.20
    xfit=source_x(4096,seed+100,scale); xval=source_x(4096,seed+101,scale)
    with torch.no_grad(): yfit=model(xfit); yval=model(xval)
    candidates=[]
    for d in range(4):
        coef=fit_poly(xfit,yfit,d); pv=predict_poly(xval,coef,d)
        f=nmse(pv,yval); terms=len(exponents_2d(d)); params=terms*2
        candidates.append({'degree':d,'terms_per_output':terms,'parameter_count':params,'network_fidelity_nmse':f,'coefficients':coef.tolist()})
    valid=[c for c in candidates if c['network_fidelity_nmse']<=tol]
    chosen=min(valid,key=lambda c:c['parameter_count']) if valid else min(candidates,key=lambda c:c['network_fidelity_nmse'])
    return chosen,candidates

def score(seed,kind,steps=5000,tol=0.01):
    target=target_linear if kind=='linear' else target_quadratic
    truth_degree=1 if kind=='linear' else 2
    model=train_organism(seed,target,steps)
    chosen,cands=decode(model,seed,tol)
    scale = 0.35 if kind=='linear' else 0.20
    xid=source_x(4096,seed+200,scale); xood=source_x(4096,seed+201,2*scale)
    with torch.no_grad(): nid=model(xid); nood=model(xood); tid=target(xid); tood=target(xood)
    coef=torch.tensor(chosen['coefficients']); did=predict_poly(xid,coef,chosen['degree']); dood=predict_poly(xood,coef,chosen['degree'])
    return {
      'kind':kind,'truth_degree':truth_degree,'chosen_degree':chosen['degree'],'correct_family':chosen['degree']==truth_degree,
      'network_id_vs_truth_nmse':nmse(nid,tid),'network_ood2x_vs_truth_nmse':nmse(nood,tood),
      'decoded_id_vs_truth_nmse':nmse(did,tid),'decoded_ood2x_vs_truth_nmse':nmse(dood,tood),
      'chosen':chosen,'candidates':cands
    }

def run(seed=0,steps=5000,tol=0.01):
    return {'seed':seed,'fidelity_tolerance':tol,'linear':score(seed,'linear',steps,tol),'quadratic':score(seed+1000,'quadratic',steps,tol)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=0); p.add_argument('--steps',type=int,default=5000); p.add_argument('--tol',type=float,default=0.01); p.add_argument('--out',default='results/gate2_seed0.json'); a=p.parse_args()
    r=run(a.seed,a.steps,a.tol); Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(r,indent=2))
    print(json.dumps({k:{'chosen_degree':r[k]['chosen_degree'],'truth_degree':r[k]['truth_degree'],'network_ood':r[k]['network_ood2x_vs_truth_nmse'],'decoded_ood':r[k]['decoded_ood2x_vs_truth_nmse'],'candidate_fidelity':[c['network_fidelity_nmse'] for c in r[k]['candidates']]} for k in ['linear','quadratic']},indent=2))
if __name__=='__main__':main()
