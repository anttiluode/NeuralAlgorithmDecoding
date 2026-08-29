#!/usr/bin/env python3
"""
Gate 1: film the birth of a simple mathematical operator inside a fuzzy neural circuit.

Same calibration organism as Gate 0, but the object is now DEVELOPMENT rather than the final model.
At checkpoints we ask:
  - when does the network solve the task?
  - when does a compact affine operator become visible in its behaviour?
  - when does the local Jacobian become nearly input-invariant?
  - how distributed is the implementation across hidden units?
  - when does the decoded math extrapolate better than the neural approximation?

No claim of novelty: this is a calibration for later unknown algorithms.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
from torch import nn

A = torch.tensor([[1.0, 0.8], [0.4, 1.2]], dtype=torch.float32)
A_INV = torch.linalg.inv(A)

class Demixer(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, 2)
    def forward(self, x, return_hidden=False):
        h = torch.tanh(self.fc1(x)); y = self.fc2(h)
        return (y, h) if return_hidden else y

def make_data(n, seed, scale=0.35):
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(n, generator=g) - 0.5
    s1 = -torch.sign(u) * torch.log1p(-2 * torch.abs(u) + 1e-7) / math.sqrt(2.0)
    s2 = (torch.rand(n, generator=g) * 2 - 1) * math.sqrt(3.0)
    s = torch.stack([s1, s2], dim=1) * scale
    return s @ A.T, s

def nmse(a,b): return float(torch.mean((a-b)**2)/(torch.var(b)+1e-12))
def rel_fro(a,b): return float(torch.linalg.norm(a-b)/(torch.linalg.norm(b)+1e-12))

def fit_affine_to_network(model, x):
    with torch.no_grad(): y = model(x)
    X = torch.cat([x, torch.ones(len(x),1)], dim=1)
    theta = torch.linalg.lstsq(X,y).solution
    return theta[:2].T, theta[2]

def jacobians(model, x):
    with torch.no_grad():
        z = model.fc1(x)
        d = 1 - torch.tanh(z)**2
        J = torch.einsum('oh,nh,hi->noi', model.fc2.weight, d, model.fc1.weight)
    return J

def unit_mean_jacobian_contributions(model,x):
    with torch.no_grad():
        z = model.fc1(x); dbar=(1-torch.tanh(z)**2).mean(0)
        # C[h,o,i] = W2[o,h] * mean tanh'(z_h) * W1[h,i]
        C = torch.einsum('oh,h,hi->hoi', model.fc2.weight, dbar, model.fc1.weight)
    return C

def contribution_stats(C):
    norms = torch.linalg.norm(C.reshape(C.shape[0],-1),dim=1)
    total = float(norms.sum()) + 1e-12
    p = norms / total
    participation = float(1.0/(torch.sum(p*p)+1e-12))
    summed = C.sum(0)
    cancellation = float(norms.sum()/(torch.linalg.norm(summed)+1e-12))
    # Greedily pick the contribution that best reduces residual to the full mean Jacobian.
    remaining = list(range(C.shape[0])); chosen=[]; partial=torch.zeros_like(summed); curve=[]
    target_norm = torch.linalg.norm(summed)+1e-12
    while remaining:
        best=None
        for h in remaining:
            err=float(torch.linalg.norm((partial+C[h])-summed)/target_norm)
            if best is None or err < best[0]: best=(err,h)
        err,h=best; chosen.append(h); remaining.remove(h); partial=partial+C[h]
        curve.append([len(chosen),err])
    k10 = next((k for k,e in curve if e<=0.10), len(curve))
    k05 = next((k for k,e in curve if e<=0.05), len(curve))
    return {
        'unit_contribution_norms': norms.tolist(),
        'jacobian_unit_participation_ratio': participation,
        'jacobian_cancellation_ratio': cancellation,
        'greedy_units_for_10pct_jacobian': int(k10),
        'greedy_units_for_5pct_jacobian': int(k05),
        'greedy_jacobian_curve': curve,
    }

def checkpoint(model, x_id, s_id, x_ood, s_ood, step):
    with torch.no_grad():
        y_id=model(x_id); y_ood=model(x_ood)
    B,b=fit_affine_to_network(model,x_id)
    math_id=x_id@B.T+b; math_ood=x_ood@B.T+b
    J=jacobians(model,x_id)
    Jmean=J.mean(0)
    Jdev=float(torch.sqrt(torch.mean((J-Jmean)**2))/(torch.sqrt(torch.mean(Jmean**2))+1e-12))
    C=unit_mean_jacobian_contributions(model,x_id)
    rec={
        'step':int(step),
        'network_id_nmse':nmse(y_id,s_id),
        'network_ood2x_nmse':nmse(y_ood,s_ood),
        'decoded_affine_id_vs_network_nmse':nmse(math_id,y_id),
        'decoded_affine_ood2x_nmse':nmse(math_ood,s_ood),
        'decoded_matrix_rel_error':rel_fro(B,A_INV),
        'jacobian_mean_rel_error':rel_fro(Jmean,A_INV),
        'jacobian_input_variation':Jdev,
        'decoded_matrix':B.tolist(),
    }
    rec.update(contribution_stats(C))
    return rec

def first_step(rows,key,pred):
    for r in rows:
        if pred(r[key]): return r['step']
    return None

def run(seed=0,steps=4000):
    torch.manual_seed(seed); np.random.seed(seed)
    model=Demixer(16)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-4)
    xtr,strue=make_data(8192,seed+1,0.35)
    xid,sid=make_data(2048,seed+2,0.35)
    xood,sood=make_data(2048,seed+3,0.70)
    checkpoints=[0,5,10,20,40,60,80,100,150,200,300,500,750,1000,1500,2000,3000,4000]
    checkpoints=[c for c in checkpoints if c<=steps]
    if steps not in checkpoints: checkpoints.append(steps)
    rows=[]; ci=0
    for step in range(steps+1):
        if ci<len(checkpoints) and step==checkpoints[ci]:
            rows.append(checkpoint(model,xid,sid,xood,sood,step)); ci+=1
        if step==steps: break
        idx=torch.randint(0,len(xtr),(256,)); pred=model(xtr[idx]); loss=torch.mean((pred-strue[idx])**2)
        opt.zero_grad(); loss.backward(); opt.step()
    summary={
        'seed':seed,
        'operator_5pct_step':first_step(rows,'decoded_matrix_rel_error',lambda v:v<0.05),
        'operator_1pct_step':first_step(rows,'decoded_matrix_rel_error',lambda v:v<0.01),
        'behavior_1pct_nmse_step':first_step(rows,'network_id_nmse',lambda v:v<0.01),
        'behavior_0p1pct_nmse_step':first_step(rows,'network_id_nmse',lambda v:v<0.001),
        'locally_linear_step':first_step(rows,'jacobian_input_variation',lambda v:v<0.05),
        'math_beats_neural_ood_10x_step':next((r['step'] for r in rows if r['decoded_affine_ood2x_nmse']*10<r['network_ood2x_nmse']),None),
        'final':rows[-1],
    }
    return {'organism':'2 -> tanh(16) -> 2 supervised demixer','true_law':'s = A^-1 x','mixing_matrix':A.tolist(),'exact_inverse':A_INV.tolist(),'trajectory':rows,'summary':summary}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=0); p.add_argument('--steps',type=int,default=4000); p.add_argument('--out',default='results/gate1_seed0.json'); a=p.parse_args()
    r=run(a.seed,a.steps); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(r,indent=2)); print(json.dumps(r['summary'],indent=2))
if __name__=='__main__': main()
