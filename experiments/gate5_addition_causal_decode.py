#!/usr/bin/env python3
"""
Gate 5: decode column addition by causal response equivalence.

A GRU sees pairs of decimal digits from least-significant to most-significant and must
emit each output digit of their sum. It is trained only on short sequences. The hidden
algorithm requires a carry state.

A first geometric decoder failed: Euclidean hidden-state centroids were not valid causal
states even though the neural network was almost perfect.

This decoder instead defines state by what it DOES under interventions:
  hidden state h
      -> query every digit-pair input
      -> 100-output response signature
      -> cluster states with the same causal response map

A small automatically selected diagnostic probe set then classifies successor hidden
states, allowing construction of a complete causal Mealy transducer. Finally a tiny
symbolic search asks whether that transducer is exactly base-B carry arithmetic.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn

BASE=10
INPUT_DIM=20

class AddGRU(nn.Module):
    def __init__(self,hidden=16):
        super().__init__()
        self.hidden=hidden
        self.gru=nn.GRU(INPUT_DIM,hidden,batch_first=True)
        self.head=nn.Linear(hidden,BASE)
    def forward(self,x,h0=None,return_hidden=False):
        hseq,hn=self.gru(x,h0)
        logits=self.head(hseq)
        return (logits,hseq,hn) if return_hidden else logits

def encode_pairs(a,b):
    return torch.cat([
        nn.functional.one_hot(a.long(),BASE).float(),
        nn.functional.one_hot(b.long(),BASE).float()
    ],dim=-1)

def make_batch(batch,T,seed=None):
    if seed is None:
        a=torch.randint(0,BASE,(batch,T))
        b=torch.randint(0,BASE,(batch,T))
    else:
        g=torch.Generator().manual_seed(seed)
        a=torch.randint(0,BASE,(batch,T),generator=g)
        b=torch.randint(0,BASE,(batch,T),generator=g)
    carry=torch.zeros(batch,dtype=torch.long)
    ys=[]
    for t in range(T):
        total=a[:,t]+b[:,t]+carry
        ys.append(total%BASE)
        carry=total//BASE
    return encode_pairs(a,b),torch.stack(ys,1),a,b

def train(seed=0,steps=2000,T=8):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m=AddGRU(16)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-5)
    cps=[]
    cp={0,100,300,600,1000,2000,3000}
    for step in range(steps+1):
        if step in cp:
            x,y,_,_=make_batch(512,T,seed+10000+step)
            with torch.no_grad():
                acc=float((m(x).argmax(-1)==y).float().mean())
            cps.append({'step':step,'train_horizon_accuracy':acc})
        if step==steps:
            break
        x,y,_,_=make_batch(512,T)
        log=m(x)
        loss=nn.functional.cross_entropy(log.reshape(-1,BASE),y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
        opt.step()
    return m,cps

def symbols_tensor():
    a=torch.tensor([a for a in range(BASE) for b in range(BASE)])
    b=torch.tensor([b for a in range(BASE) for b in range(BASE)])
    return encode_pairs(a[:,None],b[:,None]),[(int(aa),int(bb)) for aa,bb in zip(a,b)]

def query_many(model,states,input_indices=None):
    """Query all selected symbols from each injected hidden state."""
    X,symbols=symbols_tensor()
    if input_indices is not None:
        X=X[input_indices]
        symbols=[symbols[i] for i in input_indices]
    N=len(states)
    M=len(X)
    xb=X.repeat(N,1,1)
    h0=states[:,None,:].repeat(1,M,1).reshape(N*M,-1).unsqueeze(0)
    with torch.no_grad():
        log,_,hn=model(xb,h0,return_hidden=True)
    return log[:,0].reshape(N,M,BASE),hn[0].reshape(N,M,-1),symbols

def kmeans(X,k,seed=0,iters=60):
    rng=np.random.default_rng(seed)
    X=np.asarray(X,np.float64)
    cent=X[rng.choice(len(X),k,replace=False)].copy()
    lab=np.full(len(X),-1,dtype=int)
    for _ in range(iters):
        d=((X[:,None,:]-cent[None,:,:])**2).sum(-1)
        nl=d.argmin(1)
        if np.array_equal(nl,lab):
            break
        lab=nl
        for j in range(k):
            pts=X[lab==j]
            if len(pts):
                cent[j]=pts.mean(0)
    return lab,cent

def collect_prev_states(model,seed=0,nseq=512,T=12):
    x,y,a,b=make_batch(nseq,T,seed+20000)
    with torch.no_grad():
        _,hseq,_=model(x,return_hidden=True)
    prev=torch.zeros_like(hseq)
    prev[:,1:]=hseq[:,:-1]
    H=prev.reshape(-1,model.hidden)
    idx=torch.randperm(len(H))[:1024]
    return torch.cat([H[idx],torch.zeros(32,model.hidden)],0)

def discover_states(model,seed=0):
    H=collect_prev_states(model,seed)
    logits,_,symbols=query_many(model,H)
    pred=logits.argmax(-1)

    # Categorical response signatures: two hidden states are considered similar when
    # they answer the same counterfactual input queries, not when they are nearby in
    # Euclidean neural coordinates.
    sig=nn.functional.one_hot(pred,BASE).float().reshape(len(H),-1).numpy()

    candidates=[]
    for k in range(1,5):
        lab,_=kmeans(sig,k,seed+30000+k)
        table=np.zeros((k,100),dtype=int)
        cons=[]
        counts=np.bincount(lab,minlength=k)
        for s in range(k):
            rows=pred.numpy()[lab==s]
            if not len(rows):
                continue
            for j in range(100):
                bc=np.bincount(rows[:,j],minlength=BASE)
                table[s,j]=bc.argmax()
                cons.append(bc.max()/bc.sum())
        consistency=float(np.mean(cons)) if cons else 0
        candidates.append((k,consistency,lab,table,counts))

    good=[c for c in candidates if c[1]>=0.995]
    chosen=min(good,key=lambda z:z[0]) if good else max(candidates,key=lambda z:z[1])
    k,cons,lab,out_table,counts=chosen

    # Diagnostic probes are input symbols whose expected answers distinguish the
    # discovered causal states. They are later used to classify successor states.
    scores=[]
    for j in range(100):
        scores.append((len(set(out_table[:,j].tolist())),j))
    probes=[j for _,j in sorted(scores,reverse=True)
            if len(set(out_table[:,j].tolist()))>1][:8]

    reps=[]
    for s in range(k):
        ids=np.flatnonzero(lab==s)
        reps.append(H[ids[:min(32,len(ids))]])

    return {
        'k':k,
        'signature_consistency':cons,
        'labels':lab,
        'output_table':out_table,
        'counts':counts,
        'probes':probes,
        'representatives':reps,
        'H':H,
        'candidate_summary':[{'k':c[0],'consistency':c[1]} for c in candidates]
    }

def classify_by_probes(model,states,disc):
    probes=disc['probes']
    log,_,_=query_many(model,states,probes)
    p=log.argmax(-1).numpy()
    expected=disc['output_table'][:,probes]
    labels=[]
    agreement=[]
    for row in p:
        m=np.mean(row[None,:]==expected,axis=1)
        labels.append(int(np.argmax(m)))
        agreement.append(float(np.max(m)))
    return np.array(labels),np.array(agreement)

def build_transducer(model,disc):
    k=disc['k']
    out=np.zeros((k,100),dtype=int)
    trans=np.zeros((k,100),dtype=int)
    ocon=[]
    tcon=[]

    for s in range(k):
        reps=disc['representatives'][s]
        log,next_h,_=query_many(model,reps)
        pred=log.argmax(-1).numpy()

        out[s]=np.array([
            np.bincount(pred[:,j],minlength=BASE).argmax()
            for j in range(100)
        ])
        ocon.extend([np.mean(pred[:,j]==out[s,j]) for j in range(100)])

        flat=next_h.reshape(-1,model.hidden)
        lab,agree=classify_by_probes(model,flat,disc)
        lab=lab.reshape(len(reps),100)

        for j in range(100):
            trans[s,j]=np.bincount(lab[:,j],minlength=k).argmax()
            tcon.append(np.mean(lab[:,j]==trans[s,j]))

    reset,_=classify_by_probes(model,torch.zeros(1,model.hidden),disc)

    # Keep an actual observed hidden state for interventions instead of a geometric
    # centroid, because the failed precursor showed that centroids can be off-manifold.
    representatives=[disc['representatives'][s][0].tolist() for s in range(k)]

    return {
        'k':k,
        'output_table':out.tolist(),
        'transition_table':trans.tolist(),
        'reset_state':int(reset[0]),
        'centers':representatives,
        'output_consistency':float(np.mean(ocon)),
        'transition_consistency':float(np.mean(tcon)),
        'diagnostic_probes':disc['probes'],
        'signature_consistency':disc['signature_consistency'],
        'candidate_summary':disc['candidate_summary']
    }

def infer_program(machine):
    if machine['k']!=2:
        return {'found':False,'mismatches':999999}

    out=np.asarray(machine['output_table'])
    tab=np.asarray(machine['transition_table'])
    symbols=[(a,b) for a in range(BASE) for b in range(BASE)]
    best=None

    # Search base and abstract-state -> carry-bit assignment. No true carry labels are
    # available to this search.
    for B in range(2,17):
        for carry_values in [(0,1),(1,0)]:
            c2s={carry_values[s]:s for s in range(2)}
            mism=0
            for s,c in enumerate(carry_values):
                for j,(a,b) in enumerate(symbols):
                    total=a+b+c
                    po=total%B
                    pc=int(total>=B)
                    mism+=int(po!=out[s,j])
                    mism+=int(c2s.get(pc,-1)!=tab[s,j])
            q={'base':B,'state_to_carry':list(carry_values),'mismatches':mism}
            if best is None or mism<best['mismatches']:
                best=q

    best['found']=best['mismatches']==0
    if best['found']:
        best['program']='total=a+b+carry; digit=total%base; carry_next=int(total>=base)'
    return best

def run_machine(a,b,m):
    state=m['reset_state']
    ys=[]
    for aa,bb in zip(a,b):
        j=int(aa)*10+int(bb)
        ys.append(m['output_table'][state][j])
        state=m['transition_table'][state][j]
    return np.array(ys)

def long_eval(model,m,seed=0,T=256):
    x,y,a,b=make_batch(256,T,seed+40000)
    with torch.no_grad():
        npred=model(x).argmax(-1).numpy()
    fp=np.stack([run_machine(a[i].numpy(),b[i].numpy(),m) for i in range(len(a))])
    truth=y.numpy()
    return {
        'neural_long_accuracy':float((npred==truth).mean()),
        'machine_long_accuracy':float((fp==truth).mean()),
        'machine_fidelity_vs_neural':float((fp==npred).mean())
    }

def state_swap(model,m,disc,seed=0,trials=256,prefix=6,suffix=16):
    if m['k']!=2:
        return {'performed':False}

    reps=[disc['representatives'][s][0] for s in range(2)]
    scores=[]
    first=[]

    for i in range(trials):
        x,y,a,b=make_batch(1,prefix+suffix,seed+50000+i)
        with torch.no_grad():
            _,_,hn=model(x[:,:prefix],return_hidden=True)

        cur,_=classify_by_probes(model,hn[0],disc)
        target=1-int(cur[0])
        hs=reps[target].reshape(1,1,-1)

        with torch.no_grad():
            log,_,_=model(x[:,prefix:],hs,return_hidden=True)
        npred=log.argmax(-1).numpy()[0]

        state=target
        exp=[]
        for aa,bb in zip(a.numpy()[0,prefix:],b.numpy()[0,prefix:]):
            j=int(aa)*10+int(bb)
            exp.append(m['output_table'][state][j])
            state=m['transition_table'][state][j]
        exp=np.array(exp)

        scores.append(np.mean(npred==exp))
        first.append(npred[0]==exp[0])

    return {
        'performed':True,
        'first_step_fidelity':float(np.mean(first)),
        'suffix_fidelity':float(np.mean(scores))
    }

def run(seed=0,steps=2000):
    model,cps=train(seed,steps,8)
    disc=discover_states(model,seed)
    machine=build_transducer(model,disc)
    program=infer_program(machine)
    le=long_eval(model,machine,seed)
    swap=state_swap(model,machine,disc,seed)

    return {
        'seed':seed,
        'train_horizon':8,
        'hidden_size':16,
        'state_discovery':{
            'k':disc['k'],
            'signature_consistency':disc['signature_consistency'],
            'diagnostic_probes':disc['probes'],
            'candidate_summary':disc['candidate_summary']
        },
        'machine':machine,
        'program':program,
        'long_eval':le,
        'state_swap':swap,
        'checkpoints':cps
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--out',default='results/gate5_seed0.json')
    a=p.parse_args()
    r=run(a.seed,a.steps)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(r,indent=2))
    print(json.dumps({
        'discovery':r['state_discovery'],
        'machine':{
            'k':r['machine']['k'],
            'out_cons':r['machine']['output_consistency'],
            'trans_cons':r['machine']['transition_consistency']
        },
        'program':r['program'],
        'long':r['long_eval'],
        'swap':r['state_swap']
    },indent=2))

if __name__=='__main__':
    main()
