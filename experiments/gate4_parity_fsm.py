#!/usr/bin/env python3
"""
Gate 4: decode a recurrent neural computation into a tiny executable state machine.

A GRU is trained on running parity. The true algorithm is a 2-state automaton, but the
decoder is not given parity labels or the automaton. It only sees:
  - input bits,
  - hidden states,
  - neural output predictions.

The decoder overclusters hidden-state space, builds an empirical transition table
cluster x input -> next cluster, then merges behaviorally equivalent states by partition
refinement. The extracted FSM is run on much longer sequences than the neural training
horizon.

Finally, hidden-state centroids are used for a causal state-swap intervention.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn

class ParityGRU(nn.Module):
    def __init__(self,hidden=8):
        super().__init__(); self.hidden=hidden; self.gru=nn.GRU(1,hidden,batch_first=True); self.head=nn.Linear(hidden,2)
    def forward(self,x,h0=None,return_hidden=False):
        hseq,hn=self.gru(x,h0); logits=self.head(hseq)
        return (logits,hseq,hn) if return_hidden else logits

def make_batch(batch,T,seed=None):
    if seed is None: bits=torch.randint(0,2,(batch,T))
    else:
        g=torch.Generator().manual_seed(seed); bits=torch.randint(0,2,(batch,T),generator=g)
    parity=torch.cumsum(bits,dim=1)%2
    return bits.float().unsqueeze(-1), parity.long(), bits.long()

def train(seed=0,steps=2000,T=24):
    torch.manual_seed(seed); np.random.seed(seed); m=ParityGRU(8); opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-5)
    cps=[]; cp={0,100,300,600,1000,2000,3000}
    for step in range(steps+1):
        if step in cp:
            x,y,_=make_batch(256,T,seed+10000+step)
            with torch.no_grad(): pred=m(x).argmax(-1); acc=float((pred==y).float().mean())
            cps.append({'step':step,'train_horizon_accuracy':acc})
        if step==steps: break
        x,y,_=make_batch(256,T)
        logits=m(x); loss=nn.functional.cross_entropy(logits.reshape(-1,2),y.reshape(-1))
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
    return m,cps

def kmeans(X,k,seed=0,iters=50):
    rng=np.random.default_rng(seed); X=np.asarray(X,np.float64)
    centers=X[rng.choice(len(X),size=k,replace=False)].copy()
    labels=np.zeros(len(X),dtype=np.int64)
    for _ in range(iters):
        d=((X[:,None,:]-centers[None,:,:])**2).sum(-1); new=d.argmin(1)
        if np.array_equal(new,labels): break
        labels=new
        for j in range(k):
            pts=X[labels==j]
            if len(pts): centers[j]=pts.mean(0)
    d=((X[:,None,:]-centers[None,:,:])**2).sum(-1); labels=d.argmin(1)
    return labels,centers

def build_candidate(hseq,bits,neural_pred,k,seed=0):
    B,T,H=hseq.shape
    prev=np.zeros((B,T,H),dtype=np.float64); prev[:,1:]=hseq[:,:-1]
    flat_prev=prev.reshape(-1,H); flat_next=hseq.reshape(-1,H); flat_bits=bits.reshape(-1); flat_out=neural_pred.reshape(-1)
    all_states=np.concatenate([flat_prev,flat_next],axis=0)
    all_labels,centers=kmeans(all_states,k,seed)
    lp=all_labels[:len(flat_prev)]; ln=all_labels[len(flat_prev):]
    trans=np.zeros((k,2,k),dtype=np.int64); out_counts=np.zeros((k,2),dtype=np.int64)
    for a,b,n,o in zip(lp,flat_bits,ln,flat_out):
        trans[a,b,n]+=1; out_counts[n,o]+=1
    table=trans.argmax(-1); out_map=out_counts.argmax(-1)
    trans_cons=float(sum(trans[s,b,table[s,b]] for s in range(k) for b in range(2))/(trans.sum()+1e-12))
    out_cons=float(sum(out_counts[s,out_map[s]] for s in range(k))/(out_counts.sum()+1e-12))
    reset=int(np.argmin(np.sum(centers**2,axis=1)))
    counts=np.bincount(np.concatenate([lp,ln]),minlength=k)
    return {'k':k,'transition_consistency':trans_cons,'output_consistency':out_cons,'transition_table':table.tolist(),'output_map':out_map.tolist(),'reset_state':reset,'centers':centers.tolist(),'cluster_counts':counts.tolist()}

def minimize_candidate(c):
    tab=np.asarray(c['transition_table'],dtype=int); out=np.asarray(c['output_map'],dtype=int); k=c['k']
    block=np.array(out,dtype=int)
    _,block=np.unique(block,return_inverse=True)
    while True:
        sig=[(int(out[s]),int(block[tab[s,0]]),int(block[tab[s,1]])) for s in range(k)]
        mapping={}; nb=np.zeros(k,dtype=int)
        for s,sg in enumerate(sig):
            if sg not in mapping: mapping[sg]=len(mapping)
            nb[s]=mapping[sg]
        if np.array_equal(nb,block): break
        block=nb
    K=int(block.max()+1)
    mtab=np.zeros((K,2),dtype=int); mout=np.zeros(K,dtype=int); members=[]; reps=[]
    counts=np.asarray(c.get('cluster_counts',[1]*k)); centers=np.asarray(c['centers'])
    for b in range(K):
        ms=np.flatnonzero(block==b).tolist(); members.append(ms); r=max(ms,key=lambda s:int(counts[s])); reps.append(centers[r].tolist()); mout[b]=out[r]
        for bit in range(2): mtab[b,bit]=block[tab[r,bit]]
    return {'k':K,'raw_k':k,'transition_consistency':c['transition_consistency'],'output_consistency':c['output_consistency'],'transition_table':mtab.tolist(),'output_map':mout.tolist(),'reset_state':int(block[c['reset_state']]),'centers':reps,'raw_members':members}

def extract_fsm(model,seed=0):
    x,y,bits=make_batch(2048,32,seed+20000)
    with torch.no_grad(): logits,hseq,_=model(x,return_hidden=True); pred=logits.argmax(-1)
    H=hseq.detach().numpy(); B=bits.numpy(); P=pred.numpy()
    raw=[build_candidate(H,B,P,k,seed+30000+k) for k in range(2,7)]
    minimized=[minimize_candidate(c) for c in raw]
    good=[c for c in minimized if c['transition_consistency']>=0.995 and c['output_consistency']>=0.995]
    chosen=min(good,key=lambda c:(c['k'],-c['transition_consistency']*c['output_consistency'],c['raw_k'])) if good else max(minimized,key=lambda c:c['transition_consistency']*c['output_consistency'])
    return chosen,{'raw':raw,'minimized':minimized}

def run_fsm(bits,cand):
    state=cand['reset_state']; out=[]; tab=cand['transition_table']; om=cand['output_map']
    for b in bits:
        state=tab[state][int(b)]; out.append(om[state])
    return np.array(out,dtype=np.int64)

def long_eval(model,cand,seed=0,T=1024):
    x,y,bits=make_batch(256,T,seed+40000)
    with torch.no_grad(): npred=model(x).argmax(-1).numpy()
    truth=y.numpy(); B=bits.numpy(); fp=np.stack([run_fsm(row,cand) for row in B])
    return {'neural_long_accuracy_vs_truth':float((npred==truth).mean()),'fsm_long_accuracy_vs_truth':float((fp==truth).mean()),'fsm_fidelity_vs_neural':float((fp==npred).mean())}

def swap_intervention(model,cand,seed=0,trials=256,prefix=12,suffix=12):
    centers=torch.tensor(cand['centers'],dtype=torch.float32)
    if cand['k']<2: return {'performed':False}
    success=[]; first=[]
    for i in range(trials):
        x,y,bits=make_batch(1,prefix+suffix,seed+50000+i)
        with torch.no_grad():
            _,hpre,hn=model(x[:,:prefix],return_hidden=True)
        hcur=hn[0,0]; d=torch.sum((centers-hcur[None,:])**2,dim=1); cur=int(torch.argmin(d))
        others=[j for j in range(cand['k']) if j!=cur]; target=max(others,key=lambda j:float(torch.sum((centers[j]-hcur)**2)))
        hswap=centers[target].reshape(1,1,-1)
        with torch.no_grad(): slogits,_,_=model(x[:,prefix:],hswap,return_hidden=True); sp=slogits.argmax(-1).numpy()[0]
        state=target; expected=[]
        for b in bits.numpy()[0,prefix:]:
            state=cand['transition_table'][state][int(b)]; expected.append(cand['output_map'][state])
        expected=np.array(expected)
        success.append(float((sp==expected).mean())); first.append(float(sp[0]==expected[0]))
    return {'performed':True,'suffix_fidelity_to_injected_fsm':float(np.mean(success)),'first_step_fidelity':float(np.mean(first))}

def run(seed=0,steps=2000):
    model,cps=train(seed,steps,24); chosen,candidates=extract_fsm(model,seed); le=long_eval(model,chosen,seed,1024); swap=swap_intervention(model,chosen,seed)
    return {'seed':seed,'train_horizon':24,'hidden_size':8,'chosen_fsm':chosen,'candidates':candidates,'long_eval':le,'state_swap':swap,'checkpoints':cps}

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=2000);p.add_argument('--out',default='results/gate4_seed0.json');a=p.parse_args();r=run(a.seed,a.steps);Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(r,indent=2));print(json.dumps({'chosen':r['chosen_fsm'],'long':r['long_eval'],'swap':r['state_swap']},indent=2))
if __name__=='__main__': main()
