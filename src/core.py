"""Portable methods extracted from the archived experiment; no alternative data or models."""
from __future__ import annotations
import math, random, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import minimize

K=16
GRID=[0.,.25,.5,.75,1.]
def qlike(y,p):
    d=np.asarray(y,float)-np.asarray(p,float)
    return np.exp(np.clip(d,-40,40))-d-1

def macro_score(y,p,m):
    l=qlike(y,p)
    return float(np.mean([np.mean(l[m==j]) for j in np.unique(m)]))

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def raw_global_state(p,year):
    yrs=pd.DatetimeIndex(p['dates']).year.to_numpy(); T,N=p['y'].shape
    H=p['h'].astype(np.float32); mask=p['available']; tr=np.flatnonzero(yrs<=year-4)
    X=np.zeros_like(H,dtype=np.float32)
    mu=np.zeros((N,H.shape[-1]),np.float32); sd=np.ones_like(mu)
    for j in range(N):
        ok=mask[tr,j] & np.isfinite(H[tr,j]).all(1)
        v=H[tr[ok],j]
        if len(v):
            mu[j]=v.mean(0); sd[j]=np.maximum(v.std(0),.05)
        X[:,j]=np.clip((H[:,j]-mu[j])/sd[j],-8,8)
    X[~mask]=0
    G=np.concatenate([X.reshape(T,-1),mask.astype(np.float32)],1)
    G=G/np.maximum(np.linalg.norm(G,axis=1,keepdims=True),1e-8)
    return X,mask,G.astype(np.float32),mu,sd

def fold_meta(p,har,year):
    yrs=pd.DatetimeIndex(p['dates']).year.to_numpy(); N=len(p['symbols'])
    avail=p['available'] & np.isfinite(har['HARX_QLIKE']) & np.isfinite(p['y'])
    mm=avail.copy(); mm[yrs<year-3]=False; mm[yrs>year]=False
    qflat=np.flatnonzero(mm.reshape(-1)); t=qflat//N; i=qflat%N
    return dict(qflat=qflat,t=t,i=i,years=yrs[t],date=p['dates'][t],y=p['y'].reshape(-1)[qflat].astype(np.float32),base=har['HARX_QLIKE'].reshape(-1)[qflat].astype(np.float32))

class LearnedGlobalEncoder(nn.Module):
    def __init__(self,n_markets=31,state_dim=8,d=32,me=6,heads=4):
        super().__init__(); self.n=n_markets; self.d=d
        self.market=nn.Parameter(torch.randn(n_markets,me)*.03)
        self.token=nn.Sequential(nn.Linear(state_dim+me+1,d),nn.SiLU(),nn.Linear(d,d),nn.LayerNorm(d))
        self.attn=nn.MultiheadAttention(d,heads,batch_first=True,dropout=0.0)
        self.ff=nn.Sequential(nn.Linear(d,d*2),nn.SiLU(),nn.Linear(d*2,d))
        self.norm=nn.LayerNorm(d)
        self.poolq=nn.Parameter(torch.randn(d)*.03)
        self.proj=nn.Sequential(nn.Linear(d*2,d),nn.SiLU(),nn.Linear(d,d))
        self.err_head=nn.Linear(d,n_markets)
    def forward(self,x,mask):
        # x B,N,F, mask B,N True=available
        B,N,_=x.shape
        me=self.market[None].expand(B,-1,-1)
        tok=self.token(torch.cat([x,me,mask.float().unsqueeze(-1)],-1))
        a,_=self.attn(tok,tok,tok,key_padding_mask=~mask,need_weights=False)
        tok=self.norm(tok+a+self.ff(tok+a))
        sc=(tok*self.poolq).sum(-1)/math.sqrt(self.d); sc=sc.masked_fill(~mask,-1e4)
        w=torch.softmax(sc,-1)*mask; w=w/w.sum(-1,keepdim=True).clamp_min(1e-8)
        att=(w.unsqueeze(-1)*tok).sum(1)
        mean=(tok*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        z=self.proj(torch.cat([att,mean],-1)); z=F.normalize(z,dim=-1)
        return z,self.err_head(z)

def prepare_regime_targets(p,har,year,X,mask):
    yrs=pd.DatetimeIndex(p['dates']).year.to_numpy(); err=p['y']-har['HARX_QLIKE']; train=(yrs<=year-4)
    dayok=train & (np.sum(np.isfinite(err)&p['available'],axis=1)>=10)
    ids=np.flatnonzero(dayok)
    emu=np.zeros(err.shape[1],np.float32); esd=np.ones(err.shape[1],np.float32)
    for j in range(err.shape[1]):
        v=err[ids,j]; ok=np.isfinite(v)
        if ok.sum()>20:
            emu[j]=np.mean(v[ok]); esd[j]=max(np.std(v[ok]),.1)
    ez=(err-emu)/esd
    valid=np.isfinite(ez)&p['available']; ez=np.nan_to_num(ez,nan=0.0).astype(np.float32)
    return ids,ez,valid,emu,esd

def train_encoder(p,har,year,seed,out,metric_refine=False,epochs=12):
    tag='metric' if metric_refine else 'reg'
    fn=out/f'encoder_{tag}_{year}_{seed}.pt'; zfn=out/f'Z_{tag}_{year}_{seed}.npy'
    X,mask,Graw,mu,sd=raw_global_state(p,year)
    if fn.exists() and zfn.exists():
        return np.load(zfn),Graw
    ids,ez,ev,emu,esd=prepare_regime_targets(p,har,year,X,mask)
    seed_all(seed+5000+(100 if metric_refine else 0))
    m=LearnedGlobalEncoder(len(p['symbols']),X.shape[-1],32)
    opt=torch.optim.AdamW(m.parameters(),lr=1.5e-3,weight_decay=2e-3)
    rng=np.random.default_rng(seed+5000)
    xt=torch.tensor(X); mt=torch.tensor(mask); yt=torch.tensor(ez); vt=torch.tensor(ev)
    bs=96
    for ep in range(epochs):
        m.train(); order=rng.permutation(ids)
        for ii in np.array_split(order,max(1,math.ceil(len(order)/bs))):
            if not len(ii): continue
            z,pred=m(xt[ii],mt[ii]); v=vt[ii]; diff=F.smooth_l1_loss(pred,yt[ii],reduction='none')
            loss=(diff*v).sum()/v.sum().clamp_min(1)
            # Global mean error direction matters for regime-level volatility surprises.
            pm=(pred*v).sum(-1)/v.sum(-1).clamp_min(1); tm=(yt[ii]*v).sum(-1)/v.sum(-1).clamp_min(1)
            loss=loss+0.15*F.smooth_l1_loss(pm,tm)
            if metric_refine and len(ii)>=8:
                # Make embedding geometry track the cross-market HARX-error pattern.
                evv=yt[ii]*v
                norm=torch.sqrt((evv*evv).sum(-1,keepdim=True)).clamp_min(1e-6)
                et=evv/norm
                target=(et@et.T).detach()
                predsim=z@z.T
                ov=(v.float()@v.float().T)>=8
                eye=torch.eye(len(ii),dtype=torch.bool)
                pmask=ov & ~eye
                if pmask.any(): loss=loss+0.30*F.mse_loss(predsim[pmask],target[pmask])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),2); opt.step()
    m.eval(); Z=[]
    with torch.no_grad():
        for st in range(0,len(X),256):
            z,_=m(xt[st:st+256],mt[st:st+256]); Z.append(z.cpu().numpy())
    Z=np.concatenate(Z).astype(np.float32)
    torch.save({'state_dict':m.state_dict(),'year':year,'seed':seed,'metric_refine':metric_refine,'train_dates':len(ids),'emu':emu,'esd':esd},fn)
    np.save(zfn,Z)
    return Z,Graw

def retrieve_dates(p,har,Z,meta,seed,kind='true',K=16):
    di=p['dates'].astype(int); err=p['y']-har['HARX_QLIKE']
    bank=np.flatnonzero((np.isfinite(err)&p['available']).sum(1)>=10)
    cut=p['window_start'].reshape(-1)[meta['qflat']]
    Q=len(cut); I=np.zeros((Q,K),np.int32); V=np.zeros((Q,K),bool); S=np.zeros((Q,K),np.float32)
    rng=np.random.default_rng(seed+731)
    for st in range(0,Q,256):
        en=min(Q,st+256); elig=di[bank][None,:]<cut[st:en,None]
        sim=Z[meta['t'][st:en]]@Z[bank].T
        rank=rng.random(sim.shape,dtype=np.float32) if kind=='random' else sim.copy(); rank[~elig]=-1e9
        kk=min(96,len(bank)); ix=np.argpartition(rank,-kk,axis=1)[:,-kk:]; vv=np.take_along_axis(rank,ix,1); ix=np.take_along_axis(ix,np.argsort(-vv,1),1)
        for r in range(en-st):
            pick=[]; weeks={}
            for b in ix[r]:
                if not elig[r,b]: continue
                wk=int(di[bank[b]]//7)
                if weeks.get(wk,0)>=2: continue
                pick.append(b); weeks[wk]=weeks.get(wk,0)+1
                if len(pick)==K: break
            n=len(pick)
            if n:
                I[st+r,:n]=bank[pick]; V[st+r,:n]=True; S[st+r,:n]=sim[r,pick]
    assert np.all(di[I][V] < np.broadcast_to(cut[:,None],I.shape)[V])
    return dict(index=I,valid=V,sim=S)

def make_pack(p,har,meta,pk,shuffle=False,seed=0,K=16):
    I=pk['index'][:,:K]; V=pk['valid'][:,:K]; S=pk['sim'][:,:K].astype(np.float32)
    err=p['y']-har['HARX_QLIKE']; E=err[I].astype(np.float32); M=np.isfinite(E)&p['available'][I]&V[:,:,None]
    E=np.nan_to_num(E,nan=0.0); E=np.clip(E,-4,4)
    if shuffle:
        rng=np.random.default_rng(seed); perm={}
        for s in np.unique(I[V]):
            valid=np.isfinite(err[s])&p['available'][s]; ix=np.flatnonzero(valid); pp=ix.copy(); rng.shuffle(pp); perm[int(s)]=(ix,pp)
        for b in range(len(I)):
            for k,s in enumerate(I[b]):
                if not V[b,k]: continue
                ix,pp=perm[int(s)]; tmp=E[b,k].copy(); E[b,k,ix]=tmp[pp]
    return dict(E=torch.tensor(E),M=torch.tensor(M),S=torch.tensor(S),V=torch.tensor(V),qi=torch.tensor(meta['i'].astype(np.int64)))

class RegimeMarketGraph(nn.Module):
    def __init__(self,n=31,rank=8):
        super().__init__(); self.u=nn.Parameter(torch.randn(n,rank)*.03); self.v=nn.Parameter(torch.randn(n,rank)*.03)
        self.diag=nn.Parameter(torch.tensor(3.4)); self.logtau=nn.Parameter(torch.tensor(math.log(.05))); self.scale=nn.Parameter(torch.tensor(-.5)); self.n=n
    def adjacency(self):
        logits=(self.u@self.v.T)/math.sqrt(self.u.shape[1])+self.diag*torch.eye(self.n,device=self.u.device)
        return torch.softmax(logits,-1)
    def forward(self,E,M,S,V,qi):
        A=self.adjacency()[qi]
        w=A[:,None,:]*M; w=w/w.sum(-1,keepdim=True).clamp_min(1e-8); per=(w*E).sum(-1)
        tau=torch.exp(self.logtau).clamp(.01,.5); mx=S.masked_fill(~V,-1e4).max(-1,keepdim=True).values
        rw=torch.exp(((S-mx)/tau).clamp(-30,30))*V; rw=rw/rw.sum(-1,keepdim=True).clamp_min(1e-8)
        raw=(rw*per).sum(-1); c=2.5*torch.tanh(raw/2.5)*torch.sigmoid(self.scale)*2
        return torch.where(V.any(-1),c,torch.zeros_like(c))

def sub(x,ii): return {k:v[ii] for k,v in x.items()}

def predict_graph(m,x,rows):
    out=[];m.eval()
    with torch.no_grad():
        for ii in np.array_split(rows,max(1,math.ceil(len(rows)/1024))):
            if len(ii): out.append(m(**sub(x,ii)).cpu().numpy())
    return np.concatenate(out)

def fit_graph(p,har,meta,pk,year,seed,steps=450,shuffle=False,shseed=0):
    tr=np.flatnonzero((meta['years']>=year-3)&(meta['years']<=year-2)); va=np.flatnonzero(meta['years']==year-1); te=np.flatnonzero(meta['years']==year)
    x=make_pack(p,har,meta,pk,shuffle,shseed,K)
    baseq=meta['base'].astype(np.float32); y=torch.tensor(meta['y']); bt=torch.tensor(baseq)
    cnt=np.bincount(meta['i'][tr],minlength=len(p['symbols'])); mw=torch.tensor((1/np.maximum(cnt[meta['i']],1)).astype(np.float32))
    seed_all(seed+8800); rng=np.random.default_rng(seed+8800); m=RegimeMarketGraph(len(p['symbols']))
    opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.002)
    for step in range(steps):
        ii=rng.choice(tr,min(512,len(tr)),replace=len(tr)<512); c=m(**sub(x,ii)); d=(y[ii]-bt[ii]-.5*c).clamp(-15,15)
        loss=((torch.exp(d)-d-1)*mw[ii]).sum()/mw[ii].sum().clamp_min(1e-8)+1e-4*(c*c).mean()+1e-4*((m.u*m.u).mean()+(m.v*m.v).mean())
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),2);opt.step()
    cv=predict_graph(m,x,va); grid=[(a,macro_score(meta['y'][va],baseq[va]+a*cv,meta['i'][va])) for a in GRID]
    lam,vs=min(grid,key=lambda z:(z[1],z[0])); ct=predict_graph(m,x,te); pr=baseq[te]+lam*ct
    info=dict(lambda_=lam,val=vs,test=macro_score(meta['y'][te],pr,meta['i'][te]),base=macro_score(meta['y'][te],baseq[te],meta['i'][te]),train_n=len(tr),test_n=len(te),tau=float(torch.exp(m.logtau).detach()),diag=float(m.diag.detach()))
    return m,info,pr,te

def simple_global(p,har,meta,pk,rows,alpha=.5,temp=.05):
    err=p['y']-har['HARX_QLIKE']; out=np.zeros(len(rows),np.float32)
    for zz,r in enumerate(rows):
        vals=[]; sims=[]; i=meta['i'][r]
        for k,s in enumerate(pk['index'][r]):
            if not pk['valid'][r,k]: continue
            e=err[s]; m=np.isfinite(e)&p['available'][s]
            if not m.any(): continue
            gm=float(np.nanmean(e[m])); sm=float(e[i]) if m[i] else gm
            vals.append(alpha*sm+(1-alpha)*gm); sims.append(pk['sim'][r,k])
        if vals:
            ss=np.asarray(sims); w=np.exp((ss-ss.max())/temp); w=w/w.sum(); out[zz]=w@np.asarray(vals)
    return out
