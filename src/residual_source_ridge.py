"""Residual source-transfer experiment.
Start from the joint-state OwnOnly raw-retrieval forecast, then ask whether other
markets' historical OOS errors add incremental information. Source coefficients are
ridge regressions estimated only on Y-3,Y-2 prequential error panels. Hyperparameters
(alpha,tau) are selected on 2016-2017 development and frozen for 2018-2019. A scalar
source multiplier is selected on Y-1 calibration for each year/seed. Uniform-source
control uses the same source-only information and calibration.
Exploratory: the family was developed after inspection of the original test years.
"""
from pathlib import Path
import json, numpy as np, pandas as pd
from paths import ROOT,R,OUT,CACHE
RES=OUT/'residual_source_ridge';RES.mkdir(exist_ok=True)
SEEDS=[11,42,73]
ALPHAS=[.1,1.,10.,100.]
TAUS=[.02,.05,.1,.2]
GAMMAS=np.linspace(-.5,1.5,41)

def ql(y,p):
 d=np.clip(np.asarray(y)-np.asarray(p),-40,40);return np.exp(d)-d-1

def macro(y,p,i):
 d=ql(y,p);return float(np.mean([d[i==j].mean() for j in np.unique(i)]))

def fit_beta(year,alpha):
 p=dict(np.load(R/'data/panel.npz'));har=dict(np.load(R/'data/har_prequential.npz'));yrs=pd.DatetimeIndex(p['dates']).year.to_numpy();E=(p['y']-har['HARX_QLIKE']).astype(float);avail=p['available']&np.isfinite(E)
 tr=(yrs>=year-3)&(yrs<=year-2);Et=E[tr].copy();At=avail[tr]
 mu=np.zeros(31);sd=np.ones(31)
 for j in range(31):
  v=Et[At[:,j],j];mu[j]=np.mean(v) if len(v) else 0.;sd[j]=max(np.std(v),.05) if len(v) else 1.
 Z=np.where(At,(Et-mu)/sd,0.)
 B=np.zeros((31,31))
 for i in range(31):
  ok=At[:,i];src=np.array([j for j in range(31) if j!=i]);X=Z[ok][:,src];y=(Et[ok,i]-mu[i])/sd[i]
  # Ridge on standardized source errors, no intercept (centered data).
  G=X.T@X+alpha*np.eye(len(src));b=np.linalg.solve(G,X.T@y);B[i,src]=b
 return B,mu,sd

def source_signal(year,alpha,tau,kind='ridge'):
 z=np.load(CACHE/f'raw_{year}.npz');E=z['E'].astype(float);M=z['M'];S=z['S'].astype(float);V=z['V'];qi=z['qi'].astype(int)
 B,mu,sd=fit_beta(year,alpha)
 Z=((E-mu[None,None,:])/sd[None,None,:])*M
 # Exclude own source from source-only message.
 rows=np.arange(len(qi))[:,None];slots=np.arange(E.shape[1])[None,:]
 M2=M.copy();M2[rows,slots,qi[:,None]]=False;Z=Z*M2
 if kind=='ridge':
  msg=np.einsum('bkn,bn->bk',Z,B[qi])*sd[qi,None]
 elif kind=='uniform':
  cnt=M2.sum(-1);zs=Z.sum(-1)/np.maximum(cnt,1);msg=zs*sd[qi,None];msg[cnt==0]=0.
 else:raise ValueError(kind)
 mx=np.where(V,S,-1e9).max(1,keepdims=True);rw=np.exp(np.clip((S-mx)/tau,-30,30))*V;rw=rw/np.maximum(rw.sum(1,keepdims=True),1e-12);sig=(rw*msg).sum(1);sig[~V.any(1)]=0.
 return pd.DataFrame({'date':z['meta_date'].astype(str),'market':z['meta_i'].astype(int),'year':z['meta_years'].astype(int),'y':z['meta_y'].astype(float),'signal':sig})

def own_view(year,seed):
 z=np.load(CACHE/f'views_Own_{year}_{seed}.npz');return pd.DataFrame({'date':z['date'].astype(str),'market':z['i'].astype(int),'year':z['years'].astype(int),'seed':seed,'y2':z['y'].astype(float),'own':z['P'][:,0].astype(float),'base':z['base'].astype(float)})

def join(year,seed,alpha,tau,kind):
 s=source_signal(year,alpha,tau,kind);o=own_view(year,seed);q=o.merge(s,on=['date','market','year'],validate='one_to_one');assert np.max(abs(q.y2-q.y))<1e-4;return q

def calibrate(q,year):
 va=q.year==year-1;vals=[]
 for g in GAMMAS:vals.append((float(g),macro(q.loc[va,'y'].to_numpy(),q.loc[va,'own'].to_numpy()+g*q.loc[va,'signal'].to_numpy(),q.loc[va,'market'].to_numpy())))
 return min(vals,key=lambda x:(x[1],abs(x[0]))),vals

def evaluate_config(alpha,tau,kind,years=(2016,2017)):
 vals=[]
 for year in years:
  for seed in SEEDS:
   q=join(year,seed,alpha,tau,kind);(g,_),_=calibrate(q,year);te=q.year==year;vals.append(macro(q.loc[te,'y'],q.loc[te,'own']+g*q.loc[te,'signal'],q.loc[te,'market']))
 return float(np.mean(vals))

def main():
 scans=[]
 for kind in ['ridge','uniform']:
  for a in ALPHAS:
   for t in TAUS:
    sc=evaluate_config(a,t,kind);scans.append({'kind':kind,'alpha':a,'tau':t,'dev_score':sc});print(kind,a,t,sc,flush=True)
 scan=pd.DataFrame(scans);scan.to_csv(RES/'dev_scan.csv',index=False)
 selected={}
 for kind in ['ridge','uniform']:
  r=scan[scan.kind==kind].sort_values(['dev_score','alpha','tau']).iloc[0];selected[kind]={'alpha':float(r.alpha),'tau':float(r.tau),'dev_score':float(r.dev_score)}
 print('selected',selected,flush=True)
 allpred=[];cal=[]
 for kind,c in selected.items():
  for year in [2018,2019]:
   sig=source_signal(year,c['alpha'],c['tau'],kind)
   for seed in SEEDS:
    o=own_view(year,seed);q=o.merge(sig,on=['date','market','year'],validate='one_to_one');(best,vals)=calibrate(q,year);g,vs=best;te=q.year==year;z=q.loc[te].copy();z['pred']=z.own+g*z.signal;z['kind']=kind;z['gamma']=g;allpred.append(z[['date','market','year','seed','y','pred','own','base','signal','kind','gamma']]);cal.append({'kind':kind,'year':year,'seed':seed,'gamma':g,'cal_score':vs})
 pred=pd.concat(allpred,ignore_index=True);pred.to_csv(RES/'predictions.csv.gz',index=False);pd.DataFrame(cal).to_csv(RES/'calibration.csv',index=False);(RES/'selection.json').write_text(json.dumps({'development_years':[2016,2017],'evaluation_years':[2018,2019],'selected':selected,'gamma_grid':GAMMAS.tolist(),'description':'Ridge source coefficients estimated on Y-3,Y-2; alpha/tau fixed from development; gamma chosen on Y-1. Exploratory family.'},indent=2)+'\n')
 for kind,g in pred.groupby('kind'): print(kind,'test',macro(g.y,g.pred,g.market),'own',macro(g.y,g.own,g.market),'gamma mean',g.gamma.mean(),flush=True)
if __name__=='__main__':main()
