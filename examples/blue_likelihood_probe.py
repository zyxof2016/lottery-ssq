"""Fixed convex-mixture check on causal features; no automatic adoption.
Usage: python3 examples/blue_likelihood_probe.py blue-features.json
Training excludes the first 30 expert warmup rows, reserves 312 pre-boundary
rows, and is frozen before both that reserve and the original confirmation.
"""
import hashlib,json,math,sys
source=json.load(open(sys.argv[1]));rows=source['features']['rows'];pre=[r for r in rows if r['issue']<25111]
if len(pre)<343:raise ValueError('Need at least 343 pre-boundary evaluation rows')
train=pre[:-312];x=[[p[t['winner']]for p in t['p']]+[1/16]for t in train];w=[1/7]*7
for iteration in range(1000):
    mix=[sum(a*b for a,b in zip(v,w))for v in x]
    gradient=[sum(v[j]/p for v,p in zip(x,mix))for j in range(7)]
    recipient=max(range(7),key=lambda j:gradient[j]);donor=min((j for j in range(7)if w[j]>1e-14),key=lambda j:gradient[j])
    gap=(gradient[recipient]-gradient[donor])/len(x)
    if gap<1e-8:break
    delta=[v[recipient]-v[donor]for v in x];cap=w[donor]
    def derivative(a):return sum(d/(p+a*d)for p,d in zip(mix,delta))
    if derivative(cap)>=0:step=cap
    else:
        left,right=0.,cap
        for _ in range(45):
            step=(left+right)/2
            if derivative(step)>0:left=step
            else:right=step
        step=(left+right)/2
    w[recipient]+=step;w[donor]-=step
windows={'pre_boundary_validation':rows[len(train):len(train)+312],'confirmation_exploratory':[r for r in rows if 25111<=r['issue']<=26115],'tracking':[r for r in rows if r['issue']>=26116],'all_held_out':rows[len(train):],'recent60':rows[-60:]}
result={}
for name,rr in windows.items():
    hits=[0,0];log=[0.,0.];gain=loss=0;last_rank=None
    for t in rr:
        p=[sum(w[e]*t['p'][e][i]for e in range(6))+w[6]/16 for i in range(16)]
        old=sorted(range(16),key=lambda i:(-math.floor(t['baseline'][i]*1e9+.5),i));order=sorted(old,key=lambda i:-math.floor(p[i]*1e9+.5));y=t['winner']
        a,b=y in old[:2],y in order[:2];hits[0]+=a;hits[1]+=b;gain+=b and not a;loss+=a and not b
        log[0]+=math.log(t['baseline'][y]*16);log[1]+=math.log(p[y]*16);last_rank=order.index(y)+1
    result[name]={'n':len(rr),'weighted_top2_hits':hits[0],'mixture_top2_hits':hits[1],'uniform_expected_top2_hits':len(rr)/8,'paired_gains':gain,'paired_losses':loss,'mean_blue_log_excess_vs_uniform':[v/len(rr)if rr else None for v in log],'last_period_mixture_winner_rank':last_rank}
report={'kind':'fixed_blue_convex_mixture_research','data_sha256':source['data_sha256'],'source_sha256':hashlib.sha256(open(__file__,'rb').read()).hexdigest(),'training':{'n':len(train),'first_issue':train[0]['issue'],'last_issue':train[-1]['issue'],'pre_boundary_holdout':312,'iterations':iteration,'normalized_dual_gap':gap,'weights':w,'experts':['frequency','ewma','overdue','hot','anti_repeat','markov','uniform']},'windows':result,'automatic_adoption':False,'note':'Same two-blue comparison on held-out chronological rows, already inspected history; not independent prospective evidence. This weight fit is not deployed as the operational joint-ticket model.'}
json.dump(report,sys.stdout,ensure_ascii=False,indent=2);print()
