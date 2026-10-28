# Compare fixed blue candidates on pre-draw features from blue_review --features.
# This is exploratory research; no window winner is automatically promoted.
import json,math,collections
import sys
raw_report=json.load(open(sys.argv[1]));raw=raw_report['features'];rows=raw['rows'];draws=raw['draws'];rank=lambda s:sorted(range(16),key=lambda i:(-math.floor(s[i]*1e9+0.5),i))
models=['weighted','equal']+['expert'+str(i)for i in range(6)]+['deduplicated','frequency_markov','transition_shrink64','recent156','gap_hazard','hit2_frozen','rank_frozen','rank_rolling312']
results={m:[]for m in models};hit_sums=raw['warmup_hits'].copy();nfit=raw['warmup_periods'];position_count=[0]*16;position_n=0;position_history=collections.deque();past=collections.deque();counts=[0]*16;trans=[[0]*16 for _ in range(16)];prev=None;last=[-1]*16;binhistory=collections.deque();op=[0]*10;hit=[0]*10;byissue={r['issue']:r for r in rows}
def bucket(g):
 for i,limit in enumerate([0,1,2,4,7,11,15,23,31]):
  if g<=limit:return i
 return 9
for t,d in enumerate(draws):
 issue=d['issue'];winner=d['blue'];r=byissue.get(issue);bins=[bucket(t-last[i]-1)for i in range(16)]
 if r:
  experts=r['p'];orders=[rank(s)for s in experts];base=rank(r['baseline']);equal=[sum(x[i]for x in experts)/6 for i in range(16)]
  score={'weighted':r['baseline'],'equal':equal,**{'expert'+str(e):experts[e]for e in range(6)}}
  score['deduplicated']=[(experts[0][i]+experts[4][i])/8+(experts[1][i]+experts[3][i])/8+experts[2][i]/4+experts[5][i]/4 for i in range(16)]
  score['frequency_markov']=[(experts[0][i]+experts[5][i])/2 for i in range(16)]
  score['transition_shrink64']=[((trans[prev][i]if prev is not None else 0)+64*(counts[i]+1)/(t+16))/((sum(trans[prev])if prev is not None else 0)+64)for i in range(16)]
  score['recent156']=[1+sum(v==i for v in past)for i in range(16)]
  score['gap_hazard']=[(hit[bins[i]]+256/16)/(op[bins[i]]+256)for i in range(16)]
  weights=[math.exp(16*v/max(1,nfit))for v in hit_sums];total=sum(weights);score['hit2_frozen']=[sum(weights[e]*experts[e][i]for e in range(6))/total for i in range(16)]
  score['rank_frozen']=[0.]*16;score['rank_rolling312']=[0.]*16;recent=collections.Counter(position_history)
  for pos,i in enumerate(base):
   score['rank_frozen'][i]=(position_count[pos]+156/16)/(position_n+156)
   score['rank_rolling312'][i]=(recent[pos]+156/16)/(len(position_history)+156)
  for m in models:
   if m.startswith('rank_'):order=sorted(base,key=lambda i:-round(score[m][i]*1e9))
   else:order=rank(score[m])
   results[m].append(order.index(winner)+1)
  if issue<25111:
   for e in range(6):hit_sums[e]+=winner in orders[e][:2]
   nfit+=1;position_count[base.index(winner)]+=1;position_n+=1
  position_history.append(base.index(winner))
  if len(position_history)>312:position_history.popleft()
 if len(past)>=156:past.popleft()
 past.append(winner)
 opportunities=collections.Counter(bins);outcome=bins[winner];binhistory.append((opportunities,outcome))
 for b,v in opportunities.items():op[b]+=v
 hit[outcome]+=1
 if len(binhistory)>312:
  old,b=binhistory.popleft()
  for k,v in old.items():op[k]-=v
  hit[b]-=1
 counts[winner]+=1
 if prev is not None:trans[prev][winner]+=1
 last[winner]=t;prev=winner
split=next(i for i,r in enumerate(rows)if r['issue']==25111)
windows={'development':(0,split),'confirmation_exploratory':(split,split+156),'tracking':(split+156,len(rows)),'recent30':(len(rows)-30,len(rows)),'recent60':(len(rows)-60,len(rows)),'recent156':(len(rows)-156,len(rows)),'all':(0,len(rows))}
report={m:{w:{'n':b-a,'hits':[sum(v<=k for v in r[a:b])for k in [1,2,3,4,5,6]]}for w,(a,b)in windows.items()}for m,r in results.items()}
json.dump({'kind':'exploratory_blue_candidates','actual_recommendation':False,'automatic_promotion':False,'input_data_sha256':raw_report['data_sha256'],'source_sha256':__import__('hashlib').sha256(open(__file__,'rb').read()).hexdigest(),'models':models,'results':report,'ranks':results,'windows':windows},sys.stdout,indent=2)
print(file=sys.stdout)
