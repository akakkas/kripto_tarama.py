#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# TREND FİLTRE BACKTEST — 30m Trend × 15m Sinyal
# 30m SuperTrend yönü = giriş filtresi (hava)
# 15m sinyalleri = giriş (ama 30m D ile uyumlu olunca)
# 15m çıkış = vurkaç (band)

import ccxt, time, csv, os, numpy as np

N_SYMBOLS = 103
DAYS      = 120
FEE       = 0.0005

INTERVAL="15m"; PERIOD=10; MULT=3.0
LOOKBACK=3; RVOLW=20; MINVOL=20_000_000
BBP=20; BBK=2.0; MAXEXT=0.03; RVOLMIN=1.5; SWINGK=6
RETEST_TOL=0.25; TOUCHLB=10; REV_RVOL=1.5

def supertrend(h,l,c,period=PERIOD,mult=MULT):
    n=len(c); hl2=(h+l)/2.0
    tr=np.zeros(n); tr[0]=h[0]-l[0]
    for i in range(1,n):
        tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    atr=np.zeros(n)
    if n>=period:
        atr[period-1]=tr[:period].mean()
        for i in range(period,n):
            atr[i]=(atr[i-1]*(period-1)+tr[i])/period
    up=hl2+mult*atr; lo=hl2-mult*atr
    fu=up.copy(); fl=lo.copy()
    for i in range(1,n):
        fu[i]=up[i] if (up[i]<fu[i-1] or c[i-1]>fu[i-1]) else fu[i-1]
        fl[i]=lo[i] if (lo[i]>fl[i-1] or c[i-1]<fl[i-1]) else fl[i-1]
    st=np.zeros(n); d=np.ones(n,dtype=int)
    for i in range(1,n):
        if c[i]>fu[i-1]: d[i]=1
        elif c[i]<fl[i-1]: d[i]=-1
        else:
            d[i]=d[i-1]
            if d[i]==1 and fl[i]<fl[i-1]: fl[i]=fl[i-1]
            if d[i]==-1 and fu[i]>fu[i-1]: fu[i]=fu[i-1]
        st[i]=fl[i] if d[i]==1 else fu[i]
    return st,d

def bollinger(c,p=BBP,k=BBK):
    n=len(c); mb=np.full(n,np.nan); up=np.full(n,np.nan); dn=np.full(n,np.nan)
    for i in range(p-1,n):
        w=c[i-p+1:i+1]; m=w.mean(); s=w.std()
        mb[i]=m; up[i]=m+k*s; dn[i]=m-k*s
    return mb,up,dn

def rvol(v,w=RVOLW):
    if len(v)<w+1: return float("nan")
    o=v[-(w+1):-1].mean()
    return v[-1]/o if o>0 else float("nan")

def sinyal_bul(close,st,d,mb,up,dn,vol,ts,high,low,rev_rvol=REV_RVOL):
    n=len(close); rv=rvol(vol)
    yon=None; fi=None; end=max(1,n-LOOKBACK)
    for i in range(n-1,end-1,-1):
        if d[i]!=d[i-1]:
            yon="AL" if d[i]==1 else "SAT"; fi=i; break
    c=close[-1]; stv=st[-1]; upv=up[-1]; dnv=dn[-1]; mbv=mb[-1]
    if yon=="AL":
        ext=(c-stv)/stv if stv>0 else 9
        if c<upv and ext<=MAXEXT: return ("FLIP","AL",rv,int(ts[fi]))
        return (None,None,rv,0)
    if yon=="SAT":
        ext=(stv-c)/c if c>0 else 9
        if c>dnv and ext<=MAXEXT: return ("FLIP","SAT",rv,int(ts[fi]))
        return (None,None,rv,0)
    if not np.isnan(mbv) and not np.isnan(mb[-2]) and rv==rv:
        zone=(upv-mbv)*RETEST_TOL
        if d[-1]==1 and close[-2]<=mbv+zone and c>mbv+zone and c>close[-2] and rv>=RVOLMIN:
            return ("RETEST","AL",rv,int(ts[-1]))
        if d[-1]==-1 and close[-2]>=mbv-zone and c<mbv-zone and c<close[-2] and rv>=RVOLMIN:
            return ("RETEST","SAT",rv,int(ts[-1]))
    if not np.isnan(mbv) and not np.isnan(mb[-2]):
        if d[-1]==1 and (high[-TOUCHLB:]>=up[-TOUCHLB:]).any() and close[-2]>=mb[-2] and c<mbv:
            return ("REV" if rv>=rev_rvol else "TREND","SAT",rv,int(ts[-1]))
        if d[-1]==-1 and (low[-TOUCHLB:]<=dn[-TOUCHLB:]).any() and close[-2]<=mb[-2] and c>mbv:
            return ("REV" if rv>=rev_rvol else "TREND","AL",rv,int(ts[-1]))
    return (None,None,rv,0)

def rr_hesapla(yon,px,h,l,up,dn):
    if yon=="AL":
        stop=float(np.min(l[-SWINGK:])); tp=float(up[-1])
        risk=px-stop; rew=tp-px
    else:
        stop=float(np.max(h[-SWINGK:])); tp=float(dn[-1])
        risk=stop-px; rew=px-tp
    rr=rew/risk if risk>0 else 0.0
    return stop,tp,rr

def fetch_tf(ex,sym,days,tf,tf_ms):
    since=ex.milliseconds()-days*24*60*60*1000; out=[]
    while True:
        try: batch=ex.fetch_ohlcv(sym,tf,since=since,limit=1500)
        except Exception as e: print(f"    ! {sym} {tf} hata: {e}"); break
        if not batch: break
        out+=batch
        if len(batch)<1500: break
        since=batch[-1][0]+tf_ms; time.sleep(ex.rateLimit/1000)
    seen=set(); clean=[]
    for r in out:
        if r[0] not in seen: seen.add(r[0]); clean.append(r)
    clean.sort(key=lambda x:x[0]); return clean

def sim_15(i,yon,entry,stop,tp,MB,H,L,C):
    n=len(C); last=min(n-1,i+192); res={}; pend={"band"}
    for j in range(i+1,last+1):
        hi=H[j]; lo=L[j]; cl=C[j]
        if (lo<=stop) if yon=="AL" else (hi>=stop):
            res["band"]=stop; pend.clear(); break
        if "band" in pend:
            th=(hi>=tp) if yon=="AL" else (lo<=tp)
            if th: res["band"]=tp; pend.discard("band")
        if not pend: break
    for m in pend: res[m]=C[last]
    return res

def Rv(yon,entry,exitpx,risk):
    if risk<=0: return 0.0
    pnl=(exitpx-entry) if yon=="AL" else (entry-exitpx)
    return (pnl-2*FEE*entry)/risk

def pct(yon,entry,exitpx):
    return 100*((exitpx-entry) if yon=="AL" else (entry-exitpx))/entry

def run():
    ex=ccxt.binanceusdm({"enableRateLimit":True,"timeout":30000}); ex.load_markets()
    perp=[m["symbol"] for m in ex.markets.values()
          if m.get("swap") and m.get("linear") and m.get("quote")=="USDT" and m.get("active")]
    t=ex.fetch_tickers(perp)
    rows=sorted([(s,t[s].get("quoteVolume") or 0) for s in perp
                 if s in t and (t[s].get("quoteVolume") or 0)>=MINVOL],
                key=lambda x:x[1],reverse=True)[:N_SYMBOLS]
    syms=[s for s,_ in rows]
    print(f"Evren: {len(syms)} coin · {DAYS} gün · 30m TREND FILTER × 15m SİNYAL\n")

    trades=[]; need=max(PERIOD,BBP)+LOOKBACK+3
    for idx,sym in enumerate(syms,1):
        d30=fetch_tf(ex,sym,DAYS,"30m",30*60*1000)
        if len(d30)<need+100: print(f"[{idx}/{len(syms)}] {sym}: 30m yetersiz"); continue
        d15=fetch_tf(ex,sym,DAYS,"15m",15*60*1000)
        if len(d15)<need+100: print(f"[{idx}/{len(syms)}] {sym}: 15m yetersiz"); continue

        a=np.array(d30,dtype=float); H30=a[:,2]; L30=a[:,3]; C30=a[:,4]
        ST30,D30=supertrend(H30,L30,C30)

        b=np.array(d15,dtype=float); H15=b[:,2]; L15=b[:,3]; C15=b[:,4]; V15=b[:,5]; TS15=b[:,0]
        ST15,D15=supertrend(H15,L15,C15); MB15,UP15,DN15=bollinger(C15)

        idx_map={}
        j30=0
        for i15 in range(len(d15)):
            ts15=int(d15[i15][0])
            while j30+1<len(d30) and int(d30[j30+1][0])<=ts15: j30+=1
            if int(d30[j30][0])<=ts15<int(d30[j30][0])+30*60*1000:
                idx_map[i15]=(j30,D30[j30])

        seen=set(); open_until=-1; cnt=0
        for i15 in range(need,len(C15)-1):
            if i15<=open_until: continue
            if i15 not in idx_map: continue
            i30,d30_val=idx_map[i15]

            sl=slice(0,i15+1)
            tip,yon,rv,trig=sinyal_bul(C15[sl],ST15[sl],D15[sl],MB15[sl],UP15[sl],DN15[sl],
                                        V15[sl],TS15[sl],H15[sl],L15[sl])
            if tip in (None,"TREND"): continue

            if not ((yon=="AL" and d30_val==1) or (yon=="SAT" and d30_val==-1)):
                continue

            key=(tip,trig)
            if key in seen: continue
            seen.add(key)

            px=float(C15[i15]); stop,tp,rr=rr_hesapla(yon,px,H15[sl],L15[sl],UP15[sl],DN15[sl])
            risk=abs(px-stop)
            if rr<=0 or risk<=0: continue

            epx=sim_15(i15,yon,px,stop,tp,MB15,H15,L15,C15)["band"]
            trades.append(dict(sym=sym,tip=tip,yon=yon,d30=int(d30_val),
                               R=round(Rv(yon,px,epx,risk),4),pct=round(pct(yon,px,epx),3)))
            open_until=i15+10; cnt+=1
        print(f"[{idx}/{len(syms)}] {sym}: {cnt} giriş")

    if not trades: print("\nHiç giriş yok."); return
    out=os.path.expanduser("~/bt_tf_trades.csv")
    with open(out,"w",newline="") as fc:
        w=csv.DictWriter(fc,fieldnames=list(trades[0].keys())); w.writeheader()
        for tr in trades: w.writerow(tr)

    def stat(rows):
        n=len(rows)
        if n==0: return None
        Rs=[r["R"] for r in rows]
        return dict(n=n,win=100*sum(1 for x in Rs if x>0)/n,
                    expR=sum(Rs)/n,totR=sum(Rs))
    def line(name,s):
        if not s: return f"{name:<14} -"
        return (f"{name:<14} n={s['n']:<5} win%={s['win']:5.1f} | beklentiR={s['expR']:+.3f} "
                f"toplamR={s['totR']:+7.1f}")

    print("\n"+"="*78)
    print(f"TREND FİLTRE TESTİ — {len(trades)} giriş · {DAYS}g · {len(syms)} coin")
    print("30m SuperTrend yönü = FILTER, 15m sinyalleri = GİRİŞ (uyumlu olunca)")
    print("="*78)
    print(line("  TÜMÜ", stat(trades)))
    for tp_ in ("FLIP","RETEST","REV"):
        print(line(f"  {tp_}", stat([r for r in trades if r["tip"]==tp_])))
    print("="*78)
    print(f"\nDetay: {out}\nÖzeti olduğu gibi Claude'a yapıştır.")

if __name__=="__main__":
    run()
