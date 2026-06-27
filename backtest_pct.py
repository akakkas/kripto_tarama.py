#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SABİT % HEDEF TESTİ — "reel getiri" çıkışı
# Giriş: BB kırılımı (close üst bandı yukarı keser → LONG / alt bandı aşağı → SHORT), 15m ve 30m.
# Çıkış: sabit % hedef (3/5/8) VS stop (yapısal swing / sabit -%2). 12 hücre yan yana.
# 5m veri YOK — sabit % giriş zaman diliminde ölçülür. Direkt 103/120 koşulabilir.
# Çalıştır:  python backtest_pct.py

import ccxt, time, csv, os, numpy as np

N_SYMBOLS = 103
DAYS      = 120
HOLD_DAYS = 5          # bir trade en fazla kaç gün açık (büyük harekete yer ver)
FEE       = 0.0005
BBP=20; BBK=2.0; SWINGK=6
TFS=[("15m",15),("30m",30)]
STOPS=[("swing",None),("fix2",0.02)]
TARGETS=[0.03,0.05,0.08]

def bollinger(c,p=BBP,k=BBK):
    n=len(c); mb=np.full(n,np.nan); up=np.full(n,np.nan); dn=np.full(n,np.nan)
    for i in range(p-1,n):
        w=c[i-p+1:i+1]; m=w.mean(); s=w.std()
        mb[i]=m; up[i]=m+k*s; dn[i]=m-k*s
    return mb,up,dn

def brk_entry(C,UP,DN,i):
    c=C[i]; cp=C[i-1]; up=UP[i]; dn=DN[i]; upp=UP[i-1]; dnp=DN[i-1]
    if np.isnan(up): return None
    if cp<=upp and c>up: return "AL"
    if cp>=dnp and c<dn: return "SAT"
    return None

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

def simulate(i,yon,entry,stop_px,tp_px,H,L,C,maxbars):
    n=len(C); last=min(n-1,i+maxbars)
    for j in range(i+1,last+1):
        hi=H[j]; lo=L[j]
        if yon=="AL":
            hs=lo<=stop_px; ht=hi>=tp_px
            if hs: return ("stop",stop_px,j-i)
            if ht: return ("tp",tp_px,j-i)
        else:
            hs=hi>=stop_px; ht=lo<=tp_px
            if hs: return ("stop",stop_px,j-i)
            if ht: return ("tp",tp_px,j-i)
    return ("timeout",C[last],last-i)

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
                 if s in t and (t[s].get("quoteVolume") or 0)>=20_000_000],
                key=lambda x:x[1],reverse=True)[:N_SYMBOLS]
    syms=[s for s,_ in rows]
    print(f"Evren: {len(syms)} coin · {DAYS} gün · BB kırılım girişi · sabit % hedef\n")

    # trades[(tf,stopname,target)] = list of dict
    cells={}
    for tfname,_ in TFS:
        for sname,_ in STOPS:
            for tg in TARGETS:
                cells[(tfname,sname,tg)]=[]

    for idx,sym in enumerate(syms,1):
        line=[]
        for tfname,tfmin in TFS:
            data=fetch_tf(ex,sym,DAYS,tfname,tfmin*60*1000)
            if len(data)<BBP+50: line.append(f"{tfname}:yetersiz"); continue
            a=np.array(data,dtype=float); H=a[:,2]; L=a[:,3]; C=a[:,4]
            UP=bollinger(C)[1]; DN=bollinger(C)[2]
            maxbars=int(HOLD_DAYS*24*60/tfmin)
            cnt=0; open_until=-1
            for i in range(BBP+2,len(C)-1):
                if i<=open_until: continue
                yon=brk_entry(C,UP,DN,i)
                if yon is None: continue
                px=float(C[i])
                swing=float(np.min(L[max(0,i-SWINGK+1):i+1])) if yon=="AL" else float(np.max(H[max(0,i-SWINGK+1):i+1]))
                ref_bar=None
                for sname,sval in STOPS:
                    if sname=="swing": stop_px=swing
                    else: stop_px=px*(1-sval) if yon=="AL" else px*(1+sval)
                    risk=abs(px-stop_px)
                    if risk<=0: continue
                    for tg in TARGETS:
                        tp_px=px*(1+tg) if yon=="AL" else px*(1-tg)
                        res,epx,bars=simulate(i,yon,px,stop_px,tp_px,H,L,C,maxbars)
                        cells[(tfname,sname,tg)].append(dict(sym=sym,yon=yon,
                            R=round(Rv(yon,px,epx,risk),4),pctret=round(pct(yon,px,epx),3),
                            res=res,bars=bars))
                        if sname=="swing" and tg==0.03: ref_bar=bars
                cnt+=1
                open_until=i+(ref_bar if ref_bar else 3)
            line.append(f"{tfname}:{cnt}")
        print(f"[{idx}/{len(syms)}] {sym}: "+" ".join(line))

    # ---- ÖZET ----
    def stat(rows):
        n=len(rows)
        if n==0: return None
        Rs=[r["R"] for r in rows]
        wins=sum(1 for r in rows if r["res"]=="tp")
        tmo=sum(1 for r in rows if r["res"]=="timeout")
        return dict(n=n,win=100*wins/n,tmo=100*tmo/n,expR=sum(Rs)/n,totR=sum(Rs))
    out=os.path.expanduser("~/bt_pct_trades.csv")
    with open(out,"w",newline="") as fc:
        w=csv.writer(fc); w.writerow(["tf","stop","target","sym","yon","R","pctret","res","bars"])
        for (tf,sn,tg),rs in cells.items():
            for r in rs: w.writerow([tf,sn,tg,r["sym"],r["yon"],r["R"],r["pctret"],r["res"],r["bars"]])

    print("\n"+"="*84)
    print(f"SABİT % HEDEF TESTİ — BB kırılım · {DAYS}g · {len(syms)} coin · komisyon={FEE*100:.3f}%/yön")
    print("="*84)
    print(f"{'TF':>4} {'stop':>6} {'hedef':>6} {'n':>5} {'win%':>6} {'tmo%':>6} {'beklentiR':>10} {'toplamR':>9}")
    print("-"*84)
    for tfname,_ in TFS:
        for sname,_ in STOPS:
            for tg in TARGETS:
                s=stat(cells[(tfname,sname,tg)])
                if not s:
                    print(f"{tfname:>4} {sname:>6} {int(tg*100):>5}% -"); continue
                print(f"{tfname:>4} {sname:>6} {int(tg*100):>5}% {s['n']:>5} {s['win']:>5.1f} {s['tmo']:>5.1f} "
                      f"{s['expR']:>+9.3f} {s['totR']:>+8.1f}")
        print("-"*84)
    print(f"Detay: {out}\nÖzeti olduğu gibi Claude'a yapıştır.")

if __name__=="__main__":
    run()
