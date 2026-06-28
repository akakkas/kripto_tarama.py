#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# OUT-OF-SAMPLE TESTİ — karşı-trend RETEST (+0.43'ün gerçek mi şans mı sınavı)
# KURAL DONDURULDU: 15m RETEST/AL, 1h DÜŞÜŞ trendinde (karşı-trend) → gir, 1h orta bant çıkışı, swing stop.
# İki pencere: IN-SAMPLE (0-120g, +0.43'ün yeri) vs OUT-OF-SAMPLE (120-240g, taze veri).
# Aynı kural iki pencerede. OOS'ta da pozitif+bootstrap düzgün → gerçek. Sönerse → gürültü.
# Çalıştır:  python backtest_oos.py

import ccxt, time, csv, os, random, numpy as np

N_SYMBOLS = 103
WIN_DAYS  = 120          # her pencere uzunluğu
FEE       = 0.0005
HOLD_1H   = 168          # 1h bar tavanı (7 gün)

PERIOD=10; MULT=3.0; LOOKBACK=3; RVOLW=20; MINVOL=20_000_000
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

def sinyal_bul(close,st,d,mb,up,dn,vol,ts,high,low):
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
    return (None,None,rv,0)

def fetch_range(ex,sym,start_ms,end_ms,tf,tf_ms):
    since=start_ms; out=[]
    while since<end_ms:
        try: batch=ex.fetch_ohlcv(sym,tf,since=since,limit=1500)
        except Exception as e: print(f"    ! {sym} {tf} hata: {e}"); break
        if not batch: break
        out+=batch
        if len(batch)<1500: break
        since=batch[-1][0]+tf_ms; time.sleep(ex.rateLimit/1000)
    seen=set(); clean=[]
    for r in out:
        if r[0] not in seen and r[0]<end_ms: seen.add(r[0]); clean.append(r)
    clean.sort(key=lambda x:x[0]); return clean

def sim_1h(entry_ms,yon,entry,stop,TS1,H1,L1,C1,MB1,maxbars):
    j0=int(np.searchsorted(TS1,entry_ms))
    if j0>=len(C1)-1: return None
    last=min(len(C1)-1,j0+maxbars)
    for j in range(j0,last+1):
        hi=H1[j]; lo=L1[j]; cl=C1[j]; mb=MB1[j]
        if (lo<=stop) if yon=="AL" else (hi>=stop): return stop,j-j0
        if not np.isnan(mb) and ((cl<mb) if yon=="AL" else (cl>mb)): return cl,j-j0
    return C1[last],last-j0

def Rv(yon,entry,exitpx,risk):
    if risk<=0: return 0.0
    pnl=(exitpx-entry) if yon=="AL" else (entry-exitpx)
    return (pnl-2*FEE*entry)/risk

def collect(ex,syms,start_ms,end_ms,need):
    """Verilen pencerede karşı-trend RETEST trade'lerini topla."""
    trades=[]
    for sym in syms:
        d15=fetch_range(ex,sym,start_ms,end_ms,"15m",15*60*1000)
        if len(d15)<need+100: continue
        d1h=fetch_range(ex,sym,start_ms,end_ms,"1h",60*60*1000)
        if len(d1h)<BBP+20: continue
        b=np.array(d15,dtype=float); H=b[:,2]; L=b[:,3]; C=b[:,4]; V=b[:,5]; TS=b[:,0]
        ST,D=supertrend(H,L,C); MB,UP,DN=bollinger(C)
        e=np.array(d1h,dtype=float); H1=e[:,2]; L1=e[:,3]; C1=e[:,4]; TS1=e[:,0]
        MB1=bollinger(C1)[0]
        seen=set(); open_until=-1
        for i in range(need,len(C)-1):
            if i<=open_until: continue
            sl=slice(0,i+1)
            tip,yon,rv,trig=sinyal_bul(C[sl],ST[sl],D[sl],MB[sl],UP[sl],DN[sl],V[sl],TS[sl],H[sl],L[sl])
            if tip!="RETEST": continue
            if trig in seen: continue
            seen.add(trig)
            px=float(C[i])
            stop=float(np.min(L[max(0,i-SWINGK+1):i+1])) if yon=="AL" else float(np.max(H[max(0,i-SWINGK+1):i+1]))
            risk=abs(px-stop)
            if risk<=0: continue
            entry_ms=int(TS[i])+15*60*1000
            j1=int(np.searchsorted(TS1,entry_ms))
            if j1>=len(C1): continue
            mb1=MB1[min(j1,len(MB1)-1)]
            if np.isnan(mb1): continue
            confirmed=(px>mb1) if yon=="AL" else (px<mb1)
            if confirmed: continue          # SADECE karşı-trend (1h'e karşı)
            rb=sim_1h(entry_ms,yon,px,stop,TS1,H1,L1,C1,MB1,HOLD_1H)
            if rb is None: continue
            pb,bars1=rb
            trades.append(dict(sym=sym,yon=yon,R=round(Rv(yon,px,pb,risk),4)))
            open_until=i+5
    return trades

def report(name,trades):
    n=len(trades)
    print(f"\n--- {name} ---")
    if n==0: print("  (giriş yok)"); return
    Rs=[r["R"] for r in trades]
    al=[r for r in trades if r["yon"]=="AL"]; sat=[r for r in trades if r["yon"]=="SAT"]
    win=100*sum(1 for x in Rs if x>0)/n
    print(f"  TÜMÜ      n={n:<4} win%={win:4.1f}  beklentiR={sum(Rs)/n:+.3f}  toplamR={sum(Rs):+.1f}")
    if al:
        ra=[r["R"] for r in al]; print(f"  AL        n={len(al):<4} win%={100*sum(1 for x in ra if x>0)/len(al):4.1f}  beklentiR={sum(ra)/len(ra):+.3f}")
    if sat:
        rs=[r["R"] for r in sat]; print(f"  SAT       n={len(sat):<4} win%={100*sum(1 for x in rs if x>0)/len(sat):4.1f}  beklentiR={sum(rs)/len(rs):+.3f}")
    if n>=10:
        random.seed(42); boots=[]
        for _ in range(5000): boots.append(sum(random.choice(Rs) for _ in range(n))/n)
        boots.sort(); lo=boots[125]; hi=boots[-125]
        verdict = "✅ ALT SINIR>0, ŞANS DEĞİL" if lo>0 else ("⚠️ pozitif ama 0'ı kapsıyor" if sum(Rs)/n>0 else "❌ negatif")
        print(f"  bootstrap %95 CI: [{lo:+.3f}, {hi:+.3f}]  {verdict}")

def run():
    ex=ccxt.binanceusdm({"enableRateLimit":True,"timeout":30000}); ex.load_markets()
    perp=[m["symbol"] for m in ex.markets.values()
          if m.get("swap") and m.get("linear") and m.get("quote")=="USDT" and m.get("active")]
    t=ex.fetch_tickers(perp)
    rows=sorted([(s,t[s].get("quoteVolume") or 0) for s in perp
                 if s in t and (t[s].get("quoteVolume") or 0)>=MINVOL],
                key=lambda x:x[1],reverse=True)[:N_SYMBOLS]
    syms=[s for s,_ in rows]
    now=ex.milliseconds(); day=24*60*60*1000
    is_start=now-WIN_DAYS*day;        is_end=now
    oos_start=now-2*WIN_DAYS*day;     oos_end=now-WIN_DAYS*day
    need=max(PERIOD,BBP)+LOOKBACK+3

    print(f"Evren: {len(syms)} coin · karşı-trend RETEST · DONDURULMUŞ KURAL")
    print(f"IN-SAMPLE:  son {WIN_DAYS}g  |  OUT-OF-SAMPLE: {WIN_DAYS}-{2*WIN_DAYS}g öncesi\n")

    print("IN-SAMPLE veri çekiliyor...")
    is_tr=collect(ex,syms,is_start,is_end,need)
    print("OUT-OF-SAMPLE veri çekiliyor...")
    oos_tr=collect(ex,syms,oos_start,oos_end,need)

    print("\n"+"="*78)
    print("OUT-OF-SAMPLE SINAVI — karşı-trend RETEST (+0.43 gerçek mi?)")
    print("="*78)
    report("IN-SAMPLE (referans, +0.43'ün yeri)", is_tr)
    report("OUT-OF-SAMPLE (taze veri — GERÇEK SINAV)", oos_tr)
    print("\n"+"="*78)
    print("YORUM: OOS'ta AL pozitif + bootstrap alt sınır>0 ise → GERÇEK. Aksi → gürültüydü.")
    print("="*78)
    out=os.path.expanduser("~/bt_oos_trades.csv")
    with open(out,"w",newline="") as fc:
        w=csv.writer(fc); w.writerow(["pencere","sym","yon","R"])
        for r in is_tr: w.writerow(["IS",r["sym"],r["yon"],r["R"]])
        for r in oos_tr: w.writerow(["OOS",r["sym"],r["yon"],r["R"]])
    print(f"\nDetay: {out}\nÖzeti olduğu gibi Claude'a yapıştır.")

if __name__=="__main__":
    run()
