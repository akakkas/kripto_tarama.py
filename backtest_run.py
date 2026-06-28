#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# KAZANANI SÜR — 15m RETEST giriş / 1h orta bant çıkış
# Giriş: 15m RETEST (mid-band reclaim, rvol>=1.5)
# Çıkış: 1h orta bant ters kırılana kadar TUT (kazananı sür) + swing stop sert taban
# Karşılaştırma: aynı girişler, 15m vurkaç çıkışıyla (referans)
# Çalıştır:  python backtest_run.py

import ccxt, time, csv, os, numpy as np

N_SYMBOLS = 103
DAYS      = 120
FEE       = 0.0005
HOLD_DAYS = 7      # 1h çıkış için tavan (kazanana yer ver)

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
    if not np.isnan(mbv) and not np.isnan(mb[-2]):
        if d[-1]==1 and (high[-TOUCHLB:]>=up[-TOUCHLB:]).any() and close[-2]>=mb[-2] and c<mbv:
            return ("REV" if rv>=REV_RVOL else "TREND","SAT",rv,int(ts[-1]))
        if d[-1]==-1 and (low[-TOUCHLB:]<=dn[-TOUCHLB:]).any() and close[-2]<=mb[-2] and c>mbv:
            return ("REV" if rv>=REV_RVOL else "TREND","AL",rv,int(ts[-1]))
    return (None,None,rv,0)

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

# --- A) referans: 15m vurkaç (band) ---
def sim_band(i,yon,entry,stop,tp,H,L,C):
    n=len(C); last=min(n-1,i+192)
    for j in range(i+1,last+1):
        hi=H[j]; lo=L[j]
        if (lo<=stop) if yon=="AL" else (hi>=stop): return stop,j-i
        if (hi>=tp) if yon=="AL" else (lo<=tp): return tp,j-i
    return C[last],last-i

# --- B) kazananı sür: 1h orta bant ters kırılana kadar tut + swing stop ---
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
    print(f"Evren: {len(syms)} coin · {DAYS} gün · 15m RETEST giriş / 1h orta bant çıkış\n")

    trades=[]; need=max(PERIOD,BBP)+LOOKBACK+3
    maxbars_1h=int(HOLD_DAYS*24)
    for idx,sym in enumerate(syms,1):
        d15=fetch_tf(ex,sym,DAYS,"15m",15*60*1000)
        if len(d15)<need+100: print(f"[{idx}/{len(syms)}] {sym}: 15m yetersiz"); continue
        d1h=fetch_tf(ex,sym,DAYS,"1h",60*60*1000)
        if len(d1h)<BBP+20: print(f"[{idx}/{len(syms)}] {sym}: 1h yetersiz"); continue

        b=np.array(d15,dtype=float); H=b[:,2]; L=b[:,3]; C=b[:,4]; V=b[:,5]; TS=b[:,0]
        ST,D=supertrend(H,L,C); MB,UP,DN=bollinger(C)

        e=np.array(d1h,dtype=float); H1=e[:,2]; L1=e[:,3]; C1=e[:,4]; TS1=e[:,0]
        MB1=bollinger(C1)[0]

        seen=set(); open_until=-1; cnt=0
        for i in range(need,len(C)-1):
            if i<=open_until: continue
            sl=slice(0,i+1)
            tip,yon,rv,trig=sinyal_bul(C[sl],ST[sl],D[sl],MB[sl],UP[sl],DN[sl],V[sl],TS[sl],H[sl],L[sl])
            if tip!="RETEST": continue              # SADECE retest
            if trig in seen: continue
            seen.add(trig)
            px=float(C[i])
            stop=float(np.min(L[max(0,i-SWINGK+1):i+1])) if yon=="AL" else float(np.max(H[max(0,i-SWINGK+1):i+1]))
            risk=abs(px-stop)
            if risk<=0: continue
            # giriş anındaki 1h orta bant
            entry_ms=int(TS[i])+15*60*1000
            j1=int(np.searchsorted(TS1,entry_ms))
            if j1>=len(C1): continue
            mb1_now=MB1[min(j1,len(MB1)-1)]
            if np.isnan(mb1_now): continue
            # 1h TEYİT: fiyat 1h orta bandın doğru tarafında mı?
            confirmed = (px>mb1_now) if yon=="AL" else (px<mb1_now)
            # çıkış: 1h orta bant kırılana kadar sür (her iki gruba da aynı)
            rb=sim_1h(entry_ms,yon,px,stop,TS1,H1,L1,C1,MB1,maxbars_1h)
            if rb is None: continue
            pb,bars1=rb
            trades.append(dict(sym=sym,yon=yon,confirmed=int(confirmed),
                R=round(Rv(yon,px,pb,risk),4), pct=round(pct(yon,px,pb),3),
                bars1=bars1))
            open_until=i+5; cnt+=1
        print(f"[{idx}/{len(syms)}] {sym}: {cnt} giriş")

    if not trades: print("\nHiç giriş yok."); return
    out=os.path.expanduser("~/bt_run_trades.csv")
    with open(out,"w",newline="") as fc:
        w=csv.DictWriter(fc,fieldnames=list(trades[0].keys())); w.writeheader()
        for tr in trades: w.writerow(tr)

    def stat(rows):
        n=len(rows)
        if n==0: return None
        Rs=[r["R"] for r in rows]
        wins=[x for x in Rs if x>0]
        return dict(n=n,win=100*len(wins)/n,
                    expR=sum(Rs)/n,totR=sum(Rs),
                    avgpct=sum(r["pct"] for r in rows)/n,
                    avgbars=sum(r["bars1"] for r in rows)/n,
                    avgwin=(sum(wins)/len(wins)) if wins else 0,
                    avglos=(sum(x for x in Rs if x<=0)/max(1,n-len(wins))))
    def line(name,s):
        if not s: return f"{name:<30} -"
        return (f"{name:<30} n={s['n']:<5} win%={s['win']:5.1f} | bekR={s['expR']:+.3f} "
                f"topR={s['totR']:+7.1f} | ort%={s['avgpct']:+.2f} tut={s['avgbars']:.1f}s")

    conf=[r for r in trades if r["confirmed"]==1]      # 1h-uyumlu (trend-following)
    counter=[r for r in trades if r["confirmed"]==0]   # 1h-karşı (mean-reversion) ← YENİ TEZ

    print("\n"+"="*94)
    print(f"BAYRAK ÇEVRİLDİ: KARŞI-TREND RETEST (mean-reversion) · {DAYS}g · {len(syms)} coin")
    print("="*94)
    print(line("TÜMÜ", stat(trades)))
    print(line(">>> KARŞI-TREND (yeni tez)", stat(counter)))
    print(line("    uyumlu (trend-follow)", stat(conf)))
    print("-"*94)
    print("  --- karşı-trend, yöne göre ---")
    print(line("    karşı-trend AL", stat([r for r in counter if r["yon"]=="AL"])))
    print(line("    karşı-trend SAT", stat([r for r in counter if r["yon"]=="SAT"])))
    print("-"*94)
    cs=stat(counter)
    if cs and cs["n"]>0:
        print(f"  karşı-trend kazanan ort: {cs['avgwin']:+.2f}R | kaybeden ort: {cs['avglos']:+.2f}R")
        # bootstrap: karşı-trend pozitif mi yoksa şans mı?
        import random
        Rs=[r["R"] for r in counter]; n=len(Rs)
        boots=[]
        for _ in range(2000):
            s=sum(random.choice(Rs) for _ in range(n))/n
            boots.append(s)
        boots.sort()
        lo=boots[int(0.025*len(boots))]; hi=boots[int(0.975*len(boots))]
        print(f"  bootstrap %95 güven aralığı: [{lo:+.3f}, {hi:+.3f}]  (n={n})")
        if lo>0: print("  → alt sınır >0: dağıtım pozitif, ŞANS DEĞİL gibi ✅")
        else:    print("  → alt sınır <0: sıfırı kapsıyor, henüz şanstan ayrışmadı ⚠️")
    print("="*94)
    print(f"\nDetay: {out}\nÖzeti olduğu gibi Claude'a yapıştır.")

if __name__=="__main__":
    run()
