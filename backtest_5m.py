#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 5 DAKİKA ÇIKIŞ TESTİ
# Giriş: kripto_tarama.py sinyalleri (15m). Çıkış modelleri:
#   A) 15m BAND   : karşı 15m Bollinger bandı (vurkaç). Referans.
#   B) 5m  MIDBB  : 5m fiyat 5m orta bandı sana karşı kapatınca çık (Kemal'in fikri).
#   C) 15m MIDBB  : aynı kural 15m'de. B vs C = saf zaman dilimi etkisi.
# Hepsinde 15m yapısal swing stop = sert taban. timeout.
# Çalıştır:  python backtest_5m.py

import ccxt, time, csv, os, numpy as np

# ========================== AYARLAR ==========================
N_SYMBOLS = 60     # 5m veri ağır → ilk koşu 60. İstersen 103'e çıkar (uzun sürer).
DAYS      = 90     # ilk koşu 90 gün. 120'ye çıkarabilirsin.
MAXBARS_15 = 192   # 15m modeller için tavan (48 saat)
MAXBARS_5  = 576   # 5m model için tavan (48 saat = 576×5dk)
FEE       = 0.0005

INTERVAL="15m"; PERIOD=10; MULT=3.0
LOOKBACK=3; RVOLW=20; MINVOL=20_000_000
BBP=20; BBK=2.0; MAXEXT=0.03; RVOLMIN=1.5; SWINGK=6
RETEST_TOL=0.25; TOUCHLB=10; REV_RVOL=1.5

# ========== kripto_tarama.py'den BİREBİR ==========
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

# ========================== VERİ ==========================
def fetch_tf(ex, sym, days, tf, tf_ms):
    since=ex.milliseconds()-days*24*60*60*1000; out=[]
    while True:
        try: batch=ex.fetch_ohlcv(sym, tf, since=since, limit=1500)
        except Exception as e: print(f"    ! {sym} {tf} hata: {e}"); break
        if not batch: break
        out+=batch
        if len(batch)<1500: break
        since=batch[-1][0]+tf_ms; time.sleep(ex.rateLimit/1000)
    seen=set(); clean=[]
    for r in out:
        if r[0] not in seen: seen.add(r[0]); clean.append(r)
    clean.sort(key=lambda x:x[0]); return clean

# ========================== ÇIKIŞ SİMÜLATÖRLERİ ==========================
def sim_15(i, yon, entry, stop, tp_band, MB, H, L, C):
    """A=band tp, C=15m orta bant kırılımı. stop sert taban. Döner {m:(price,bar)}."""
    n=len(C); last=min(n-1,i+MAXBARS_15); res={}; pend={"A","C"}
    for j in range(i+1,last+1):
        hi=H[j]; lo=L[j]; cl=C[j]; mb=MB[j]
        if (lo<=stop) if yon=="AL" else (hi>=stop):
            for m in list(pend): res[m]=(stop, j-i)
            pend.clear(); break
        if "A" in pend and ((hi>=tp_band) if yon=="AL" else (lo<=tp_band)):
            res["A"]=(tp_band, j-i); pend.discard("A")
        if "C" in pend and not np.isnan(mb) and ((cl<mb) if yon=="AL" else (cl>mb)):
            res["C"]=(cl, j-i); pend.discard("C")
        if not pend: break
    for m in pend: res[m]=(C[last], last-i)
    return res

def sim_5(entry_ms, yon, entry, stop, TS5, H5, L5, C5, MB5):
    """B=5m orta bant kırılımı. Döner (exit_px, bars5) veya None."""
    j0=int(np.searchsorted(TS5, entry_ms))
    if j0>=len(C5)-2: return None
    last=min(len(C5)-1, j0+MAXBARS_5)
    for j in range(j0, last+1):
        hi=H5[j]; lo=L5[j]; cl=C5[j]; mb=MB5[j]
        if (lo<=stop) if yon=="AL" else (hi>=stop):
            return (stop, j-j0)
        if not np.isnan(mb) and ((cl<mb) if yon=="AL" else (cl>mb)):
            return (cl, j-j0)
    return (C5[last], last-j0)

def Rv(yon, entry, exitpx, risk):
    if risk<=0 or exitpx is None: return None
    pnl=(exitpx-entry) if yon=="AL" else (entry-exitpx)
    return (pnl-2*FEE*entry)/risk

def pct(yon, entry, exitpx):
    return 100*((exitpx-entry) if yon=="AL" else (entry-exitpx))/entry

# ========================== ANA ==========================
def run():
    ex=ccxt.binanceusdm({"enableRateLimit":True,"timeout":30000}); ex.load_markets()
    perp=[m["symbol"] for m in ex.markets.values()
          if m.get("swap") and m.get("linear") and m.get("quote")=="USDT" and m.get("active")]
    t=ex.fetch_tickers(perp)
    rows=sorted([(s,t[s].get("quoteVolume") or 0) for s in perp
                 if s in t and (t[s].get("quoteVolume") or 0)>=MINVOL],
                key=lambda x:x[1],reverse=True)[:N_SYMBOLS]
    syms=[s for s,_ in rows]
    print(f"Evren: {len(syms)} coin · {DAYS} gün · giriş 15m / çıkış 5m\n")

    trades=[]; need=max(PERIOD,BBP)+LOOKBACK+3
    for idx,sym in enumerate(syms,1):
        d15=fetch_tf(ex,sym,DAYS,"15m",15*60*1000)
        if len(d15)<need+MAXBARS_15+50:
            print(f"[{idx}/{len(syms)}] {sym}: 15m yetersiz"); continue
        d5=fetch_tf(ex,sym,DAYS,"5m",5*60*1000)
        if len(d5)<BBP+50:
            print(f"[{idx}/{len(syms)}] {sym}: 5m yetersiz"); continue
        a=np.array(d15,dtype=float)
        TS=a[:,0]; H=a[:,2]; L=a[:,3]; C=a[:,4]; V=a[:,5]
        ST,D=supertrend(H,L,C); MB,UP,DN=bollinger(C)
        b=np.array(d5,dtype=float)
        TS5=b[:,0]; H5=b[:,2]; L5=b[:,3]; C5=b[:,4]; MB5,_,_=bollinger(C5)
        seen=set(); open_until=-1; cs=0
        for i in range(need, len(C)-1):
            if i<=open_until: continue
            sl=slice(0,i+1)
            tip,yon,rv,trig=sinyal_bul(C[sl],ST[sl],D[sl],MB[sl],UP[sl],DN[sl],V[sl],TS[sl],H[sl],L[sl])
            if tip in (None,"TREND"): continue
            key=(tip,trig)
            if key in seen: continue
            seen.add(key)
            px=float(C[i]); stop,tp,rr=rr_hesapla(yon,px,H[sl],L[sl],UP[sl],DN[sl])
            risk=abs(px-stop)
            if rr<=0 or risk<=0: continue
            r15=sim_15(i,yon,px,stop,tp,MB,H,L,C)
            entry_ms=int(TS[i])+15*60*1000
            ex5=sim_5(entry_ms,yon,px,stop,TS5,H5,L5,C5,MB5)
            if ex5 is None: continue   # 5m verisi kapsamıyorsa adil olsun diye trade'i atla
            px_b,bars5=ex5
            px_a,bar_a=r15["A"]; px_c,bar_c=r15["C"]
            row=dict(sym=sym,tip=tip,yon=yon,rr=round(rr,3))
            row["R_A"]=round(Rv(yon,px,px_a,risk),4); row["pct_A"]=round(pct(yon,px,px_a),3)
            row["R_C"]=round(Rv(yon,px,px_c,risk),4); row["pct_C"]=round(pct(yon,px,px_c),3)
            row["R_B"]=round(Rv(yon,px,px_b,risk),4); row["pct_B"]=round(pct(yon,px,px_b),3)
            trades.append(row); cs+=1
            # kilidi SADECE referans modele (A=15m BAND, en kısa tutan) bağla.
            # B ve C paralel ölçülür; uzun tutuşları yeni giriş bulmayı engellemesin.
            open_until=i+bar_a
        print(f"[{idx}/{len(syms)}] {sym}: {cs} giriş")

    if not trades: print("\nHiç giriş yok."); return
    out=os.path.expanduser("~/bt_5m_trades.csv")
    with open(out,"w",newline="") as fc:
        w=csv.DictWriter(fc,fieldnames=list(trades[0].keys())); w.writeheader()
        for tr in trades: w.writerow(tr)

    def stat(rows,m):
        n=len(rows)
        if n==0: return None
        Rs=[r[f"R_{m}"] for r in rows]; ps=[r[f"pct_{m}"] for r in rows]
        return dict(n=n,win=100*sum(1 for x in Rs if x>0)/n,
                    expR=sum(Rs)/n,totR=sum(Rs),avgpct=sum(ps)/n)
    def line(name,s):
        if not s: return f"{name:<14} -"
        return (f"{name:<14} n={s['n']:<4} win%={s['win']:5.1f} | beklentiR={s['expR']:+.3f} "
                f"toplamR={s['totR']:+7.1f} | ort%hareket={s['avgpct']:+5.2f}")
    MODELS=[("A","15m BAND (vurkaç)"),("B","5m MIDBB (senin fikrin)"),("C","15m MIDBB")]
    print("\n"+"="*80)
    print(f"5dk ÇIKIŞ TESTİ — {len(trades)} giriş · {DAYS}g · {len(syms)} coin · komisyon={FEE*100:.3f}%/yön")
    print("="*80)
    for code,name in MODELS:
        print(f"\n--- {name} ---")
        print(line("  TÜMÜ", stat(trades,code)))
        for tp_ in ("FLIP","RETEST","REV"):
            print(line(f"  {tp_}", stat([r for r in trades if r['tip']==tp_],code)))
    print("="*80)
    print(f"\nDetay: {out}\nÖzeti olduğu gibi Claude'a yapıştır.")

if __name__=="__main__":
    run()
