#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SAF BOLLINGER TESTİ — SuperTrend YOK
# Giriş: 15m Bollinger (3 ayrı strateji yan yana). Çıkış: 5m orta bant kırılımı.
#   MRREV : alt bant altına sark+geri kapat → LONG ; üst bant üstüne çık+geri kapat → SHORT  (fade)
#   BRK   : üst bandı yukarı kapat → LONG ; alt bandı aşağı kapat → SHORT  (kırılım/momentum)
#   MIDX  : orta bandı yukarı kes → LONG ; aşağı kes → SHORT  (orta bant geçişi)
# Stop: son SWINGK swing low/high (sert taban). timeout=MAXBARS_5.
# Çalıştır:  python backtest_bb.py

import ccxt, time, csv, os, numpy as np

# ========================== AYARLAR ==========================
N_SYMBOLS = 60
DAYS      = 90
MAXBARS_5 = 576      # 5m çıkış tavanı (48 saat)
FEE       = 0.0005
BBP=20; BBK=2.0; SWINGK=6
INTERVAL="15m"

def bollinger(c,p=BBP,k=BBK):
    n=len(c); mb=np.full(n,np.nan); up=np.full(n,np.nan); dn=np.full(n,np.nan)
    for i in range(p-1,n):
        w=c[i-p+1:i+1]; m=w.mean(); s=w.std()
        mb[i]=m; up[i]=m+k*s; dn[i]=m-k*s
    return mb,up,dn

def bb_entry(C, MB, UP, DN, i, variant):
    """i = son kapalı bar. Döner 'AL'/'SAT'/None."""
    c=C[i]; cp=C[i-1]; mb=MB[i]; up=UP[i]; dn=DN[i]; upp=UP[i-1]; dnp=DN[i-1]; mbp=MB[i-1]
    if np.isnan(mb) or np.isnan(mbp): return None
    if variant=="MRREV":
        if cp<dnp and c>=dn: return "AL"
        if cp>upp and c<=up: return "SAT"
    elif variant=="BRK":
        if cp<=upp and c>up: return "AL"
        if cp>=dnp and c<dn: return "SAT"
    elif variant=="MIDX":
        if cp<=mbp and c>mb: return "AL"
        if cp>=mbp and c<mb: return "SAT"
    return None

def stop_hesapla(yon,h,l):
    return float(np.min(l[-SWINGK:])) if yon=="AL" else float(np.max(h[-SWINGK:]))

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

def sim_5(entry_ms, yon, stop, TS5, H5, L5, C5, MB5):
    j0=int(np.searchsorted(TS5, entry_ms))
    if j0>=len(C5)-2: return None
    last=min(len(C5)-1, j0+MAXBARS_5)
    for j in range(j0, last+1):
        hi=H5[j]; lo=L5[j]; cl=C5[j]; mb=MB5[j]
        if (lo<=stop) if yon=="AL" else (hi>=stop): return (stop, j-j0)
        if not np.isnan(mb) and ((cl<mb) if yon=="AL" else (cl>mb)): return (cl, j-j0)
    return (C5[last], last-j0)

def Rv(yon, entry, exitpx, risk):
    if risk<=0: return 0.0
    pnl=(exitpx-entry) if yon=="AL" else (entry-exitpx)
    return (pnl-2*FEE*entry)/risk

def pct(yon, entry, exitpx):
    return 100*((exitpx-entry) if yon=="AL" else (entry-exitpx))/entry

VARIANTS=["MRREV","BRK","MIDX"]

def run():
    ex=ccxt.binanceusdm({"enableRateLimit":True,"timeout":30000}); ex.load_markets()
    perp=[m["symbol"] for m in ex.markets.values()
          if m.get("swap") and m.get("linear") and m.get("quote")=="USDT" and m.get("active")]
    t=ex.fetch_tickers(perp)
    rows=sorted([(s,t[s].get("quoteVolume") or 0) for s in perp
                 if s in t and (t[s].get("quoteVolume") or 0)>=20_000_000],
                key=lambda x:x[1],reverse=True)[:N_SYMBOLS]
    syms=[s for s,_ in rows]
    print(f"Evren: {len(syms)} coin · {DAYS} gün · saf BB (giriş 15m / çıkış 5m)\n")

    trades=[]; need=BBP+3
    for idx,sym in enumerate(syms,1):
        d15=fetch_tf(ex,sym,DAYS,"15m",15*60*1000)
        if len(d15)<need+200: print(f"[{idx}/{len(syms)}] {sym}: 15m yetersiz"); continue
        d5=fetch_tf(ex,sym,DAYS,"5m",5*60*1000)
        if len(d5)<BBP+50: print(f"[{idx}/{len(syms)}] {sym}: 5m yetersiz"); continue
        a=np.array(d15,dtype=float); TS=a[:,0]; H=a[:,2]; L=a[:,3]; C=a[:,4]
        MB,UP,DN=bollinger(C)
        b=np.array(d5,dtype=float); TS5=b[:,0]; H5=b[:,2]; L5=b[:,3]; C5=b[:,4]; MB5,_,_=bollinger(C5)
        per={v:0 for v in VARIANTS}
        for v in VARIANTS:
            open_until=-1
            for i in range(need, len(C)-1):
                if i<=open_until: continue
                yon=bb_entry(C,MB,UP,DN,i,v)
                if yon is None: continue
                px=float(C[i]); stop=stop_hesapla(yon,H[:i+1],L[:i+1]); risk=abs(px-stop)
                if risk<=0: continue
                entry_ms=int(TS[i])+15*60*1000
                ex5=sim_5(entry_ms,yon,stop,TS5,H5,L5,C5,MB5)
                if ex5 is None: continue
                epx,bars5=ex5
                trades.append(dict(sym=sym,strat=v,yon=yon,
                                   R=round(Rv(yon,px,epx,risk),4),
                                   pct=round(pct(yon,px,epx),3)))
                per[v]+=1
                open_until=i+max(1,(bars5+2)//3)
        print(f"[{idx}/{len(syms)}] {sym}: "+" ".join(f"{v}={per[v]}" for v in VARIANTS))

    if not trades: print("\nHiç giriş yok."); return
    out=os.path.expanduser("~/bt_bb_trades.csv")
    with open(out,"w",newline="") as fc:
        w=csv.DictWriter(fc,fieldnames=list(trades[0].keys())); w.writeheader()
        for tr in trades: w.writerow(tr)

    def stat(rows):
        n=len(rows)
        if n==0: return None
        Rs=[r["R"] for r in rows]; ps=[r["pct"] for r in rows]
        return dict(n=n,win=100*sum(1 for x in Rs if x>0)/n,
                    expR=sum(Rs)/n,totR=sum(Rs),avgpct=sum(ps)/n)
    def line(name,s):
        if not s: return f"{name:<18} -"
        return (f"{name:<18} n={s['n']:<5} win%={s['win']:5.1f} | beklentiR={s['expR']:+.3f} "
                f"toplamR={s['totR']:+7.1f} | ort%hareket={s['avgpct']:+5.2f}")

    print("\n"+"="*82)
    print(f"SAF BB TESTİ — {len(trades)} giriş · {DAYS}g · {len(syms)} coin · komisyon={FEE*100:.3f}%/yön")
    print("Giriş 15m BB, çıkış 5m orta bant. SuperTrend YOK.")
    print("="*82)
    names={"MRREV":"MRREV (dipten dönüş)","BRK":"BRK (kırılım)","MIDX":"MIDX (orta bant geçiş)"}
    for v in VARIANTS:
        rs=[r for r in trades if r["strat"]==v]
        print(f"\n--- {names[v]} ---")
        print(line("  TÜMÜ", stat(rs)))
        print(line("  AL", stat([r for r in rs if r["yon"]=="AL"])))
        print(line("  SAT", stat([r for r in rs if r["yon"]=="SAT"])))
    print("="*82)
    print(f"\nDetay: {out}\nÖzeti olduğu gibi Claude'a yapıştır.")

if __name__=="__main__":
    run()
