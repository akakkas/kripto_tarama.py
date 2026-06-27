#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ÇIKIŞ KARŞILAŞTIRMA BACKTEST'i
# Aynı girişler (kripto_tarama.py sinyalleri), 3 farklı çıkış modeli yan yana:
#   A) BAND   : karşı Bollinger bandı (vurkaç — mevcut tp). Referans.
#   B) MIDBB  : BB orta bandı sana karşı kırılana kadar SÜR (Kemal'in tezi).
#   C) STFLIP : SuperTrend ters dönene kadar SÜR.
# Hepsinde yapısal swing stop = sert taban. timeout = MAXBARS.
# Çalıştır:  python backtest_exit.py

import ccxt, time, csv, os, numpy as np

# ========================== AYARLAR ==========================
N_SYMBOLS   = 150
DAYS        = 120
MAXBARS     = 192    # trend sürmeye izin var → 96'dan 192'ye (48 saat) çıkardım
FEE         = 0.0005
ONLY_REV_RVOL = 1.5    # REV eşiği

INTERVAL="15m"; PERIOD=10; MULT=3.0
LOOKBACK=3; RVOLW=20; MINVOL=20_000_000
BBP=20; BBK=2.0; MAXEXT=0.03; RVOLMIN=1.5; SWINGK=6
RETEST_TOL=0.25; TOUCHLB=10; REV_RVOL=1.5

# ========== kripto_tarama.py'den BİREBİR fonksiyonlar ==========
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

# ========================== VERİ ÇEKME ==========================
def fetch_all(ex, sym, days):
    tf_ms=15*60*1000
    since=ex.milliseconds()-days*24*60*60*1000
    out=[]
    while True:
        try:
            batch=ex.fetch_ohlcv(sym, INTERVAL, since=since, limit=1500)
        except Exception as e:
            print(f"    ! {sym} fetch hata: {e}"); break
        if not batch: break
        out+=batch
        if len(batch)<1500: break
        since=batch[-1][0]+tf_ms
        time.sleep(ex.rateLimit/1000)
    seen=set(); clean=[]
    for r in out:
        if r[0] not in seen:
            seen.add(r[0]); clean.append(r)
    clean.sort(key=lambda x:x[0])
    return clean

# ========================== ÇIKIŞ SİMÜLATÖRÜ ==========================
def simulate_exits(i, yon, entry, stop, tp_band, MB, D, H, L, C):
    """Aynı girişe 3 çıkış modeli uygula. Her model: kendi kuralı VEYA yapısal stop VEYA timeout.
    Döner: {model: (exit_px, bars)} ."""
    n=len(C); last=min(n-1, i+MAXBARS)
    res={}; pending={"A","B","C"}
    for j in range(i+1, last+1):
        hi=H[j]; lo=L[j]; cl=C[j]; mb=MB[j]; d=D[j]
        # yapısal stop (sert taban) — tüm bekleyen modelleri vurur
        stop_hit = (lo<=stop) if yon=="AL" else (hi>=stop)
        if stop_hit:
            for m in list(pending): res[m]=(stop, j-i)
            pending.clear(); break
        # A: karşı band (vurkaç)
        if "A" in pending:
            tp_hit = (hi>=tp_band) if yon=="AL" else (lo<=tp_band)
            if tp_hit: res["A"]=(tp_band, j-i); pending.discard("A")
        # B: orta bant sana karşı kapanış
        if "B" in pending and not np.isnan(mb):
            brk = (cl<mb) if yon=="AL" else (cl>mb)
            if brk: res["B"]=(cl, j-i); pending.discard("B")
        # C: SuperTrend ters dönüş
        if "C" in pending:
            flip = (d==-1) if yon=="AL" else (d==1)
            if flip: res["C"]=(cl, j-i); pending.discard("C")
        if not pending: break
    for m in pending:  # hâlâ açık → timeout, son barın kapanışı
        res[m]=(C[last], last-i)
    return res

def realized_R(yon, entry, exitpx, risk):
    if risk<=0: return 0.0
    pnl=(exitpx-entry) if yon=="AL" else (entry-exitpx)
    return (pnl-2*FEE*entry)/risk

def pct_move(yon, entry, exitpx):
    return 100*((exitpx-entry) if yon=="AL" else (entry-exitpx))/entry

# ========================== ANA AKIŞ ==========================
def run():
    ex=ccxt.binanceusdm({"enableRateLimit":True,"timeout":30000}); ex.load_markets()
    perp=[m["symbol"] for m in ex.markets.values()
          if m.get("swap") and m.get("linear") and m.get("quote")=="USDT" and m.get("active")]
    t=ex.fetch_tickers(perp)
    rows=sorted([(s,t[s].get("quoteVolume") or 0) for s in perp
                 if s in t and (t[s].get("quoteVolume") or 0)>=MINVOL],
                key=lambda x:x[1],reverse=True)[:N_SYMBOLS]
    syms=[s for s,_ in rows]
    print(f"Evren: {len(syms)} coin · {DAYS} gün · {INTERVAL} · MAXBARS={MAXBARS}\n")

    trades=[]; need=max(PERIOD,BBP)+LOOKBACK+3
    for idx,sym in enumerate(syms,1):
        data=fetch_all(ex, sym, DAYS)
        if len(data)<need+MAXBARS+50:
            print(f"[{idx}/{len(syms)}] {sym}: yetersiz"); continue
        a=np.array(data,dtype=float)
        TS=a[:,0]; H=a[:,2]; L=a[:,3]; C=a[:,4]; V=a[:,5]
        ST,D=supertrend(H,L,C); MB,UP,DN=bollinger(C)
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
            ex_res=simulate_exits(i,yon,px,stop,tp,MB,D,H,L,C)
            row=dict(sym=sym,tip=tip,yon=yon,rr=round(rr,3),rv=round(float(rv),3),risk=risk,entry=px)
            for m,(epx,bars) in ex_res.items():
                row[f"R_{m}"]=round(realized_R(yon,px,epx,risk),4)
                row[f"pct_{m}"]=round(pct_move(yon,px,epx),3)
                row[f"bars_{m}"]=bars
            trades.append(row)
            # bir sonraki sinyale, en geç çıkıştan sonra bak (model B/C uzun sürebilir)
            open_until=i+max(b for _,b in ex_res.values()); cs+=1
        print(f"[{idx}/{len(syms)}] {sym}: {cs} giriş")

    if not trades:
        print("\nHiç giriş yok."); return
    out=os.path.expanduser("~/bt_exit_trades.csv")
    with open(out,"w",newline="") as fcsv:
        w=csv.DictWriter(fcsv,fieldnames=list(trades[0].keys())); w.writeheader()
        for tr in trades: w.writerow(tr)

    def stat(rows,m):
        n=len(rows)
        if n==0: return None
        Rs=[r[f"R_{m}"] for r in rows]
        wins=sum(1 for x in Rs if x>0)
        pcts=[r[f"pct_{m}"] for r in rows]
        bars=[r[f"bars_{m}"] for r in rows]
        return dict(n=n,win=100*wins/n,expR=sum(Rs)/n,totR=sum(Rs),
                    avgpct=sum(pcts)/n,avgbar=sum(bars)/n)
    def line(name,s):
        if not s: return f"{name:<16} -"
        return (f"{name:<16} n={s['n']:<4} win%={s['win']:5.1f} | beklentiR={s['expR']:+.3f} "
                f"toplamR={s['totR']:+7.1f} | ort%hareket={s['avgpct']:+5.2f} ort.bar={s['avgbar']:4.0f}")

    MODELS=[("A","BAND (vurkaç)"),("B","MIDBB (orta bant sür)"),("C","STFLIP (flip'e kadar)")]
    print("\n"+"="*82)
    print(f"ÇIKIŞ KARŞILAŞTIRMA — {len(trades)} giriş · {DAYS}g · {len(syms)} coin · komisyon={FEE*100:.3f}%/yön")
    print("Aynı girişler, 3 farklı çıkış. ort%hareket = fiyatın yakaladığı yön (kaldıraçsız).")
    print("="*82)
    for code,name in MODELS:
        print(f"\n--- {name} ---")
        print(line("  TÜMÜ", stat(trades,code)))
        for tp_ in ("FLIP","RETEST","REV"):
            print(line(f"  {tp_}", stat([r for r in trades if r['tip']==tp_],code)))
    print("="*82)
    print(f"\nDetay: {out}")
    print("Özeti olduğu gibi Claude'a yapıştır.")

if __name__=="__main__":
    run()
