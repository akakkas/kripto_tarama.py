#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# OVERSOLD BOUNCE LONG — "düşüşte dipten hacimli sıçrama" tezi
# Tez: 1h DÜŞÜŞ trendinde + 15m fiyat alt banda yakın/altında + hacimli yeşil tepki → LONG
# Çıkış: 15m orta banda dönüş (mean-reversion kâr al) + swing stop sert taban
# n'i büyütmek için RETEST şartını gevşettik: sadece mid reclaim değil, dipten her hacimli bounce.
# Bootstrap %95 CI ile şans/gerçek ayrımı.
# Çalıştır:  python backtest_ovs.py

import ccxt, time, csv, os, random, numpy as np

N_SYMBOLS = 103
DAYS      = 120
FEE       = 0.0005
HOLD_BARS = 96       # 15m bar (24 saat) mean-reversion için tavan

BBP=20; BBK=2.0; RVOLW=20; MINVOL=20_000_000; SWINGK=6
H1_PERIOD=10; H1_MULT=3.0          # 1h trend yönü için SuperTrend
BOUNCEK=3                          # son kaç barda alt banda değme arıyoruz
RVOLMIN=1.3                        # hacim teyidi (biraz gevşek, n için)
NEARBAND=0.005                     # alt banda "yakın" toleransı (%0.5)

def supertrend(h,l,c,period,mult):
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
    d=np.ones(n,dtype=int)
    for i in range(1,n):
        if c[i]>fu[i-1]: d[i]=1
        elif c[i]<fl[i-1]: d[i]=-1
        else: d[i]=d[i-1]
    return d

def bollinger(c,p=BBP,k=BBK):
    n=len(c); mb=np.full(n,np.nan); up=np.full(n,np.nan); dn=np.full(n,np.nan)
    for i in range(p-1,n):
        w=c[i-p+1:i+1]; m=w.mean(); s=w.std()
        mb[i]=m; up[i]=m+k*s; dn[i]=m-k*s
    return mb,up,dn

def rvol_at(v,i,w=RVOLW):
    if i<w: return float("nan")
    o=v[i-w:i].mean()
    return v[i]/o if o>0 else float("nan")

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

# mean-reversion çıkış: orta banda DÖNÜŞ (long için yukarı) = kâr, swing stop = taban
def sim_revert(i,entry,stop,H,L,C,MB,maxbars):
    n=len(C); last=min(n-1,i+maxbars)
    for j in range(i+1,last+1):
        hi=H[j]; lo=L[j]; cl=C[j]; mb=MB[j]
        if lo<=stop: return stop,j-i           # sert taban
        if not np.isnan(mb) and hi>=mb:         # orta banda döndü → kâr al
            return mb,j-i
    return C[last],last-i

def Rv(entry,exitpx,risk):
    if risk<=0: return 0.0
    return ((exitpx-entry)-2*FEE*entry)/risk

def pct(entry,exitpx):
    return 100*(exitpx-entry)/entry

def run():
    ex=ccxt.binanceusdm({"enableRateLimit":True,"timeout":30000}); ex.load_markets()
    perp=[m["symbol"] for m in ex.markets.values()
          if m.get("swap") and m.get("linear") and m.get("quote")=="USDT" and m.get("active")]
    t=ex.fetch_tickers(perp)
    rows=sorted([(s,t[s].get("quoteVolume") or 0) for s in perp
                 if s in t and (t[s].get("quoteVolume") or 0)>=MINVOL],
                key=lambda x:x[1],reverse=True)[:N_SYMBOLS]
    syms=[s for s,_ in rows]
    print(f"Evren: {len(syms)} coin · {DAYS} gün · OVERSOLD BOUNCE LONG tezi\n")

    trades=[]
    for idx,sym in enumerate(syms,1):
        d15=fetch_tf(ex,sym,DAYS,"15m",15*60*1000)
        if len(d15)<BBP+50: print(f"[{idx}/{len(syms)}] {sym}: 15m yetersiz"); continue
        d1h=fetch_tf(ex,sym,DAYS,"1h",60*60*1000)
        if len(d1h)<H1_PERIOD+20: print(f"[{idx}/{len(syms)}] {sym}: 1h yetersiz"); continue

        b=np.array(d15,dtype=float); H=b[:,2]; L=b[:,3]; C=b[:,4]; V=b[:,5]; TS=b[:,0]
        MB,UP,DN=bollinger(C)
        e=np.array(d1h,dtype=float); H1=e[:,2]; L1=e[:,3]; C1=e[:,4]; TS1=e[:,0]
        D1=supertrend(H1,L1,C1,H1_PERIOD,H1_MULT)

        open_until=-1; cnt=0
        for i in range(BBP+2,len(C)-1):
            if i<=open_until: continue
            mb=MB[i]; dn=DN[i]
            if np.isnan(mb) or np.isnan(dn): continue
            # 1) 1h DÜŞÜŞ trendi mi?
            entry_ms=int(TS[i])
            j1=int(np.searchsorted(TS1,entry_ms+15*60*1000))-1
            if j1<0 or j1>=len(D1): continue
            if D1[j1]!=-1: continue            # sadece 1h düşüşte
            # 2) son BOUNCEK barda fiyat alt banda değdi/altına indi mi? (oversold)
            lo_win=L[max(0,i-BOUNCEK+1):i+1]; dn_win=DN[max(0,i-BOUNCEK+1):i+1]
            touched=False
            for k in range(len(lo_win)):
                if not np.isnan(dn_win[k]) and lo_win[k]<=dn_win[k]*(1+NEARBAND):
                    touched=True; break
            if not touched: continue
            # 3) şu an hacimli YEŞİL tepki barı mı? (bounce başladı)
            rv=rvol_at(V,i)
            if rv!=rv or rv<RVOLMIN: continue
            if C[i]<=C[i-1]: continue          # yeşil/yükselen kapanış
            if C[i]<=dn*(1+NEARBAND): continue # banttan yukarı ayrılmış olsun (dönüş teyidi)

            px=float(C[i])
            stop=float(np.min(L[max(0,i-SWINGK+1):i+1]))
            risk=px-stop
            if risk<=0: continue
            epx,bars=sim_revert(i,px,stop,H,L,C,MB,HOLD_BARS)
            trades.append(dict(sym=sym,R=round(Rv(px,epx,risk),4),
                               pct=round(pct(px,epx),3),bars=bars,rvol=round(rv,2)))
            open_until=i+bars; cnt+=1
        print(f"[{idx}/{len(syms)}] {sym}: {cnt} giriş")

    if not trades: print("\nHiç giriş yok."); return
    out=os.path.expanduser("~/bt_ovs_trades.csv")
    with open(out,"w",newline="") as fc:
        w=csv.DictWriter(fc,fieldnames=list(trades[0].keys())); w.writeheader()
        for tr in trades: w.writerow(tr)

    Rs=[r["R"] for r in trades]; n=len(Rs)
    wins=[x for x in Rs if x>0]
    expR=sum(Rs)/n; win=100*len(wins)/n
    avgwin=sum(wins)/len(wins) if wins else 0
    los=[x for x in Rs if x<=0]; avglos=sum(los)/len(los) if los else 0
    avgpct=sum(r["pct"] for r in trades)/n; avgbars=sum(r["bars"] for r in trades)/n
    # bootstrap
    random.seed(42); boots=[]
    for _ in range(5000):
        s=sum(random.choice(Rs) for _ in range(n))/n; boots.append(s)
    boots.sort(); lo=boots[int(0.025*len(boots))]; hi=boots[int(0.975*len(boots))]

    print("\n"+"="*88)
    print(f"OVERSOLD BOUNCE LONG — {n} giriş · {DAYS}g · {len(syms)} coin")
    print("1h düşüş trendi + 15m alt-bant oversold + hacimli yeşil tepki → LONG, orta banda dönüşte kâr")
    print("="*88)
    print(f"  n={n}  win%={win:.1f}  beklentiR={expR:+.3f}  toplamR={sum(Rs):+.1f}")
    print(f"  ort%hareket={avgpct:+.2f}  ort.tutuş={avgbars:.1f} bar ({avgbars*15/60:.1f}s)")
    print(f"  kazanan ort={avgwin:+.2f}R  kaybeden ort={avglos:+.2f}R  ort.rvol={sum(r['rvol'] for r in trades)/n:.2f}")
    print("-"*88)
    print(f"  BOOTSTRAP %95 güven aralığı: [{lo:+.3f}, {hi:+.3f}]  (5000 örnek)")
    if lo>0:
        print("  ✅✅ ALT SINIR > 0 — dağıtım pozitif, ŞANS DEĞİL. GERÇEK İZ.")
    elif expR>0:
        print("  ⚠️ pozitif ama alt sınır sıfırı kapsıyor — iz var, kanıt henüz yok")
    else:
        print("  ❌ negatif — tez bu örneklemde tutmadı")
    print("="*88)
    print(f"\nDetay: {out}\nÖzeti olduğu gibi Claude'a yapıştır.")

if __name__=="__main__":
    run()
