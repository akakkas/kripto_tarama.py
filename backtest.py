#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# kripto_tarama.py SİNYAL BACKTEST'i
# Sinyal mantığı (supertrend / bollinger / rvol / sinyal_bul / rr_hesapla)
# kripto_tarama.py'den BİREBİR kopyalandı. Test edilen şey = canlıdaki sinyal.
#
# Çalıştır:  python backtest.py
# Çıktı:     terminale özet tablo + ~/bt_trades.csv (her trade tek tek)

import ccxt, time, csv, os, sys, numpy as np

# ========================== AYARLAR ==========================
N_SYMBOLS   = 30        # kaç coin (hacme göre en üst N). İlk denemede 30, sonra 150'ye çıkar.
DAYS        = 60        # kaç günlük 15m geçmiş
MAXBARS     = 96        # bir trade en fazla kaç bar açık kalır (96 = 24 saat). Sonra "timeout" ile kapanır.
FEE         = 0.0005    # tek yön komisyon (0.05% taker). Round-trip ~0.1%. Gerçekçilik için.
TIE_IS_STOP = True      # aynı barda hem stop hem tp dokunursa: kötümser say (stop). Look-ahead bias'ı önler.
REV_SWEEP   = [1.2, 1.5, 2.0, 2.5]   # REV için rvol eşiği taraması

# ===== kripto_tarama.py ile AYNI parametreler =====
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
    """15m OHLCV'yi sayfalayarak çek (Binance limit=1500/istek)."""
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
    # zaman sırası + tekrar temizliği
    seen=set(); clean=[]
    for r in out:
        if r[0] not in seen:
            seen.add(r[0]); clean.append(r)
    clean.sort(key=lambda x:x[0])
    return clean

def htf_dir_map(ts15, h, l, c):
    """15m barlardan 1h SuperTrend yönü üret. {1h_index: yön} döner.
    htf_yon ile aynı mantık: o 15m bar kapanırken son KAPALI 1h barın yönü."""
    HMS=3600*1000
    hidx=(ts15//HMS).astype(np.int64)
    uniq=np.unique(hidx)
    o_=[]; h_=[]; l_=[]; c_=[]
    for u in uniq:
        m=hidx==u
        o_.append(c[m][0]); h_.append(h[m].max()); l_.append(l[m].min()); c_.append(c[m][-1])
    if len(c_)<PERIOD+3:
        return {}
    st,d=supertrend(np.array(h_),np.array(l_),np.array(c_))
    return {int(uniq[i]): int(d[i]) for i in range(len(uniq))}

# ========================== SİMÜLATÖR ==========================
def simulate(entry_i, yon, entry_px, stop, tp, H, L, C):
    """entry_i barının KAPANIŞINDA gir, sonraki barlarda stop/tp ara.
    Döner: (sonuc, exit_px, exit_i) ; sonuc: 'tp'|'stop'|'timeout'"""
    n=len(C); last=min(n-1, entry_i+MAXBARS)
    for j in range(entry_i+1, last+1):
        hi=H[j]; lo=L[j]
        if yon=="AL":
            hit_stop = lo<=stop; hit_tp = hi>=tp
            if hit_stop and hit_tp:
                return ("stop" if TIE_IS_STOP else "tp", stop if TIE_IS_STOP else tp, j)
            if hit_stop: return ("stop", stop, j)
            if hit_tp:   return ("tp", tp, j)
        else:
            hit_stop = hi>=stop; hit_tp = lo<=tp
            if hit_stop and hit_tp:
                return ("stop" if TIE_IS_STOP else "tp", stop if TIE_IS_STOP else tp, j)
            if hit_stop: return ("stop", stop, j)
            if hit_tp:   return ("tp", tp, j)
    return ("timeout", C[last], last)

def realized_R(yon, entry_px, exit_px, risk):
    if risk<=0: return 0.0
    pnl = (exit_px-entry_px) if yon=="AL" else (entry_px-exit_px)
    fee = 2*FEE*entry_px
    return (pnl-fee)/risk

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
    print(f"Evren: {len(syms)} coin · {DAYS} gün · {INTERVAL}\n")

    trades=[]   # her giriş: dict
    need=max(PERIOD,BBP)+LOOKBACK+3

    for idx,sym in enumerate(syms,1):
        data=fetch_all(ex, sym, DAYS)
        if len(data)<need+MAXBARS+50:
            print(f"[{idx}/{len(syms)}] {sym}: yetersiz veri ({len(data)})"); continue
        a=np.array(data,dtype=float)
        TS=a[:,0]; O=a[:,1]; H=a[:,2]; L=a[:,3]; C=a[:,4]; V=a[:,5]
        # indikatörleri TÜM seri üzerinde BİR KEZ hesapla (causal, slice ile aynı sonuç)
        ST,D=supertrend(H,L,C); MB,UP,DN=bollinger(C)
        hmap=htf_dir_map(TS,H,L,C); HMS=3600*1000

        seen=set(); open_until=-1   # aynı sembolde aynı anda tek trade
        count_sym=0
        # bar bar ilerle: i = "son kapalı bar". Girişten sonra forward simüle.
        for i in range(need, len(C)-1):
            if i<=open_until:   # açık trade bitene kadar yeni sinyal alma
                continue
            sl=slice(0,i+1)
            tip,yon,rv,trig=sinyal_bul(C[sl],ST[sl],D[sl],MB[sl],UP[sl],DN[sl],
                                       V[sl],TS[sl],H[sl],L[sl])
            if tip in (None,"TREND"): continue
            key=(tip,trig)
            if key in seen: continue
            seen.add(key)
            px=float(C[i])
            stop,tp,rr=rr_hesapla(yon,px,H[sl],L[sl],UP[sl],DN[sl])
            risk=abs(px-stop)
            if rr<=0 or risk<=0:
                # tp girişin yanlış tarafında → girilecek yer yok. Canlıda R:R<=0 GÖRÜNÜR
                # ama trade'e dönüşmez (PORTAL örneği). Sahte anında-tp'yi önlemek için atla.
                continue
            hi_idx=int(TS[i]//HMS)-1
            hy=hmap.get(hi_idx,0)
            htf_ok = 1 if (yon=="AL" and hy==1) or (yon=="SAT" and hy==-1) else 0
            res,exit_px,exit_i=simulate(i,yon,px,stop,tp,H,L,C)
            R=realized_R(yon,px,exit_px,risk)
            trades.append(dict(sym=sym,tip=tip,yon=yon,ts=int(TS[i]),
                               entry=px,stop=stop,tp=tp,rr=round(rr,3),rv=round(float(rv),3),
                               htf_ok=htf_ok,result=res,exitpx=exit_px,bars=exit_i-i,R=round(R,4)))
            open_until=exit_i; count_sym+=1
        print(f"[{idx}/{len(syms)}] {sym}: {count_sym} sinyal")

    if not trades:
        print("\nHiç sinyal bulunamadı. DAYS/N_SYMBOLS artır."); return

    # ---- CSV ----
    out=os.path.expanduser("~/bt_trades.csv")
    with open(out,"w",newline="") as fcsv:
        w=csv.DictWriter(fcsv,fieldnames=list(trades[0].keys())); w.writeheader()
        for tr in trades: w.writerow(tr)

    # ---- ÖZET ----
    def stat(rows):
        n=len(rows)
        if n==0: return None
        wins=sum(1 for r in rows if r["result"]=="tp")
        losses=sum(1 for r in rows if r["result"]=="stop")
        tmo=sum(1 for r in rows if r["result"]=="timeout")
        Rs=[r["R"] for r in rows]
        rrs=[r["rr"] for r in rows]
        return dict(n=n,win=100*wins/n,loss=100*losses/n,tmo=100*tmo/n,
                    expR=sum(Rs)/n,totR=sum(Rs),avgrr=sum(rrs)/n)
    def line(name,s):
        if not s: return f"{name:<14} -"
        return (f"{name:<14} n={s['n']:<4} win%={s['win']:5.1f} loss%={s['loss']:5.1f} "
                f"tmo%={s['tmo']:5.1f} | beklentiR={s['expR']:+.3f} toplamR={s['totR']:+.1f} "
                f"ort.RR={s['avgrr']:.2f}")

    print("\n"+"="*78)
    print(f"BACKTEST ÖZET — {len(trades)} trade · {DAYS}g · {len(syms)} coin · komisyon={FEE*100:.3f}%/yön")
    print("="*78)
    print(line("TÜMÜ", stat(trades)))
    for tp_ in ("FLIP","RETEST","REV"):
        print(line(tp_, stat([r for r in trades if r["tip"]==tp_])))
    print("-"*78)
    print("HTF AYRIMI (1h teyit yönü):")
    print(line("  HTF ✅", stat([r for r in trades if r["htf_ok"]==1])))
    print(line("  HTF ⚠️", stat([r for r in trades if r["htf_ok"]==0])))
    print(line("  REV ✅", stat([r for r in trades if r["tip"]=="REV" and r["htf_ok"]==1])))
    print(line("  REV ⚠️", stat([r for r in trades if r["tip"]=="REV" and r["htf_ok"]==0])))

    # ---- REV_RVOL taraması (eşik ≥ mevcut için kayıtlı REV trade'lerini filtrele) ----
    print("-"*78)
    revs=[r for r in trades if r["tip"]=="REV"]
    print(f"REV_RVOL TARAMASI (şu an eşik={REV_RVOL}) — eşiği yükseltince REV kalitesi:")
    print(f"  {'eşik':>5} {'n':>5} {'win%':>7} {'beklentiR':>11} {'toplamR':>9}")
    for thr in sorted(set(REV_SWEEP)):
        if thr < REV_RVOL:
            print(f"  {thr:>5} : mevcut eşik {REV_RVOL} altında — bu tur ölçülemez (ayrı koşu gerekir)")
            continue
        sub=[r for r in revs if r["rv"]>=thr]
        s=stat(sub)
        if not s: print(f"  {thr:>5} {0:>5}"); continue
        print(f"  {thr:>5} {s['n']:>5} {s['win']:>6.1f}% {s['expR']:>+10.3f} {s['totR']:>+8.1f}")
    print("  not: yaklaşıktır — eşik yükselince atlanan trade başka bir trade'e yer açabilir.")
    print("       Kesin sonuç için REV_RVOL'ü yukarıda değiştirip yeniden çalıştır.")

    print("="*78)
    print(f"\nDetay CSV: {out}")
    print("Bu özeti olduğu gibi Claude'a yapıştır. Birlikte yorumlayalım.")

if __name__=="__main__":
    run()
