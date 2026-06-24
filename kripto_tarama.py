import ccxt, numpy as np
from datetime import datetime, timezone, timedelta

INTERVAL="15m"; PERIOD=10; MULT=3.0; LIMIT=200
LOOKBACK=3; RVOLW=20; TOPN=150; MINVOL=20_000_000
TZ=timezone(timedelta(hours=3))

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

def analiz_et(d,lb=LOOKBACK):
    n=len(d); end=max(1,n-lb)
    for i in range(n-1,end-1,-1):
        if d[i]!=d[i-1]: return ("AL" if d[i]==1 else "SAT"),i
    return None,None

def rvol(v,w=RVOLW):
    if len(v)<w+1: return float("nan")
    o=v[-(w+1):-1].mean()
    return v[-1]/o if o>0 else float("nan")

def main():
    ex=ccxt.binanceusdm({"enableRateLimit":True}); ex.load_markets()
    perp=[m["symbol"] for m in ex.markets.values()
          if m.get("swap") and m.get("linear") and m.get("quote")=="USDT" and m.get("active")]
    t=ex.fetch_tickers(perp)
    rows=sorted([(s,t[s].get("quoteVolume") or 0) for s in perp if s in t and (t[s].get("quoteVolume") or 0)>=MINVOL],
                key=lambda x:x[1],reverse=True)[:TOPN]
    syms=[s for s,_ in rows]
    print(f"Evren: {len(syms)} perp · {INTERVAL}")
    bul=[]
    for s in syms:
        try: raw=ex.fetch_ohlcv(s,INTERVAL,limit=LIMIT)
        except Exception as e: print("  !",s,e); continue
        if not raw or len(raw)<PERIOD+LOOKBACK+3: continue
        a=np.array(raw[:-1],dtype=float)
        st,d=supertrend(a[:,2],a[:,3],a[:,4])
        yon,_=analiz_et(d)
        if yon is None: continue
        px=float(a[-1,4]); rv=rvol(a[:,5])
        bul.append((s,yon,px,rv))
        print(f"  {'🟢' if yon=='AL' else '🔴'} {s.split('/')[0]:<10} {yon}  px:{px}  rvol:{rv:.1f}x")
    ts=datetime.now(TZ).strftime("%H:%M")
    print(f"\nÖZET {ts} — AL:{sum(1 for b in bul if b[1]=='AL')} SAT:{sum(1 for b in bul if b[1]=='SAT')}")

if __name__=="__main__":
    main()
