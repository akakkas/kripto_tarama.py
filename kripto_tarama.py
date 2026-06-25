import ccxt, os, sys, time, requests, numpy as np
from datetime import datetime, timezone, timedelta

INTERVAL="15m"; HTF="1h"; PERIOD=10; MULT=3.0; LIMIT=200
LOOKBACK=3; RVOLW=20; TOPN=150; MINVOL=20_000_000
BBP=20; BBK=2.0; MAXEXT=0.03; RVOLMIN=1.5; SWINGK=6
BAR_SEC=15*60; BUFFER=5
TZ=timezone(timedelta(hours=3))

def load_env(path=os.path.expanduser("~/.ktenv")):
    d={}
    try:
        with open(path) as f:
            for ln in f:
                ln=ln.strip()
                if "=" in ln and not ln.startswith("#"):
                    k,v=ln.split("=",1); d[k.strip()]=v.strip()
    except FileNotFoundError: pass
    return d

_e=load_env()
TG_TOKEN=_e.get("TG_TOKEN",""); TG_CHAT_ID=_e.get("TG_CHAT_ID","")
TG_ON=bool(TG_TOKEN and TG_CHAT_ID)

def tg(text):
    if not TG_ON:
        print("[TG kapali]\n"+text); return
    try:
        r=requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                        json={"chat_id":TG_CHAT_ID,"text":text}, timeout=15)
        j=r.json()
        if not j.get("ok"): print("  ! TG hata:", j.get("description"))
    except Exception as ex:
        print("  ! TG:", ex)

def f(x):
    if x is None or x!=x: return "-"
    a=abs(x)
    if a>=100: return f"{x:.2f}"
    if a>=1: return f"{x:.3f}"
    if a>=0.01: return f"{x:.5f}"
    return f"{x:.7f}"

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

def sinyal_bul(close,st,d,mb,up,dn,vol,ts):
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
        if d[-1]==1 and close[-2]<mb[-2] and c>mbv and rv>=RVOLMIN:
            return ("RETEST","AL",rv,int(ts[-1]))
        if d[-1]==-1 and close[-2]>mb[-2] and c<mbv and rv>=RVOLMIN:
            return ("RETEST","SAT",rv,int(ts[-1]))
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

def htf_yon(ex,sym,tf=HTF):
    try: raw=ex.fetch_ohlcv(sym,tf,limit=100)
    except Exception: return 0
    if not raw or len(raw)<PERIOD+3: return 0
    a=np.array(raw[:-1],dtype=float)
    st,d=supertrend(a[:,2],a[:,3],a[:,4])
    return int(d[-1])

def tarama():
    ex=ccxt.binanceusdm({"enableRateLimit":True}); ex.load_markets()
    perp=[m["symbol"] for m in ex.markets.values()
          if m.get("swap") and m.get("linear") and m.get("quote")=="USDT" and m.get("active")]
    t=ex.fetch_tickers(perp)
    rows=sorted([(s,t[s].get("quoteVolume") or 0) for s in perp if s in t and (t[s].get("quoteVolume") or 0)>=MINVOL],
                key=lambda x:x[1],reverse=True)[:TOPN]
    syms=[s for s,_ in rows]
    print(f"Evren: {len(syms)} perp · {INTERVAL}")
    bul=[]; need=max(PERIOD,BBP)+LOOKBACK+3
    for s in syms:
        try: raw=ex.fetch_ohlcv(s,INTERVAL,limit=LIMIT)
        except Exception as e: print("  !",s,e); continue
        if not raw or len(raw)<need: continue
        a=np.array(raw[:-1],dtype=float)
        tsa=a[:,0]; h,l,c,v=a[:,2],a[:,3],a[:,4],a[:,5]
        st,d=supertrend(h,l,c); mb,up,dn=bollinger(c)
        tip,yon,rv,trig=sinyal_bul(c,st,d,mb,up,dn,v,tsa)
        if tip is None: continue
        px=float(c[-1])
        stop,tp,rr=rr_hesapla(yon,px,h,l,up,dn)
        hy=htf_yon(ex,s)
        htf_ok=1 if (yon=="AL" and hy==1) or (yon=="SAT" and hy==-1) else 0
        bul.append((s,tip,yon,px,rv,trig,stop,tp,rr,htf_ok))
        ico="♻️" if tip=="RETEST" else ("🟢" if yon=="AL" else "🔴")
        teyit="✅" if htf_ok else "⚠️"
        print(f"  {ico} {s.split('/')[0]:<10} {tip}/{yon} {teyit}  px:{f(px)} stop:{f(stop)} tp:{f(tp)} R:R{rr:.1f} rvol:{rv:.1f}x")
    return bul,len(syms)

def satir(b):
    s,tip,yon,px,rv,trig,stop,tp,rr,htf_ok=b
    ico="♻️" if tip=="RETEST" else ("🟢" if yon=="AL" else "🔴")
    teyit="✅1s" if htf_ok else "⚠️1s"
    return (f"{ico} {s.split('/')[0]} {tip}/{yon} {teyit}\n"
            f"    px:{f(px)} stop:{f(stop)} tp:{f(tp)} R:R {rr:.1f} rvol:{rv:.1f}x")

def ozet_mesaj(bul,nc,hepsi=True):
    ts=datetime.now(TZ).strftime("%H:%M")
    lines=["📡 KRİPTO SUPERTREND", f"🕐 {ts} · {INTERVAL} · {nc} perp", "──────────────"]
    if bul: lines+= [satir(b) for b in bul]
    elif hepsi: lines.append("Temiz kurulum yok")
    al=sum(1 for b in bul if b[2]=='AL'); sat=sum(1 for b in bul if b[2]=='SAT')
    lines+=["──────────────", f"AL:{al} SAT:{sat}"]
    return "\n".join(lines)

def bekle():
    now=time.time(); nxt=(now//BAR_SEC+1)*BAR_SEC+BUFFER
    return max(1,nxt-now)

def loop():
    print("Döngü başladı — her 15dk taranacak. Durdurmak için Ctrl+C.")
    tg("🤖 Kripto tarayıcı başladı — 15dk döngü aktif.")
    seen=set()
    while True:
        try:
            bul,nc=tarama()
            yeni=[b for b in bul if (b[0],b[1],b[5]) not in seen]
            for b in yeni: seen.add((b[0],b[1],b[5]))
            if yeni:
                ts=datetime.now(TZ).strftime("%H:%M")
                msg=["📡 KRİPTO SUPERTREND", f"🕐 {ts} · {INTERVAL}", "──────────────"]+[satir(b) for b in yeni]
                tg("\n".join(msg))
            if len(seen)>5000: seen.clear()
        except Exception as e:
            print("loop hata:", e)
        time.sleep(bekle())

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="once":
        bul,nc=tarama()
        ts=datetime.now(TZ).strftime("%H:%M")
        al=sum(1 for b in bul if b[2]=='AL'); sat=sum(1 for b in bul if b[2]=='SAT')
        print(f"\nÖZET {ts} — AL:{al} SAT:{sat}  (toplam {len(bul)})")
        tg(ozet_mesaj(bul,nc,hepsi=True))
    else:
        loop()