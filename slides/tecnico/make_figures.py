# Gera as figuras dos slides (dados reais) em docs/slides/tecnico/figures/
import json, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:/Users/jose.souza/Documents/fleven"
FIGS = os.path.join(ROOT, "docs/slides/tecnico/figures")
os.makedirs(FIGS, exist_ok=True)

AMBER="#F5970A"; BLUE="#2563EB"; GREEN="#22C55E"; ORANGE="#C2410C"; DARK="#0A0A0F"; GRAY="#64748B"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.transparent": True,
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#334155", "axes.labelcolor": "#0f172a",
    "xtick.color": "#334155", "ytick.color": "#334155", "text.color": "#0f172a",
})

GROUPS = ["G01","G02","G03","G04","G05"]
STRAT = "fedper_fedavg"

def client_dirs(exp):
    return sorted(glob.glob(os.path.join(ROOT, "metrics", exp, "client_*")),
                  key=lambda p: int(p.split("_")[-1]))

def r2(y_true, y_pred):
    y_true=np.asarray(y_true,float); y_pred=np.asarray(y_pred,float)
    m=np.isfinite(y_true)&np.isfinite(y_pred)
    y_true,y_pred=y_true[m],y_pred[m]
    ss_res=np.sum((y_true-y_pred)**2); ss_tot=np.sum((y_true-y_true.mean())**2)
    return 1-ss_res/ss_tot if ss_tot>0 else float("nan")

# ---------- 1) R2 global por grupo (FedPer + FedAvg) ----------
r2_by_group={}; plen_by_group={}
for g in GROUPS:
    exp=f"{g}_{STRAT}"
    yt,yp=[],[]; pl=5
    for c in client_dirs(exp):
        fp=os.path.join(c,"final_predictions.json")
        if not os.path.exists(fp): continue
        d=json.load(open(fp))
        yt+=d.get("y_true",[]); yp+=d.get("y_pred",[]); pl=d.get("prediction_length",pl) or pl
    if yt:
        r2_by_group[g]=r2(yt,yp); plen_by_group[g]=pl

print("R2 por grupo:", {k:round(v,3) for k,v in r2_by_group.items()})
print("horizonte (pred_len) por grupo:", plen_by_group)

fig,ax=plt.subplots(figsize=(5.2,3.0))
gs=list(r2_by_group.keys()); vals=[r2_by_group[g] for g in gs]
bars=ax.bar(gs, vals, color=BLUE, width=0.6)
# destaca o melhor
best=int(np.nanargmax(vals)); bars[best].set_color(AMBER)
for b,v,g in zip(bars,vals,gs):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    # marca o horizonte quando difere do comum (5), para leitura honesta
    if plen_by_group.get(g,5)!=5:
        ax.text(b.get_x()+b.get_width()/2, 0.04, f"h={plen_by_group[g]}", ha="center",
                va="bottom", fontsize=8, color="white", fontweight="bold")
ax.set_ylabel("$R^2$ global"); ax.set_ylim(0, max(vals)*1.2)
ax.set_title("FedPer + FedAvg - desempenho por grupo", fontsize=11)
fig.savefig(os.path.join(FIGS,"results_r2.pdf")); plt.close(fig)

# ---------- 1b) R2 por PASSO do horizonte (degradacao) ----------
def r2_per_step(exp):
    pl=None; parts={}
    for c in client_dirs(exp):
        fp=os.path.join(c,"final_predictions.json")
        if not os.path.exists(fp): continue
        d=json.load(open(fp))
        yt=np.asarray(d.get("y_true",[]),float); yp=np.asarray(d.get("y_pred",[]),float)
        pl=d.get("prediction_length",pl) or 5
        n=len(yt)//pl
        if n==0: continue
        yt2=yt[:n*pl].reshape(n,pl); yp2=yp[:n*pl].reshape(n,pl)
        for s in range(pl):
            parts.setdefault(s,[[],[]])
            parts[s][0].append(yt2[:,s]); parts[s][1].append(yp2[:,s])
    steps=sorted(parts); return steps, [r2(np.concatenate(parts[s][0]),
                                          np.concatenate(parts[s][1])) for s in steps]

fig,ax=plt.subplots(figsize=(5.6,3.1))
gcolors={"G01":BLUE,"G02":AMBER,"G03":GREEN,"G04":ORANGE,"G05":"#7c3aed"}
for g in GROUPS:
    exp=f"{g}_{STRAT}"
    if not client_dirs(exp): continue
    steps,vals=r2_per_step(exp)
    ax.plot([s+1 for s in steps],vals,marker="o",ms=4,color=gcolors.get(g,GRAY),label=g)
ax.set_xlabel("Passo à frente (horizonte de previsão)"); ax.set_ylabel("$R^2$")
ax.set_xticks(range(1,6))
ax.set_title("R² por passo de previsão - FedPer+FedAvg",fontsize=11)
ax.legend(frameon=False,fontsize=8,ncol=5,loc="lower left")
fig.savefig(os.path.join(FIGS,"results_r2_step.pdf")); plt.close(fig)

# ---------- 2) Curvas de perda GLOBAL (grupo showcase) ----------
SHOW="G01"
exp=f"{SHOW}_{STRAT}"
def mean_curve(section):
    per={}
    for c in client_dirs(exp):
        h=json.load(open(os.path.join(c,"metrics_history.json")))
        for e in h.get(section,[]):
            per.setdefault(e["round"],[]).append(e["loss"])
    rounds=sorted(per); return rounds,[np.mean(per[r]) for r in rounds]

rt,tr=mean_curve("train"); re,ev=mean_curve("eval")
fig,ax=plt.subplots(figsize=(5.2,3.0))
ax.plot(rt,tr,color=BLUE,marker="o",ms=3,label="Treino (médio)")
ax.plot(re,ev,color=ORANGE,marker="s",ms=3,label="Validação (médio)")
ax.set_xlabel("Round federado"); ax.set_ylabel("Perda (MSE norm.)")
ax.set_title(f"Perda global - {SHOW} (FedPer+FedAvg)", fontsize=11)
ax.legend(frameon=False, fontsize=9)
fig.savefig(os.path.join(FIGS,"loss_global.pdf")); plt.close(fig)

# ---------- 3) Curvas de perda POR CLIENTE (eval) ----------
fig,ax=plt.subplots(figsize=(5.2,3.0))
for i,c in enumerate(client_dirs(exp)):
    h=json.load(open(os.path.join(c,"metrics_history.json")))
    ev_c=h.get("eval",[])
    if not ev_c: continue
    rr=[e["round"] for e in ev_c]; ll=[e["loss"] for e in ev_c]
    ax.plot(rr,ll,color=GRAY,alpha=0.45,lw=1,
            label="Clientes" if i==0 else None)
ax.plot(re,ev,color=ORANGE,lw=2.6,label="Média")
ax.set_xlabel("Round federado"); ax.set_ylabel("Perda de validação")
ax.set_title(f"Perda por cliente - {SHOW}", fontsize=11)
ax.legend(frameon=False, fontsize=9)
fig.savefig(os.path.join(FIGS,"loss_per_client.pdf")); plt.close(fig)

# ---------- 4) Distribuicao do alvo (y_true real, grupo showcase) ----------
yt=[]
for c in client_dirs(exp):
    fp=os.path.join(c,"final_predictions.json")
    if os.path.exists(fp):
        yt+=json.load(open(fp)).get("y_true",[])
yt=np.array(yt,float); yt=yt[np.isfinite(yt)]
fig,ax=plt.subplots(figsize=(5.2,3.0))
ax.hist(yt, bins=40, color=AMBER, edgecolor=ORANGE, linewidth=0.4)
ax.set_xlabel("Consumo de energia (kWh)"); ax.set_ylabel("Frequência")
ax.set_title(f"Distribuição do consumo - {SHOW}", fontsize=11)
fig.savefig(os.path.join(FIGS,"eda_target_dist.pdf")); plt.close(fig)

# ---------- 5) Volume de dados por cliente (stats.csv) ----------
stats_path=glob.glob(os.path.join(ROOT,"results/eda/G01_*/G01_stats.csv"))
if stats_path:
    df=pd.read_csv(stats_path[0])
    var=df["variable"].iloc[0]  # uma variavel por cliente
    sub=df[df["variable"]==var].sort_values("client")
    fig,ax=plt.subplots(figsize=(5.2,3.0))
    ax.bar(sub["client"].astype(str), sub["count"]/1000.0, color=GREEN, width=0.7)
    ax.set_xlabel("Cliente"); ax.set_ylabel("Amostras (mil)")
    ax.set_title("Volume de dados por cliente - G01", fontsize=11)
    fig.savefig(os.path.join(FIGS,"eda_samples.pdf")); plt.close(fig)

# ---------- 6) EDA das FEATURES (dados brutos) ----------
FEATURES=["Vehicle Speed[km/h]","MAF[g/sec]","Engine RPM[RPM]","Absolute Load[%]"]
TARGET="Energy_Consumption"
SHORT={"Vehicle Speed[km/h]":"Velocidade (km/h)","MAF[g/sec]":"MAF (g/s)",
       "Engine RPM[RPM]":"RPM","Absolute Load[%]":"Carga (%)","Energy_Consumption":"Consumo"}
data_glob=glob.glob(os.path.join(ROOT,"data/G01_*/train/client_*/*.parquet"))
if data_glob:
    import random; random.seed(0)
    files=data_glob if len(data_glob)<=500 else random.sample(data_glob,500)
    dfs=[]
    for f in files:
        try: dfs.append(pd.read_parquet(f, columns=FEATURES+[TARGET]))
        except Exception: pass
    raw=pd.concat(dfs, ignore_index=True)

    # histogramas 2x2 das features
    fig,axes=plt.subplots(2,2,figsize=(6.6,4.2))
    cols=[BLUE,AMBER,GREEN,ORANGE]
    for ax,feat,col in zip(axes.ravel(),FEATURES,cols):
        v=raw[feat].replace([np.inf,-np.inf],np.nan).dropna().values
        ax.hist(v,bins=40,color=col,edgecolor="white",linewidth=0.2)
        ax.set_title(SHORT[feat],fontsize=10); ax.set_yticks([])
        ax.tick_params(labelsize=8)
    fig.suptitle("Distribuição das features - G01",fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS,"eda_features.pdf")); plt.close(fig)

    # mapa de correlacao features x alvo
    corr=raw[FEATURES+[TARGET]].corr()
    labels=[SHORT[c] for c in corr.columns]
    fig,ax=plt.subplots(figsize=(4.8,4.0))
    im=ax.imshow(corr.values,cmap="RdBu_r",vmin=-1,vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,rotation=40,ha="right",fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels,fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val=corr.values[i,j]
            ax.text(j,i,f"{val:.2f}",ha="center",va="center",
                    color="white" if abs(val)>0.5 else "#0f172a",fontsize=8)
    fig.colorbar(im,fraction=0.046,pad=0.04)
    ax.set_title("Correlação (features × alvo) - G01",fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS,"eda_corr.pdf")); plt.close(fig)
    print("EDA das features OK")

print("Figuras geradas em", FIGS)
print(sorted(os.listdir(FIGS)))
