
# ============================
#      Mount Google Drive
# ============================

from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gene Expression Programming (GEP) for CFST columns
- Inputs:  A1 (kN), A2 (kN), A3 (-), A4 (-)
- Target:  Pcc (kN)
- Split:   first 140 train, next 60 validation
- Settings: head=7, tail=8, genes=3, linking=+, pop=30, gens=15000 (configurable), fitness=RMSE
- Function set: +, -, *, /, Nop, X2, X3, Sqrt, 3Rt, Ln, Log
- Constants: 5 per gene, integers in [-10, +10]
Outputs:
  - TXT report with formula and metrics
  - JSON with chromosomes/consts and metrics
  - PNG Expression Trees for G1–G3
Usage:
  python gep_cfct_final.py --excel "CFST (GEP).xlsx" --sheet 0 --gens 15000 --pop 30 --seeds 8
"""
import numpy as np, pandas as pd, math, random, re, json, os, argparse, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# In[ ]:


# ---------- protected primitives ----------
def pdiv(a,b):
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(np.abs(b)>1e-12, a/b, 0.0)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
def psqrt(a): return np.sqrt(np.abs(a))
def pcbrt(a): return np.sign(a)*np.power(np.abs(a), 1/3)
def pln(a): return np.log(np.abs(a)+1e-12)
def plog10(a): return np.log10(np.abs(a)+1e-12)
def px2(a): return np.power(a,2.0)
def px3(a): return np.power(a,3.0)
def pnop(a): return a

FUN = {'+':(2, lambda a,b:a+b), '-':(2, lambda a,b:a-b), '*':(2, lambda a,b:a*b), '/':(2, pdiv),
       'X2':(1,px2),'X3':(1,px3),'Sqrt':(1,psqrt),'3Rt':(1,pcbrt),'Ln':(1,pln),'Log':(1,plog10),'Nop':(1,pnop)}
FUN_ARITY = {k:v[0] for k,v in FUN.items()}
VAR = ['A1','A2','A3','A4']


# In[ ]:


# ---------- decoder ----------
class Node:
    __slots__=("s","ch")
    def __init__(self,s): self.s=s; self.ch=[]

def decode_gene(sym):
    root = Node(sym[0])
    need = FUN_ARITY.get(root.s,0)
    q=[(root, need)]
    i=1
    while q:
        node, ar = q.pop(0)
        for _ in range(ar):
            t = sym[i] if i<len(sym) else 'A1'
            i += 1
            c=Node(t); node.ch.append(c)
            q.append((c, FUN_ARITY.get(t,0)))
    return root

def eval_node(node, X, C):
    s=node.s
    if s in FUN:
        ar,fn = FUN[s]
        if ar==1: return fn(eval_node(node.ch[0],X,C))
        else:     return fn(eval_node(node.ch[0],X,C), eval_node(node.ch[1],X,C))
    else:
        if s in VAR: return X[:, VAR.index(s)]
        if s.startswith('C') and s[1:].isdigit():
            idx=int(s[1:]); return np.full(X.shape[0], float(C[idx]), float)
        return np.zeros(X.shape[0], float)

def expr_str(sym):
    def rec(n):
        s=n.s
        if s in FUN:
            ar,_=FUN[s]
            if ar==1:
                inner=rec(n.ch[0])
                return inner if s=='Nop' else f"{s}({inner})"
            else:
                return f"({rec(n.ch[0])} {s} {rec(n.ch[1])})"
        else: return s
    return rec(decode_gene(sym))

def substitute_consts(expr, consts):
    def repl(m): return str(consts[int(m.group(1))])
    return re.sub(r'C([0-4])', repl, expr)



# In[ ]:


# ---------- individual & operators ----------
def rand_gene(HEAD_SYM, TAIL_SYM, HEAD, TAIL):
    return [random.choice(HEAD_SYM) for _ in range(HEAD)] + [random.choice(TAIL_SYM) for _ in range(TAIL)]

class Ind:
    __slots__=("genes","consts","fit","a","b")
    def __init__(self, genes, consts):
        self.genes = genes
        self.consts = consts
        self.fit=1e99; self.a=1.0; self.b=0.0
    def clone(self):
        c=Ind([g.copy() for g in self.genes], self.consts.copy())
        c.fit=self.fit; c.a=self.a; c.b=self.b
        return c

def gene_out(ind, X):
    G=np.zeros((X.shape[0], len(ind.genes)), float)
    for g in range(len(ind.genes)):
        v = eval_node(decode_gene(ind.genes[g]), X, ind.consts[g])
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0); v = np.clip(v, -1e8, 1e8)
        G[:,g] = v
    return G

def raw_sum(ind, X): return np.sum(gene_out(ind, X), axis=1)

def fit_lin_scale(y, r):
    A = np.vstack([r, np.ones_like(r)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])

def rmse(y,yh): return float(np.sqrt(np.mean((y-yh)**2)))
def R2(y,yh):
    ssr=float(np.sum((y-yh)**2)); sst=float(np.sum((y-np.mean(y))**2))
    return 1.0-ssr/sst if sst>0 else float('nan')

def fitness(ind, X_tr, y_tr):
    r = raw_sum(ind, X_tr)
    a,b = fit_lin_scale(y_tr, r)
    ind.a, ind.b = a, b
    return rmse(y_tr, a*r + b)

def tour(pop, k=3):  # tournament
    idx=np.random.choice(len(pop), size=k, replace=False)
    cand=[pop[i] for i in idx]
    return min(cand, key=lambda z:z.fit)

def mutate(ind, HEAD, TAIL, HEAD_SYM, TAIL_SYM, MUT, CLOW, CHIGH):
    c=ind.clone()
    for g in range(len(c.genes)):
        for i in range(HEAD+TAIL):
            if random.random()<MUT:
                c.genes[g][i] = random.choice(HEAD_SYM) if i<HEAD else random.choice(TAIL_SYM)
        # numeric constants
        for k in range(c.consts.shape[1]):
            if random.random()<0.10:
                if random.random()<0.5:
                    c.consts[g,k] = int(np.clip(c.consts[g,k] + random.choice([-1,1]), CLOW, CHIGH))
                else:
                    c.consts[g,k] = int(np.random.randint(CLOW, CHIGH+1))
    return c

def inversion(ind, HEAD, INV):
    c=ind.clone()
    for g in range(len(c.genes)):
        if random.random()<INV:
            s=random.randint(0,HEAD-1); e=random.randint(s,HEAD-1)
            c.genes[g][s:e+1]=list(reversed(c.genes[g][s:e+1]))
    return c

def recomb_one(p1,p2, GENE_SZ):
    c1, c2 = p1.clone(), p2.clone()
    f1, f2 = sum(c1.genes, []), sum(c2.genes, [])
    pt = random.randint(1,len(f1)-1)
    nf1 = f1[:pt]+f2[pt:]; nf2=f2[:pt]+f1[pt:]
    def unflat(f): return [f[i*GENE_SZ:(i+1)*GENE_SZ] for i in range(len(c1.genes))]
    c1.genes, c2.genes = unflat(nf1), unflat(nf2)
    return c1, c2

def recomb_two(p1,p2, GENE_SZ):
    c1, c2 = p1.clone(), p2.clone()
    f1, f2 = sum(c1.genes, []), sum(c2.genes, [])
    a = random.randint(0,len(f1)-2); b = random.randint(a+1,len(f1)-1)
    nf1 = f1[:a]+f2[a:b]+f1[b:]; nf2=f2[:a]+f1[a:b]+f2[b:]
    def unflat(f): return [f[i*GENE_SZ:(i+1)*GENE_SZ] for i in range(len(c1.genes))]
    c1.genes, c2.genes = unflat(nf1), unflat(nf2)
    return c1, c2

def gene_recomb(p1,p2):
    c1, c2 = p1.clone(), p2.clone()
    for g in range(len(c1.genes)):
        if random.random()<0.2333:
            c1.genes[g], c2.genes[g] = c2.genes[g].copy(), c1.genes[g].copy()
    return c1, c2


# In[ ]:


# ---------- training loop ----------
def train_once(X_tr, y_tr, HEAD, TAIL, GENES, POP, N_GEN, MUT, INV,
               REC1, REC2, GREC, N_CONST, CLOW, CHIGH, seed):
    random.seed(seed); np.random.seed(seed)
    GENE_SZ = HEAD+TAIL
    HEAD_SYM = list(FUN.keys()) + VAR + [f'C{i}' for i in range(N_CONST)]
    TAIL_SYM = VAR + [f'C{i}' for i in range(N_CONST)]
    pop=[]
    for _ in range(POP):
        genes=[rand_gene(HEAD_SYM, TAIL_SYM, HEAD, TAIL) for _ in range(GENES)]
        consts=np.vstack([np.random.randint(CLOW,CHIGH+1,size=N_CONST) for _ in range(GENES)]).astype(float)
        pop.append(Ind(genes,consts))
    for i in pop: i.fit = fitness(i, X_tr, y_tr)
    best = min(pop, key=lambda z:z.fit).clone()
    for g in range(N_GEN):
        nxt=[best.clone()]  # elitism
        while len(nxt)<POP:
            r=random.random()
            if r<REC1 and len(nxt)<=POP-2:
                p1,p2=tour(pop), tour(pop); c1,c2=recomb_one(p1,p2, GENE_SZ)
                c1=inversion(mutate(c1,HEAD,TAIL,HEAD_SYM,TAIL_SYM,MUT,CLOW,CHIGH), HEAD, INV)
                c2=inversion(mutate(c2,HEAD,TAIL,HEAD_SYM,TAIL_SYM,MUT,CLOW,CHIGH), HEAD, INV)
                nxt.extend([c1,c2])
            elif r<REC1+REC2 and len(nxt)<=POP-2:
                p1,p2=tour(pop), tour(pop); c1,c2=recomb_two(p1,p2, GENE_SZ)
                c1=inversion(mutate(c1,HEAD,TAIL,HEAD_SYM,TAIL_SYM,MUT,CLOW,CHIGH), HEAD, INV)
                c2=inversion(mutate(c2,HEAD,TAIL,HEAD_SYM,TAIL_SYM,MUT,CLOW,CHIGH), HEAD, INV)
                nxt.extend([c1,c2])
            elif r<REC1+REC2+GREC and len(nxt)<=POP-2:
                p1,p2=tour(pop), tour(pop); c1,c2=gene_recomb(p1,p2)
                c1=inversion(mutate(c1,HEAD,TAIL,HEAD_SYM,TAIL_SYM,MUT,CLOW,CHIGH), HEAD, INV)
                c2=inversion(mutate(c2,HEAD,TAIL,HEAD_SYM,TAIL_SYM,MUT,CLOW,CHIGH), HEAD, INV)
                nxt.extend([c1,c2])
            else:
                p=tour(pop); c=inversion(mutate(p,HEAD,TAIL,HEAD_SYM,TAIL_SYM,MUT,CLOW,CHIGH), HEAD, INV); nxt.append(c)
        for i in nxt: i.fit = fitness(i, X_tr, y_tr)
        pop = nxt
        cur=min(pop, key=lambda z:z.fit)
        if cur.fit < best.fit: best = cur.clone()
    # gather result
    def predict(ind, X): return ind.a*np.sum(gene_out(ind,X),axis=1)+ind.b
    yhat_tr = predict(best, X_tr)
    res = dict(
        a=best.a, b=best.b,
        genes=[expr_str(g) for g in best.genes],
        genes_arrays=best.genes,
        consts=best.consts.astype(int).tolist(),
        r2_tr=R2(y_tr,yhat_tr),
        rmse_tr=rmse(y_tr,yhat_tr)
    )
    return res, best

def predict_from(best, X):
    # rebuild individual-like
    class Dummy: pass
    d=Dummy(); d.genes=best['genes_arrays']; d.consts=np.array(best['consts']).astype(float); d.a=float(best['a']); d.b=float(best['b'])
    def gene_outX(d, X):
        G=np.zeros((X.shape[0], len(d.genes)), float)
        for g in range(len(d.genes)):
            v = eval_node(decode_gene(d.genes[g]), X, d.consts[g])
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0); v = np.clip(v,-1e8,1e8)
            G[:,g]=v
        return G
    raw = np.sum(gene_outX(d,X), axis=1)
    return d.a*raw + d.b


# In[ ]:


# ---------- drawing ----------
class TNode:
    def __init__(self, s): self.s=s; self.ch=[]
def decode_nodes(symbols):
    root=TNode(symbols[0]); need=FUN_ARITY.get(root.s,0)
    q=[(root,need)]; i=1
    while q:
        node, ar = q.pop(0)
        for _ in range(ar):
            sym = symbols[i] if i<len(symbols) else 'A1'
            i+=1; c=TNode(sym); node.ch.append(c)
            q.append((c, FUN_ARITY.get(sym,0)))
    return root
class DNode:
    def __init__(self,label,children=None): self.label=label; self.children=children or []
def to_draw(root, consts):
    lab=root.s
    if lab.startswith('C') and lab[1:].isdigit():
        lab=f"{lab}={consts[int(lab[1:])]}"
    d=DNode(lab)
    for c in root.ch: d.children.append(to_draw(c,consts))
    return d

def layout_tree(root, x_spacing=2.2, y_spacing=2.0):
    def count_leaves(n): return 1 if not n.children else sum(count_leaves(c) for c in n.children)
    pos=[]
    def assign(n,x_left,depth):
        if not n.children:
            xc=x_left+0.5; pos.append((n,xc,-depth)); return x_left+1.0
        cur=x_left; centers=[]
        for c in n.children:
            cur=assign(c,cur,depth+1); centers.append(pos[-1][1])
        xc=sum(centers)/len(centers); pos.append((n,xc,-depth)); return cur
    assign(root,0.0,0); xs=[p[1] for p in pos]; minx=min(xs)
    return [(n,(x-minx)*x_spacing,y*y_spacing) for n,x,y in pos]

def draw_tree(symbols, consts, title, out_png):
    root=decode_nodes(symbols)
    droot=to_draw(root, consts)
    pos=layout_tree(droot)
    fig, ax = plt.subplots(figsize=(10,6), dpi=160); ax.set_axis_off()
    node_pos={n:(x,y) for n,x,y in pos}
    for n,x,y in pos:
        for c in n.children:
            x2,y2=node_pos[c]
            ax.add_patch(FancyArrowPatch((x,y-0.12),(x2,y2+0.12),arrowstyle="-",mutation_scale=10,linewidth=1.1))
    for n,x,y in pos:
        box=FancyBboxPatch((x-0.9,y-0.4),1.8,0.8,boxstyle="round,pad=0.02,rounding_size=0.06",linewidth=1.1,edgecolor="black",facecolor="white")
        ax.add_patch(box); ax.text(x,y,n.label,ha="center",va="center",fontsize=10)
    ax.set_title(title, fontsize=12, pad=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)


# In[ ]:


# ---------- main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--excel", type=str, default="CFST (GEP).xlsx")
    ap.add_argument("--sheet", type=int, default=0)
    ap.add_argument("--gens", type=int, default=15000)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--head", type=int, default=7)
    ap.add_argument("--tail", type=int, default=8)
    ap.add_argument("--genes", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--outdir", type=str, default=".")
    args=ap.parse_args()

    df=pd.read_excel(args.excel, sheet_name=args.sheet)
    df = df.drop(columns=[c for c in df.columns if str(c).lower().startswith('unnamed')], errors='ignore')
    X = df[['A1','A2','A3','A4']].to_numpy(float)
    y = (df['Pcc (N)'].to_numpy(float))/1000.0

    X_tr, y_tr = X[:140], y[:140]
    X_va, y_va = X[140:200], y[140:200]

    HEAD, TAIL, GENES = args.head, args.tail, args.genes
    GENE_SZ = HEAD + TAIL
    POP, N_GEN = args.pop, args.gens
    MUT=0.044; INV=0.05; REC1=0.2333; REC2=0.2333; GREC=0.2333
    N_CONST=5; CLOW=-10; CHIGH=10

    best=None; best_ind=None
    for s in range(args.seeds):
        res, ind = train_once(X_tr, y_tr, HEAD, TAIL, GENES, POP, N_GEN, MUT, INV, REC1, REC2, GREC, N_CONST, CLOW, CHIGH, seed=2025+s)
        # compute validation metrics
        y_tr_hat = predict_from(res, X_tr)
        y_va_hat = predict_from(res, X_va)
        res['r2_tr']=float(R2(y_tr, y_tr_hat)); res['rmse_tr']=float(np.sqrt(np.mean((y_tr-y_tr_hat)**2)))
        res['r2_va']=float(R2(y_va, y_va_hat)); res['rmse_va']=float(np.sqrt(np.mean((y_va-y_va_hat)**2)))
        if (best is None) or (res['r2_va']>best['r2_va']):
            best, best_ind = res, ind

    # final predictions
    y_tr_hat = predict_from(best, X_tr)
    y_va_hat = predict_from(best, X_va)

    # exports
    os.makedirs(args.outdir, exist_ok=True)
    json_path=os.path.join(args.outdir,"GEP_best_model.json")
    open(json_path,"w",encoding="utf-8").write(json.dumps(best, ensure_ascii=False, indent=2))

    # Formula text
    G_sub=[substitute_consts(best['genes'][i], best['consts'][i]) for i in range(GENES)]
    formula=f"Pcc_hat [kN] = {best['a']:.12f} * ( G1 + G2 + G3 ) + {best['b']:.12f}"
    lines=[
        "فرمول نهایی (با مقیاس‌گذاری خطی):",
        formula,
        "که در آن:"
    ]
    for i in range(GENES):
        lines.append(f"G{i+1} = {best['genes'][i]}")
        lines.append(f"G{i+1} (پس از جایگذاری ثابت‌ها) = {G_sub[i]}")
    lines += [
        "",
        f"Train: R2={float(R2(y_tr,y_tr_hat)):.6f}, RMSE={float(np.sqrt(np.mean((y_tr-y_tr_hat)**2))):.6f} kN",
        f"Valid: R2={float(R2(y_va,y_va_hat)):.6f}, RMSE={float(np.sqrt(np.mean((y_va-y_va_hat)**2))):.6f} kN"
    ]
    txt_path=os.path.join(args.outdir,"GEP_final_report.txt")
    open(txt_path,"w",encoding="utf-8").write("\n".join(lines))

    # ETs
    for i in range(GENES):
        p=os.path.join(args.outdir, f"ET_G{i+1}.png")
        draw_tree(best['genes_arrays'][i], best['consts'][i], f"Expression Tree — G{i+1}", p)

    print("Saved:")
    print(" JSON:", json_path)
    print(" TXT :", txt_path)
    for i in range(GENES):
        print(" PNG :", os.path.join(args.outdir, f"ET_G{i+1}.png"))

if __name__=="__main__":
    main()





