"""SnapKV Compression Pipeline Benchmark v2

All experiments use SnapKV (no-softmax, matching real SGLang impl).
Grid: seq_len × compression_ratio × cpu_prefix_match_ratio.

Experiments:
  1. SnapKV algorithm sub-operation breakdown
  2. Full pipeline: CPU prefix + extend → FA → compress → write
  3. CPU prefix pipeline vs full recompute (QKV projection from hidden states)
  4. Peak GPU memory comparison

Usage:
    python benchmark/kv_compression/bench_snapkv_v2.py
    python benchmark/kv_compression/bench_snapkv_v2.py --exp 1
"""
import argparse, math, sys
from typing import Dict, List
import torch, torch.nn.functional as F
sys.path.insert(0, ".")
from benchmark.kv_compression.profiler.phase_timer import PhaseTimer

# ── SnapKV core (no softmax, matching real impl) ─────────────────────────────

def snapkv_compress(k, v, q, ntk, nh, nkh, timer, ws, label):
    sl, hd = k.shape[0], k.shape[2]
    g = nh // nkh; ws = min(ws, sl)
    with timer.phase(label):
        with timer.phase(f"{label}.slice"):
            qw = q[-ws:]; kp = k[:-ws] if sl > ws else k[:0]
        if kp.shape[0] == 0:
            return torch.arange(sl, device=k.device).unsqueeze(0).expand(nkh, -1)
        with timer.phase(f"{label}.transpose"):
            qt = qw.transpose(0,1).contiguous(); kt = kp.transpose(0,1).contiguous()
        with timer.phase(f"{label}.gqa_expand"):
            if g > 1: kt = kt.repeat_interleave(g, dim=0)
        with timer.phase(f"{label}.qk_matmul"):
            a = torch.matmul(qt, kt.transpose(-2,-1)) / math.sqrt(hd)
        with timer.phase(f"{label}.sync"):
            torch.cuda.current_stream().synchronize()
        with timer.phase(f"{label}.pool_sum"):
            s = a.sum(dim=1)
            if g > 1: s = s.view(nkh, g, -1).sum(dim=1)
            if s.shape[-1] > 5:
                s = F.max_pool1d(s.unsqueeze(0), kernel_size=5, padding=2, stride=1).squeeze(0)
        with timer.phase(f"{label}.topk"):
            nk = max(1, ntk - ws)
            _, idx = s.topk(nk, dim=-1)
            idx = torch.sort(idx, dim=-1).values
            wi = torch.arange(sl-ws, sl, device=k.device).unsqueeze(0).expand(nkh,-1)
            keep = torch.cat([idx, wi], dim=-1)
    return keep

def scatter_write(k, v, keep, rk, rv, rs, nkh, timer, label):
    with timer.phase(label):
        nk = keep.shape[1]
        for h in range(nkh):
            rk[rs[:nk], h] = k[keep[h], h]
            rv[rs[:nk], h] = v[keep[h], h]

# ── Helper ────────────────────────────────────────────────────────────────────

def avg_s(all_s):
    a = {}
    for k in all_s[0]: a[k] = sum(d[k] for d in all_s) / len(all_s)
    return a

def fmt_ms(v): return f"{v:.3f}" if v < 10 else f"{v:.1f}"

# ── Experiment 1: SnapKV algorithm breakdown ──────────────────────────────────

def run_exp1(args):
    print("\n" + "="*90)
    print("EXP 1: SnapKV Algorithm Sub-operation Breakdown")
    print("  (pure compression algorithm, no pipeline overhead)")
    print("="*90)
    dt, dev = torch.bfloat16, "cuda"
    sls = [int(x) for x in args.seq_lens.split(",")]
    rats = [float(x) for x in args.ratios.split(",")]
    ops = ["slice","transpose","gqa_expand","qk_matmul","sync","pool_sum","topk"]
    hdr = f"{'seq':>7} {'ratio':>5} {'TOTAL':>8}"
    for o in ops: hdr += f" {o:>12}"
    print(hdr); print("-"*len(hdr))

    for sl in sls:
        k = torch.randn(sl, args.nkh, args.hd, dtype=dt, device=dev)
        v = torch.randn(sl, args.nkh, args.hd, dtype=dt, device=dev)
        q = torch.randn(sl, args.nh, args.hd, dtype=dt, device=dev)
        for r in rats:
            ntk = max(1, int(sl*(1-r)))
            ss = []
            for i in range(args.warmup + args.repeat):
                t = PhaseTimer()
                snapkv_compress(k, v, q, ntk, args.nh, args.nkh, t, args.ws, "c")
                s = t.summary()
                if i >= args.warmup: ss.append(s)
            a = avg_s(ss)
            total = a.get("c", 0)
            line = f"{sl:>7} {r:>5.1f} {total:>7.3f}ms"
            for o in ops:
                ms = a.get(f"c.{o}", 0)
                pct = ms/total*100 if total > 0 else 0
                line += f" {ms:>7.3f}({pct:>3.0f}%)"
            print(line)
        print()

# ── Experiment 2: Full pipeline breakdown ─────────────────────────────────────

def run_exp2(args):
    print("\n" + "="*90)
    print("EXP 2: Full Pipeline Breakdown (CPU prefix → GlobalKVPool → FA → compress → write)")
    print("="*90)
    dt, dev = torch.bfloat16, "cuda"
    sls = [int(x) for x in args.seq_lens.split(",")]
    rats = [float(x) for x in args.ratios.split(",")]
    pfxs = [float(x) for x in args.prefix_ratios.split(",")]

    for sl in sls:
        for r in rats:
            ntk = max(1, int(sl*(1-r)))
            psz = sl + 256
            for pr in pfxs:
                plen = int(sl * pr); elen = sl - plen
                # Allocate
                kc = torch.randn(plen, args.nkh, args.hd, dtype=dt)
                vc = torch.randn(plen, args.nkh, args.hd, dtype=dt)
                ke = torch.randn(elen, args.nkh, args.hd, dtype=dt, device=dev)
                ve = torch.randn(elen, args.nkh, args.hd, dtype=dt, device=dev)
                qe = torch.randn(max(elen, args.ws), args.nh, args.hd, dtype=dt, device=dev)
                pk = torch.zeros(psz, args.nkh, args.hd, dtype=dt, device=dev)
                pv = torch.zeros(psz, args.nkh, args.hd, dtype=dt, device=dev)
                rk = torch.zeros(psz, args.nkh, args.hd, dtype=dt, device=dev)
                rv = torch.zeros(psz, args.nkh, args.hd, dtype=dt, device=dev)
                sl_t = torch.arange(sl, device=dev)
                rs_t = torch.arange(ntk, device=dev)

                ss = []
                for i in range(args.warmup + args.repeat):
                    t = PhaseTimer()
                    # Phase 1: CPU transfer + pool assemble
                    with t.phase("p1_transfer"):
                        kg = kc.to(dev, non_blocking=False)
                        vg = vc.to(dev, non_blocking=False)
                    with t.phase("p1_pool_write"):
                        pk[sl_t[:plen]] = kg; pv[sl_t[:plen]] = vg
                        if elen > 0: pk[sl_t[plen:sl]] = ke; pv[sl_t[plen:sl]] = ve
                    with t.phase("p1_pool_read"):
                        kf = pk[sl_t[:sl]].clone(); vf = pv[sl_t[:sl]].clone()
                    # Phase 2: FA
                    with t.phase("p2_fa"):
                        g = args.nh // args.nkh
                        qt2 = qe.transpose(0,1).unsqueeze(0)
                        kt2 = kf.transpose(0,1).unsqueeze(0)
                        vt2 = vf.transpose(0,1).unsqueeze(0)
                        if g > 1: kt2 = kt2.repeat_interleave(g, dim=1); vt2 = vt2.repeat_interleave(g, dim=1)
                        out = F.scaled_dot_product_attention(qt2, kt2, vt2, is_causal=True)
                        del qt2, kt2, vt2, out
                    # Phase 3: Compress
                    keep = snapkv_compress(kf, vf, qe, ntk, args.nh, args.nkh, t, args.ws, "p3_compress")
                    scatter_write(kf, vf, keep, rk, rv, rs_t, args.nkh, t, "p3_write")
                    s = t.summary()
                    if i >= args.warmup: ss.append(s)

                a = avg_s(ss)
                t1 = a.get("p1_transfer",0); pw = a.get("p1_pool_write",0)
                pr_ = a.get("p1_pool_read",0); fa = a.get("p2_fa",0)
                comp = a.get("p3_compress",0); wr = a.get("p3_write",0)
                total = t1+pw+pr_+fa+comp+wr
                print(f"  seq={sl:>6} r={r} pfx={pr:.0%} | "
                      f"total={total:>7.2f}ms  xfer={t1:.2f} pool_w={pw:.2f} pool_r={pr_:.2f} "
                      f"FA={fa:.2f} compress={comp:.2f} write={wr:.2f}")

                del kc,vc,ke,ve,qe,pk,pv,rk,rv; torch.cuda.empty_cache()
        print()

# ── Experiment 3: CPU prefix vs full recompute (QKV projection) ───────────────

def run_exp3(args):
    print("\n" + "="*90)
    print("EXP 3: CPU Prefix Pipeline vs Full Recompute (QKV projection from hidden states)")
    print("  Recompute = hidden_states @ W_q/W_k/W_v on GPU (simulating model forward)")
    print("="*90)
    dt, dev = torch.bfloat16, "cuda"
    sls = [int(x) for x in args.seq_lens.split(",")]
    rats = [float(x) for x in args.ratios.split(",")]
    pfxs = [float(x) for x in args.prefix_ratios.split(",")]
    hidden_dim = args.nh * args.hd  # e.g. 28*128=3584

    # Projection weights (shared, simulating one layer)
    W_q = torch.randn(hidden_dim, args.nh * args.hd, dtype=dt, device=dev) * 0.02
    W_k = torch.randn(hidden_dim, args.nkh * args.hd, dtype=dt, device=dev) * 0.02
    W_v = torch.randn(hidden_dim, args.nkh * args.hd, dtype=dt, device=dev) * 0.02

    for sl in sls:
        for r in rats:
            ntk = max(1, int(sl*(1-r)))
            psz = sl + 256

            for pr in pfxs:
                plen = int(sl * pr); elen = sl - plen

                # ── Strategy A: CPU prefix pipeline (current SGLang) ──
                kc = torch.randn(plen, args.nkh, args.hd, dtype=dt)
                vc = torch.randn(plen, args.nkh, args.hd, dtype=dt)
                ke = torch.randn(elen, args.nkh, args.hd, dtype=dt, device=dev)
                ve = torch.randn(elen, args.nkh, args.hd, dtype=dt, device=dev)
                qe = torch.randn(max(elen, args.ws), args.nh, args.hd, dtype=dt, device=dev)
                pk = torch.zeros(psz, args.nkh, args.hd, dtype=dt, device=dev)
                pv = torch.zeros(psz, args.nkh, args.hd, dtype=dt, device=dev)
                rk = torch.zeros(psz, args.nkh, args.hd, dtype=dt, device=dev)
                rv = torch.zeros(psz, args.nkh, args.hd, dtype=dt, device=dev)
                sl_t = torch.arange(sl, device=dev)
                rs_t = torch.arange(ntk, device=dev)

                ss_a = []
                for i in range(args.warmup + args.repeat):
                    t = PhaseTimer()
                    with t.phase("A.transfer"):
                        kg = kc.to(dev); vg = vc.to(dev)
                    with t.phase("A.pool_io"):
                        pk[sl_t[:plen]] = kg; pv[sl_t[:plen]] = vg
                        if elen > 0: pk[sl_t[plen:sl]] = ke; pv[sl_t[plen:sl]] = ve
                        kf = pk[sl_t[:sl]].clone(); vf = pv[sl_t[:sl]].clone()
                    with t.phase("A.fa"):
                        g = args.nh // args.nkh
                        qt2 = qe.transpose(0,1).unsqueeze(0)
                        kt2 = kf.transpose(0,1).unsqueeze(0)
                        vt2 = vf.transpose(0,1).unsqueeze(0)
                        if g > 1: kt2 = kt2.repeat_interleave(g,dim=1); vt2 = vt2.repeat_interleave(g,dim=1)
                        F.scaled_dot_product_attention(qt2, kt2, vt2, is_causal=True)
                        del qt2, kt2, vt2
                    keep = snapkv_compress(kf, vf, qe, ntk, args.nh, args.nkh, t, args.ws, "A.compress")
                    scatter_write(kf, vf, keep, rk, rv, rs_t, args.nkh, t, "A.write")
                    s = t.summary()
                    if i >= args.warmup: ss_a.append(s)
                avg_a = avg_s(ss_a)

                del kc,vc,ke,ve,qe,pk,pv; torch.cuda.empty_cache()

                # ── Strategy B: Full recompute (QKV projection from hidden states) ──
                hidden = torch.randn(sl, hidden_dim, dtype=dt, device=dev)

                ss_b = []
                for i in range(args.warmup + args.repeat):
                    t = PhaseTimer()
                    with t.phase("B.qkv_proj"):
                        q_full = (hidden @ W_q).view(sl, args.nh, args.hd)
                        k_full = (hidden @ W_k).view(sl, args.nkh, args.hd)
                        v_full = (hidden @ W_v).view(sl, args.nkh, args.hd)
                    with t.phase("B.fa"):
                        g = args.nh // args.nkh
                        qt2 = q_full.transpose(0,1).unsqueeze(0)
                        kt2 = k_full.transpose(0,1).unsqueeze(0)
                        vt2 = v_full.transpose(0,1).unsqueeze(0)
                        if g > 1: kt2 = kt2.repeat_interleave(g,dim=1); vt2 = vt2.repeat_interleave(g,dim=1)
                        F.scaled_dot_product_attention(qt2, kt2, vt2, is_causal=True)
                        del qt2, kt2, vt2
                    keep = snapkv_compress(k_full, v_full, q_full, ntk, args.nh, args.nkh, t, args.ws, "B.compress")
                    scatter_write(k_full, v_full, keep, rk, rv, rs_t, args.nkh, t, "B.write")
                    s = t.summary()
                    if i >= args.warmup: ss_b.append(s)
                avg_b = avg_s(ss_b)

                del hidden; torch.cuda.empty_cache()

                tA = sum(v for k2,v in avg_a.items() if k2.startswith("A.") and k2.count(".")==1)
                tB = sum(v for k2,v in avg_b.items() if k2.startswith("B.") and k2.count(".")==1)
                aX = avg_a.get("A.transfer",0); aP = avg_a.get("A.pool_io",0)
                aF = avg_a.get("A.fa",0); aC = avg_a.get("A.compress",0)
                bQ = avg_b.get("B.qkv_proj",0); bF = avg_b.get("B.fa",0)
                bC = avg_b.get("B.compress",0)
                ratio = tA/tB if tB > 0 else 0

                print(f"  seq={sl:>6} r={r} pfx={pr:.0%} | "
                      f"A(cpu_pfx)={tA:>7.2f}ms [xfer={aX:.2f} pool={aP:.2f} fa={aF:.2f} comp={aC:.2f}] | "
                      f"B(recomp)={tB:>7.2f}ms [proj={bQ:.2f} fa={bF:.2f} comp={bC:.2f}] | "
                      f"A/B={ratio:.2f}x")

                del rk, rv; torch.cuda.empty_cache()
        print()

# ── Experiment 4: Memory comparison ───────────────────────────────────────────

def run_exp4(args):
    print("\n" + "="*90)
    print("EXP 4: Peak GPU Memory — CPU Prefix vs Full Recompute")
    print("="*90)
    dt, dev = torch.bfloat16, "cuda"
    sls = [int(x) for x in args.seq_lens.split(",")]
    hidden_dim = args.nh * args.hd

    W_q = torch.randn(hidden_dim, args.nh*args.hd, dtype=dt, device=dev)*0.02
    W_k = torch.randn(hidden_dim, args.nkh*args.hd, dtype=dt, device=dev)*0.02
    W_v = torch.randn(hidden_dim, args.nkh*args.hd, dtype=dt, device=dev)*0.02
    w_mem = (W_q.nelement()+W_k.nelement()+W_v.nelement())*2/1024**2

    for sl in sls:
        r = 0.5; pr = 0.8
        ntk = max(1, int(sl*(1-r))); plen = int(sl*pr); elen = sl-plen; psz = sl+256

        # A: CPU prefix
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        kc=torch.randn(plen,args.nkh,args.hd,dtype=dt)
        vc=torch.randn(plen,args.nkh,args.hd,dtype=dt)
        ke=torch.randn(elen,args.nkh,args.hd,dtype=dt,device=dev)
        ve=torch.randn(elen,args.nkh,args.hd,dtype=dt,device=dev)
        qe=torch.randn(max(elen,args.ws),args.nh,args.hd,dtype=dt,device=dev)
        pk=torch.zeros(psz,args.nkh,args.hd,dtype=dt,device=dev)
        pv=torch.zeros(psz,args.nkh,args.hd,dtype=dt,device=dev)
        rk=torch.zeros(psz,args.nkh,args.hd,dtype=dt,device=dev)
        rv=torch.zeros(psz,args.nkh,args.hd,dtype=dt,device=dev)
        sl_t=torch.arange(sl,device=dev); rs_t=torch.arange(ntk,device=dev)
        t=PhaseTimer()
        kg=kc.to(dev);vg=vc.to(dev)
        pk[sl_t[:plen]]=kg;pv[sl_t[:plen]]=vg
        if elen>0: pk[sl_t[plen:sl]]=ke;pv[sl_t[plen:sl]]=ve
        kf=pk[sl_t[:sl]].clone();vf=pv[sl_t[:sl]].clone()
        g=args.nh//args.nkh
        qt2=qe.transpose(0,1).unsqueeze(0);kt2=kf.transpose(0,1).unsqueeze(0);vt2=vf.transpose(0,1).unsqueeze(0)
        if g>1: kt2=kt2.repeat_interleave(g,dim=1);vt2=vt2.repeat_interleave(g,dim=1)
        F.scaled_dot_product_attention(qt2,kt2,vt2,is_causal=True)
        del qt2,kt2,vt2
        keep=snapkv_compress(kf,vf,qe,ntk,args.nh,args.nkh,t,args.ws,"c")
        scatter_write(kf,vf,keep,rk,rv,rs_t,args.nkh,t,"w")
        torch.cuda.synchronize()
        peakA = torch.cuda.max_memory_allocated()/1024**2
        del kc,vc,ke,ve,qe,pk,pv,rk,rv,kf,vf; torch.cuda.empty_cache()

        # B: Full recompute
        torch.cuda.reset_peak_memory_stats()
        hidden=torch.randn(sl,hidden_dim,dtype=dt,device=dev)
        rk2=torch.zeros(psz,args.nkh,args.hd,dtype=dt,device=dev)
        rv2=torch.zeros(psz,args.nkh,args.hd,dtype=dt,device=dev)
        q_f=(hidden@W_q).view(sl,args.nh,args.hd)
        k_f=(hidden@W_k).view(sl,args.nkh,args.hd)
        v_f=(hidden@W_v).view(sl,args.nkh,args.hd)
        qt2=q_f.transpose(0,1).unsqueeze(0);kt2=k_f.transpose(0,1).unsqueeze(0);vt2=v_f.transpose(0,1).unsqueeze(0)
        if g>1: kt2=kt2.repeat_interleave(g,dim=1);vt2=vt2.repeat_interleave(g,dim=1)
        F.scaled_dot_product_attention(qt2,kt2,vt2,is_causal=True)
        del qt2,kt2,vt2
        t2=PhaseTimer()
        keep2=snapkv_compress(k_f,v_f,q_f,ntk,args.nh,args.nkh,t2,args.ws,"c")
        scatter_write(k_f,v_f,keep2,rk2,rv2,rs_t,args.nkh,t2,"w")
        torch.cuda.synchronize()
        peakB = torch.cuda.max_memory_allocated()/1024**2
        del hidden,rk2,rv2,q_f,k_f,v_f; torch.cuda.empty_cache()

        print(f"  seq={sl:>6} | A(cpu_pfx)={peakA:>8.1f}MB | B(recomp)={peakB:>8.1f}MB (incl weights {w_mem:.0f}MB) | diff={peakA-peakB:>+8.1f}MB")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-lens", default="4096,8192,16384,32768,65536")
    p.add_argument("--ratios", default="0.3,0.5,0.7")
    p.add_argument("--prefix-ratios", default="0.5,0.8,1.0")
    p.add_argument("--nkh", type=int, default=4, help="num_kv_heads")
    p.add_argument("--nh", type=int, default=28, help="num_q_heads")
    p.add_argument("--hd", type=int, default=128, help="head_dim")
    p.add_argument("--ws", type=int, default=64, help="window_size")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--exp", default="all", help="all,1,2,3,4")
    a = p.parse_args()
    print(f"Config: kv_heads={a.nkh}, q_heads={a.nh}, head_dim={a.hd}, window={a.ws}")
    if a.exp in ("all","1"): run_exp1(a)
    if a.exp in ("all","2"): run_exp2(a)
    if a.exp in ("all","3"): run_exp3(a)
    if a.exp in ("all","4"): run_exp4(a)

if __name__ == "__main__":
    main()
