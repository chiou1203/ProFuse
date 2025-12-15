#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"
import torch
import hashlib
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, count_render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import numpy as np
import faiss
import gc
import time
from autoencoder.model import Autoencoder

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---- Clusters360 loader (global proposal map) ----
def _load_clusters360_map(langfeat_dir):
    """
    Returns:
        lut: dict[str -> dict[int -> int]]  (view_name -> {label_id -> component_id})
        n_comp: int  (#global components)
    or (None, 0) if the file does not exist.
    """
    import numpy as np, os
    path = os.path.join(langfeat_dir, "clusters360_global.npz")
    if not os.path.exists(path):
        return None, 0
    data = np.load(path, allow_pickle=True)
    lut = data["view_to_label_to_comp"].item()  # dict stored in npz
    # normalize keys and ints
    lut = {str(vname): {int(k): int(v) for k, v in d.items()} for vname, d in lut.items()}
    all_ids = [cid for d in lut.values() for cid in d.values()]
    n_comp = (max(all_ids) + 1) if len(all_ids) else 0
    return lut, n_comp


def _comp_id_from_clusters360(lut, view_name, label_id):
    """Return component id or -1 if this (view, label) has no global component."""
    d = lut.get(str(view_name), None)
    if d is None:
        return -1
    return int(d.get(int(label_id), -1))


def update_voting_mat(result_dict, language_feature_mask, gt_language_feature, contribution, ids, args):
    # Select only locations where Mask is True
    mask_idx = language_feature_mask.squeeze(0).nonzero(as_tuple=True)
    
    # Get the ID and contributions of the gaussians who contributed from that location
    contrib = contribution[mask_idx]  # shape: [N, 100]
    ray_ids = ids[mask_idx]  # shape: [N, 100]
    gt_feats = gt_language_feature[:, mask_idx[0], mask_idx[1]]  # shape: [3, N]
    
    _, indices = torch.topk(contrib, args.topk, dim=1)
    ray_ids = torch.gather(ray_ids, 1, indices)
    
    # Filter only valid contributions (non-1 IDs and non-0 contributions)
    valid_mask = (ray_ids != -1)
    ray_ids = ray_ids[valid_mask].view(-1)  # shape: [M] (valid Gaussian ID)
    gt_feats = gt_feats.T.unsqueeze(1).repeat(1, args.topk, 1)[valid_mask]  # shape: [M, 3]

    unique_ids = torch.unique(ray_ids)
    
    for uid in unique_ids:
        mask = ray_ids == uid
        if uid.item() not in result_dict:
            result_dict[uid.item()] = [gt_feats[mask]]
        else:
            result_dict[uid.item()].append(gt_feats[mask])

    return result_dict


def compute_average(features):
    averaged_tensor = features.mean(dim=0).unsqueeze(0)  # 평균 계산
    averaged_tensor = averaged_tensor / (averaged_tensor.norm(dim=-1, keepdim=True) + 1e-9)
    return averaged_tensor



@torch.no_grad()

def majority_voting(gaussians, scene, pipe, background, dataset, args):

    # --- paths & cameras ---
    lf_path = "/" + os.path.join(*dataset.lf_path.split('/')[:-1], "language_features")
    viewpoint_stack = scene.getTrainCameras().copy()

    
    #clusters360_lut, clusters360_ncomp = _load_clusters360_map(lf_path)
    #use_clusters360 = (clusters360_lut is not None and clusters360_ncomp > 0)

    # --- allocs for proposal→Gaussian accumulation ---
    G = gaussians._opacity.shape[0]
    feat_dim = 512  # rows in *_f.npy
    gauss_feat_sum = torch.zeros((G, feat_dim), dtype=torch.float32, device="cuda")
    gauss_w_sum    = torch.zeros((G,),          dtype=torch.float32, device="cuda")

    # === per-view pass ===
    for i in tqdm(range(len(viewpoint_stack))):
        vp = viewpoint_stack[i]
        language_feature_name = os.path.join(lf_path, vp.image_name)

        # load language features & segmentation for this view
        feature_map = torch.from_numpy(np.load(language_feature_name + "_f.npy")).to("cuda", dtype=torch.float32)  # [M,512]
        seg_map_np  = np.load(language_feature_name + "_s.npy", allow_pickle=True)
        seg_map     = torch.from_numpy(seg_map_np[dataset.feature_level]).to("cuda", dtype=torch.long).unsqueeze(0)  # [1,H,W]

        torch.cuda.synchronize(); t0=time.time()
        # renderer outputs (top-K ids & contributions)
        render_pkg   = count_render(vp, gaussians, pipe, background)
        ids          = render_pkg["per_pixel_gaussian_ids"].detach()            # [H,W,K], cuda, int
        contribution = render_pkg["per_pixel_gaussian_contributions"].detach()  # [H,W,K], cuda, float
        torch.cuda.synchronize(); #print("T_render:", time.time()-t0)

        

        # --- NEW: sanitize ids to [-1, G-1] ---
        G = gaussians._opacity.shape[0]
        invalid_any = (ids < 0) | (ids >= G)
        if invalid_any.any():
            ids[invalid_any] = -1

        # try cluster mode for this view
        cluster_npz_path = language_feature_name + "_clusters.npz"
        use_clusters = bool(getattr(args, "use_clusters", False)) and os.path.isfile(cluster_npz_path)

        if use_clusters:
            # ========= [C360] optional global proposal map for this view =========
            # Tries clusters360_global.npz; if unavailable, falls back to per-ref *_clusters.npz (your original path).
            _use_c360 = False
            _Cmax_global = 0
            _c360_v2l2c = None
            _view_key = str(vp.image_name)

            c360_path = os.path.join(lf_path, "clusters360_global.npz")
            if os.path.exists(c360_path):
                _c360 = np.load(c360_path, allow_pickle=True)
                _c360_v2l2c = _c360["view_to_label_to_comp"].item()
                if _view_key in _c360_v2l2c:
                    _use_c360 = True
                    # global Cmax once from the whole dict
                    for _d in _c360_v2l2c.values():
                        if len(_d):
                            _Cmax_global = max(_Cmax_global, max(_d.values()) + 1)

            # ---- load cluster mapping (global-360 or per-ref local) ----
            if _use_c360:
                lut = _c360_v2l2c[_view_key]  # dict: {label_id -> component_id}
                _max_label = max(lut.keys()) if len(lut) else -1
                label2cluster = -np.ones((_max_label + 1,), dtype=np.int64)
                for _k, _v in lut.items():
                    if _k >= 0:
                        if _k >= label2cluster.shape[0]:
                            label2cluster = np.pad(label2cluster, (0, _k + 1 - label2cluster.shape[0]), constant_values=-1)
                        label2cluster[_k] = int(_v)
                label2cluster_t = torch.from_numpy(label2cluster).to("cuda", dtype=torch.long)
                # dummy holders to avoid touching downstream prints that expect these names
                ref_cluster_of_label = np.array([-1], dtype=np.int64)
            else:
                # ---- ORIGINAL per-reference mapping (unchanged) ----
                C = np.load(cluster_npz_path, allow_pickle=True)
                ref_label_vals       = C["ref_label_values"].astype(np.int64)         # [R]
                ref_cluster_of_label = C["ref_cluster_of_label"].astype(np.int64)     # [R]
                if ref_label_vals.size == 0:
                    # nothing to do in this view
                    continue
                max_label = int(ref_label_vals.max())
                label2cluster = -np.ones((max_label + 1,), dtype=np.int64)
                label2cluster[ref_label_vals] = ref_cluster_of_label
                label2cluster_t = torch.from_numpy(label2cluster).to("cuda", dtype=torch.long)

            # ---------------- per-pixel cluster id map (H,W) in [-1..] ----------------
            seg_lab  = seg_map[0]                # [H,W]
            valid_px = (seg_lab >= 0)
            cluster_id_map = torch.full_like(seg_lab, -1, dtype=torch.long)
            in_range = valid_px & (seg_lab < label2cluster_t.numel())
            if in_range.any():
                mapped = label2cluster_t[seg_lab[in_range]]
                ok = (mapped >= 0)
                cluster_id_map[in_range] = mapped
                valid_px = torch.zeros_like(valid_px)
                valid_px[in_range] = ok
            else:
                continue  # no valid pixels for this view

            # --- Gather pixels (optionally stride) ---
            rr, cc = valid_px.nonzero(as_tuple=True)
            stride = max(1, int(getattr(args, "pixel_stride", 1)))
            if stride > 1 and rr.numel() > 0:
                rr = rr[::stride]; cc = cc[::stride]
            if rr.numel() == 0:
                continue

            # --- Normalize shapes to [N,K] ---
            ids_px     = ids[rr, cc]              # [N,K] or [N]
            contrib_px = contribution[rr, cc]     # [N,K] or [N]
            if ids_px.ndim == 1:      ids_px     = ids_px.unsqueeze(-1)
            if contrib_px.ndim == 1:  contrib_px = contrib_px.unsqueeze(-1)
            K = ids_px.shape[-1]

            # --- Compute per-mask "mass" BEFORE building cluster feats ---
            mass_px = contrib_px.sum(dim=1)        # [N]
            lab_px  = seg_lab[rr, cc]              # [N]
            L_feat  = feature_map.shape[0]
            in_feat = (lab_px >= 0) & (lab_px < L_feat)
            if not in_feat.any():
                continue
            lab_px  = lab_px[in_feat]
            mass_px = mass_px[in_feat]
            lab_mass = torch.zeros((L_feat,), dtype=torch.float32, device="cuda")
            lab_mass.index_add_(0, lab_px, mass_px)

            L_feat = feature_map.shape[0]
            L_l2c  = label2cluster_t.numel()
            if L_l2c < L_feat:
                pad = torch.full((L_feat - L_l2c,),
                                 -1,
                                 device=label2cluster_t.device,
                                 dtype=label2cluster_t.dtype)
                label2cluster_t = torch.cat([label2cluster_t, pad], dim=0)
            elif L_l2c > L_feat:
                # (rare) clamp down to the feature bank size
                label2cluster_t = label2cluster_t[:L_feat]

            # --- Build per-cluster feature weighted by lab_mass ---
            l2c = label2cluster_t[:L_feat]
            valid_lab = (l2c >= 0) & (lab_mass > 0)
            if not valid_lab.any():
                continue
            labels = torch.arange(L_feat, device="cuda", dtype=torch.long)[valid_lab]
            cids   = l2c[valid_lab]
            w_lab  = lab_mass[valid_lab]

            # **Cmax**: global if using 360, else original local count
            Cmax = (_Cmax_global if _use_c360 else int(ref_cluster_of_label.max()) + 1)
            cluster_feat_sum = torch.zeros((Cmax, feat_dim), dtype=torch.float32, device="cuda")
            cluster_w_sum    = torch.zeros((Cmax,),          dtype=torch.float32, device="cuda")
            cluster_feat_sum.index_add_(0, cids, feature_map[labels] * w_lab.unsqueeze(1))
            cluster_w_sum.index_add_(0, cids, w_lab)
            nz = cluster_w_sum > 0
            cluster_feat = torch.zeros_like(cluster_feat_sum)
            cluster_feat[nz] = cluster_feat_sum[nz] / (cluster_w_sum[nz].unsqueeze(1) + 1e-8)
            cluster_feat = (cluster_feat / cluster_feat.norm(dim=1, keepdim=True).clamp_min(1e-9)).half()

            # ----- push proposal features to Gaussians via per-pixel top-K -----
            valid_ids = (ids_px >= 0)
            if not valid_ids.any():
                continue

            w_topk  = torch.where(valid_ids, contrib_px, torch.zeros_like(contrib_px)).reshape(-1)     # [N*K]
            g_topk  = torch.where(valid_ids, ids_px,      torch.zeros_like(ids_px)).reshape(-1).long() # [N*K]
            cid_rep = cluster_id_map[rr, cc].reshape(-1).repeat_interleave(K)

            keep = w_topk > 0
            if not keep.any():
                continue
            w_topk  = w_topk[keep]
            g_topk  = g_topk[keep]
            cid_rep = cid_rep[keep]

            torch.cuda.synchronize(); t1=time.time()

            # --------- your SpMM block (unchanged) ---------
            eps = float(getattr(args, "eps_contrib", 0.0))
            if eps > 0.0:
                keep2 = (w_topk >= eps)
                if not keep2.any():
                    continue
                w_topk  = w_topk[keep2]
                g_topk  = g_topk[keep2]
                cid_rep = cid_rep[keep2]

            if Cmax <= 0 or w_topk.numel() == 0:
                continue

            def _spmm_block(gids, cids, wvals, c_start, c_end):
                c_local = cids - c_start
                idx = torch.stack([gids, c_local], dim=0)
                W_blk = torch.sparse_coo_tensor(
                    idx, wvals, size=(G, c_end - c_start),
                    dtype=torch.float32, device="cuda"
                ).coalesce().to_sparse_csr()
                if (i % 20) == 0:
                    m_c_blk = torch.zeros((c_end - c_start,), dtype=torch.float32, device="cuda")
                    m_c_blk.index_add_(0, c_local, wvals)
                    if m_c_blk.numel() > 0 and float(m_c_blk.sum()) > 0:
                        vals, _ = torch.sort(m_c_blk, descending=True)
                        top = min(10, vals.numel())
                        share = float(vals[:top].sum().item() / (vals.sum().item() + 1e-8))
                        print(f"[DIAG][mass] {vp.image_name}: block[{c_start}:{c_end}) "
                              f"C_blk={c_end-c_start} top10_share={share:.3f}")
                feat_blk = cluster_feat[c_start:c_end].float()
                out_blk   = torch.matmul(W_blk, feat_blk)
                wsum_blk  = torch.matmul(W_blk, torch.ones((c_end - c_start,), device="cuda", dtype=torch.float32))
                gauss_feat_sum.add_(out_blk)
                gauss_w_sum.add_(wsum_blk)
        
            blk = int(getattr(args, "spmm_cluster_block", 0))
            if blk > 0:
                for c0 in range(0, Cmax, blk):
                    c1 = min(c0 + blk, Cmax)
                    m  = (cid_rep >= c0) & (cid_rep < c1)
                    if not m.any():
                        continue
                    _spmm_block(g_topk[m], cid_rep[m], w_topk[m], c0, c1)
            else:
                idx = torch.stack([g_topk, cid_rep], dim=0)
                W_csr = torch.sparse_coo_tensor(
                    idx, w_topk, size=(G, Cmax),
                    dtype=torch.float32, device="cuda"
                ).coalesce().to_sparse_csr()
                out  = torch.matmul(W_csr, cluster_feat.float())
                if (i % 20) == 0:
                    m_c = torch.zeros((Cmax,), dtype=torch.float32, device="cuda")
                    m_c.index_add_(0, cid_rep, w_topk)
                    if m_c.numel() > 0 and float(m_c.sum()) > 0:
                        vals, _ = torch.sort(m_c, descending=True)
                        top = min(10, vals.numel())
                        share = float(vals[:top].sum().item() / (vals.sum().item() + 1e-8))
                        print(f"[DIAG][mass] {vp.image_name}: C={Cmax} top10_share={share:.3f}")
                wsum = torch.matmul(W_csr, torch.ones((Cmax,), device="cuda", dtype=torch.float32))
                gauss_feat_sum.add_(out)
                gauss_w_sum.add_(wsum)

            torch.cuda.synchronize()

            if i % 20 == 0:
                n_valid = int(valid_px.sum().item())
                if _use_c360:
                    n_ref_labels = int(len(_c360_v2l2c.get(_view_key, {})))
                    print(f"[MV] {vp.image_name}: clusters={Cmax} ref_labels={n_ref_labels} valid_px={n_valid}")
                else:
                    n_ref_labels = int(C["ref_label_values"].size)
                    print(f"[MV] {vp.image_name}: clusters={int(ref_cluster_of_label.max())+1} "
                          f"ref_labels={n_ref_labels} valid_px={n_valid}")
            continue  # next view


        # =======================
        # fallback: mask-level path (no clusters for this view)
        # =======================
        seg_lab  = seg_map[0]                 # [H,W]
        valid_px = (seg_lab >= 0)
        rr, cc   = valid_px.nonzero(as_tuple=True)
        if rr.numel() == 0:
            continue

        contrib_px = contribution[rr, cc]     # [N,K]   (FIXED)
        ids_px     = ids[rr, cc]              # [N,K]   (FIXED)
        if ids_px.ndim == 1:     ids_px     = ids_px.unsqueeze(-1)
        if contrib_px.ndim == 1: contrib_px = contrib_px.unsqueeze(-1)
        K = ids_px.shape[-1]

        lab_px   = seg_lab[rr, cc]
        in_feat  = (lab_px >= 0) & (lab_px < feature_map.shape[0])
        if not in_feat.any():
            continue
        rr, cc    = rr[in_feat], cc[in_feat]
        lab_px    = lab_px[in_feat]
        ids_ck    = ids_px[in_feat]        # [P,K]
        contrib_ck= contrib_px[in_feat]    # [P,K]
        feats     = feature_map[lab_px]    # [P,512]

        for k in range(K):
            gk = ids_ck[:, k]
            validk = (gk >= 0) & (gk < G)  # --- NEW: bound check ---
            if not validk.any():
                continue
            wk = contrib_ck[validk, k]
            fk = feats[validk].float()
            fk.mul_(wk.unsqueeze(1))
            gauss_feat_sum.index_add_(0, gk[validk], fk)
            gauss_w_sum.index_add_(0,  gk[validk], wk)

        # Normalize to [N,K]
    # --- finalize Gaussian features ---
    nz = gauss_w_sum > 0
    averaged_tensor = torch.zeros((G, feat_dim), dtype=torch.float32, device="cuda")
    averaged_tensor[nz] = gauss_feat_sum[nz] / (gauss_w_sum[nz].unsqueeze(1) + 1e-8)
    nrm = averaged_tensor.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    averaged_tensor[nz] = averaged_tensor[nz] / nrm[nz]
    invalid_gaussians = ~nz

    # ==== DIAG: save a small pre-PQ sample and a global checksum ====
    try:
        
        run_tag = getattr(args, "run_tag", None) or "run"
        out_dir = scene.model_path  # same folder you already save to
        os.makedirs(out_dir, exist_ok=True)

        # Fixed random sample of Gaussians (so runs are comparable)
        torch.manual_seed(12345)
        S = min(10000, max(1000, G // 100))   # 1% or up to 10k
        sample_idx = torch.randperm(G, device="cuda")[:S]
        F_sample = averaged_tensor.index_select(0, sample_idx).detach().cpu().numpy().astype("float16")

        # A tiny checksum so you can diff quickly without loading the .npy
        h = hashlib.sha1(averaged_tensor[:1000].detach().cpu().numpy().tobytes()).hexdigest()[:12]

        np.save(os.path.join(out_dir, f"prePQ_{run_tag}_ids.npy"),
            sample_idx.detach().cpu().numpy().astype("int32"))
        np.save(os.path.join(out_dir, f"prePQ_{run_tag}_float16.npy"), F_sample)
        with open(os.path.join(out_dir, f"prePQ_{run_tag}_sha1.txt"), "w") as f:
            f.write(h + "\n")
    
        print(f"[DIAG][prePQ] saved sample S={S}, sha1={h}")
    except Exception as ex:
        print(f"[DIAG][prePQ] skip ({ex})")

    try:
        wcpu = gauss_w_sum.detach().cpu().numpy()
        srt = np.sort(wcpu)[::-1]
        def frac(k): 
            k = min(k, srt.size); 
            return float(srt[:k].sum() / (srt.sum() + 1e-8))
        print("[DIAG][mass] gauss_w_sum: "
              f"p50={np.percentile(wcpu,50):.4e} p90={np.percentile(wcpu,90):.4e} "
              f"top1%={frac(max(1,int(0.01*len(srt)))):.3f} top0.1%={frac(max(1,int(0.001*len(srt)))):.3f}")
    except Exception as ex:
        pass


    if args.use_pq:
        index = faiss.read_index(args.pq_index)
        if args.faiss_add:
            index.add(averaged_tensor.cpu().numpy())
        codes = index.sa_encode(averaged_tensor.cpu().numpy())
        averaged_tensor = torch.ByteTensor(codes).to("cuda")
        averaged_tensor[invalid_gaussians,:] = -1

        # ==== DIAG: dump codes for later A/B diff (small file, ~15–20MB) ====
        try:
            run_tag = getattr(args, "run_tag", None) or "run"
            np.save(os.path.join(scene.model_path, f"pqcodes_{run_tag}.npy"), codes)
            print(f"[DIAG][PQ] codes saved: pqcodes_{run_tag}.npy shape={codes.shape}")
        except Exception as ex:
            print(f"[DIAG][PQ] skip save ({ex})")

    return averaged_tensor



def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if opt.include_feature:
        if not checkpoint:
            raise ValueError("checkpoint missing!!!!!")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        if len(model_params) == 12 and opt.include_feature:
            first_iter = 0
        gaussians.restore(model_params, opt)
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    maj_feat = majority_voting(gaussians, scene, pipe, background, dataset, args)
    
    gaussians._language_feature = maj_feat
    
    iteration = 0

    if (iteration in saving_iterations):
        print("\n[ITER {}] Saving Gaussians".format(iteration))
        scene.save(iteration)

    if (iteration in checkpoint_iterations):
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        torch.save((gaussians.capture(opt.include_feature), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, testing_iterations, scene : Scene, renderFunc, renderArgs):
    # Report test and samples of training set
    if iteration in testing_iterations:
        print(f'testing for iter {iteration}')
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=55555)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[0])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[0])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[0])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser.add_argument("--name_extra", type=str, default = None)
    parser.add_argument("--mode", type=str, default = "mean")
    parser.add_argument("--topk", type=int, default = 1)
    
    parser.add_argument("--use_pq", action="store_true")
    parser.add_argument("--pq_index", type=str, default=None)
    
    parser.add_argument('--encoder_dims',
                        nargs = '+',
                        type=int,
                        default=[256, 128, 64, 32, 3],
                        )
    parser.add_argument('--decoder_dims',
                        nargs = '+',
                        type=int,
                        default=[16, 32, 64, 128, 256, 256, 512],
                        )
    parser.add_argument("--faiss_add", action="store_true")
    parser.add_argument("--use_mask_consensus", action="store_true", help="Use per-mask consensus _c.npy when present")
    parser.add_argument("--mask_consensus_pow", type=float, default=1.0, help="Exponent on mask weight w(M)^rho")
    parser.add_argument("--mask_consensus_amb_delta", type=float, default=0.02, help="Only apply on pixels with (top1-top2)<delta")
    parser.add_argument("--mask_consensus_drop_pct", type=float, default=0.0, help="Drop bottom q%% masks per image (0 disables)")
    parser.add_argument("--use_clusters", action="store_true",
                    help="Use EDGS cluster proposals (<img>_clusters.npz) for cluster-level aggregation.")
    parser.add_argument("--pairs_per_chunk", type=int, default=40000,
    help="Chunk size for (gaussian, cluster) pair accumulation to avoid OOM")
    parser.add_argument("--pixel_stride", type=int, default=1,
    help="Subsample stride for pixels inside clustered masks (1=no subsample)")
    parser.add_argument("--accum_weight_floor", type=float, default=0.0)
    parser.add_argument("--spmm_cluster_block", type=int, default=0,
                    help="If >0, process clusters in blocks of this size to bound sparse matrix size.")
    parser.add_argument("--eps_contrib", type=float, default=0.0,
                    help="Drop top-K entries with contribution < eps before forming sparse matrix.")
    parser.add_argument("--use_clusters360", action="store_true",
    help="Use global 360° proposals (clusters360_global.npz) to map per-view labels to global component IDs")


    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print(args)
    index = faiss.read_index(args.pq_index)


    try:
        args.modelpath = args.model_path + f"_{str(args.feature_level)}_{args.name_extra}_topk{args.topk}_weight_{index.coarsecode_size()+index.code_size}"
    except :
        args.model_path = args.model_path + f"_{str(args.feature_level)}_{args.name_extra}_topk{args.topk}_weight_{index.code_size}"

    if args.use_pq:
        if args.pq_index is None:
            raise ValueError("PQ index file is not provided.")
        lp._language_features_name = "language_features_pq"

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")


