import torch
import torch.nn.functional as F


def _check_lf_tensor(lf: torch.Tensor):
    if not isinstance(lf, torch.Tensor):
        raise TypeError("lf must be a torch.Tensor")
    if lf.ndim not in (4, 5):
        raise ValueError(f"lf must have shape [u, v, h, w] or [u, v, c, h, w], got {tuple(lf.shape)}")
    if not lf.is_floating_point():
        raise TypeError("lf must be a floating tensor")
    if lf.device.type != "cuda":
        raise ValueError("This implementation is intended for CUDA tensors")


def _make_patch_positions(h: int, w: int, patch_size: int, stride: int, device):
    ys = torch.arange(0, h - patch_size + 1, stride, device=device, dtype=torch.long)
    xs = torch.arange(0, w - patch_size + 1, stride, device=device, dtype=torch.long)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    positions = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)  # [P,2]
    gy = ys.numel()
    gx = xs.numel()
    return positions, gy, gx


def _extract_all_patches(img: torch.Tensor, patch_size: int, pad: int = 0):

    if pad > 0:
        img = F.pad(img.unsqueeze(0), (pad, pad, pad, pad), mode="reflect").squeeze(0)
    c, h, w = img.shape
    k = patch_size
    patches = F.unfold(img.unsqueeze(0), kernel_size=(k, k), stride=1)  # [1, c*k*k, L]
    patches = patches.squeeze(0).transpose(0, 1).contiguous().view(-1, c, k, k)
    grid_w = w - k + 1
    return patches, grid_w


def _gather_patches_by_positions(patches: torch.Tensor, grid_w: int, positions: torch.Tensor):

    idx = positions[:, 0] * grid_w + positions[:, 1]
    return patches[idx.long()]


def _build_offsets(disp_radius: int, device):
    vals = []
    for dy in range(-disp_radius, disp_radius + 1):
        for dx in range(-disp_radius, disp_radius + 1):
            vals.append([dy, dx])
    offsets = torch.tensor(vals, device=device, dtype=torch.long)  # [S,2]
    zero_idx = torch.nonzero((offsets[:, 0] == 0) & (offsets[:, 1] == 0), as_tuple=False).item()
    return offsets, zero_idx


def _search_shifts_and_gather(
    lf: torch.Tensor,
    positions: torch.Tensor,
    patch_size: int,
    disp_radius: int,
    search_chunk: int,
):

    device = lf.device
    u, v, c, h, w = lf.shape
    k = patch_size
    P = positions.shape[0]
    ref_u, ref_v = u // 2, v // 2

    offsets, zero_idx = _build_offsets(disp_radius, device)
    S = offsets.shape[0]

    ref_img = lf[ref_u, ref_v]
    ref_all, ref_grid_w = _extract_all_patches(ref_img, k, pad=0)
    ref_patches = _gather_patches_by_positions(ref_all, ref_grid_w, positions)  # [P,c,k,k]

    patches4d = torch.empty((P, u, v, c, k, k), device=device, dtype=lf.dtype)
    shift_ids = torch.empty((P, u, v), device=device, dtype=torch.long)

    patches4d[:, ref_u, ref_v] = ref_patches
    shift_ids[:, ref_u, ref_v] = zero_idx

    for uu in range(u):
        for vv in range(v):
            if uu == ref_u and vv == ref_v:
                continue

            img = lf[uu, vv]
            all_patches, grid_w = _extract_all_patches(img, k, pad=disp_radius)

            best_view_patches = torch.empty((P, c, k, k), device=device, dtype=lf.dtype)
            best_view_shift_ids = torch.empty((P,), device=device, dtype=torch.long)

            for s in range(0, P, search_chunk):
                e = min(s + search_chunk, P)
                pos = positions[s:e]  # [Pc,2]
                Pc = pos.shape[0]

                shifted = pos.unsqueeze(0) + offsets.unsqueeze(1) + disp_radius  # [S,Pc,2]
                idx = shifted[..., 0] * grid_w + shifted[..., 1]  # [S,Pc]

                cand = all_patches[idx.reshape(-1)].view(S, Pc, c, k, k)  # [S,Pc,c,k,k]
                dist = ((cand - ref_patches[s:e].unsqueeze(0)) ** 2).mean(dim=(2, 3, 4))  # [S,Pc]
                best = dist.argmin(dim=0)  # [Pc]

                cand_perm = cand.permute(1, 0, 2, 3, 4).contiguous()  # [Pc,S,c,k,k]
                best_view_patches[s:e] = cand_perm[torch.arange(Pc, device=device), best]
                best_view_shift_ids[s:e] = best

            patches4d[:, uu, vv] = best_view_patches
            shift_ids[:, uu, vv] = best_view_shift_ids

    return patches4d, shift_ids, offsets


def _gather_with_shift_ids(
    lf: torch.Tensor,
    positions: torch.Tensor,
    patch_size: int,
    disp_radius: int,
    shift_ids: torch.Tensor,
    offsets: torch.Tensor,
    gather_chunk: int,
):
    device = lf.device
    u, v, c, h, w = lf.shape
    k = patch_size
    P = positions.shape[0]
    ref_u, ref_v = u // 2, v // 2

    patches4d = torch.empty((P, u, v, c, k, k), device=device, dtype=lf.dtype)

    for uu in range(u):
        for vv in range(v):
            img = lf[uu, vv]
            all_patches, grid_w = _extract_all_patches(img, k, pad=disp_radius)

            out_view = torch.empty((P, c, k, k), device=device, dtype=lf.dtype)

            for s in range(0, P, gather_chunk):
                e = min(s + gather_chunk, P)
                pos = positions[s:e]  # [Pc,2]
                sid = shift_ids[s:e, uu, vv]  # [Pc]
                off = offsets[sid]  # [Pc,2]

                shifted = pos + off + disp_radius  # [Pc,2]
                idx = shifted[:, 0] * grid_w + shifted[:, 1]
                out_view[s:e] = all_patches[idx.long()]

            patches4d[:, uu, vv] = out_view

    return patches4d


def _group_similar_vectorized(
    patches4d: torch.Tensor,
    gy: int,
    gx: int,
    stride: int,
    sim_radius: int,
    max_group_size: int,
    group_pool: int = 4,
):
    device = patches4d.device
    P, u, v, c, k, _ = patches4d.shape

    feat = patches4d.mean(dim=(1, 2))  # [P,c,k,k]
    if group_pool is not None and group_pool > 0 and (feat.shape[-1] != group_pool or feat.shape[-2] != group_pool):
        feat = F.adaptive_avg_pool2d(feat, (group_pool, group_pool))
    D = feat.shape[1] * feat.shape[2] * feat.shape[3]

    feat_map = feat.reshape(gy, gx, D).permute(2, 0, 1).unsqueeze(0).contiguous()  # [1,D,gy,gx]

    r = max(0, sim_radius // stride)
    kw = 2 * r + 1
    K = kw * kw

    feat_pad = F.pad(feat_map, (r, r, r, r), mode="replicate")
    cand_feat = F.unfold(feat_pad, kernel_size=(kw, kw), stride=1)  # [1, D*K, P]
    cand_feat = cand_feat.squeeze(0).transpose(0, 1).contiguous().view(P, K, D)  # [P,K,D]

    ref_feat = feat.reshape(P, D).unsqueeze(1)  # [P,1,D]
    dist = ((cand_feat - ref_feat) ** 2).mean(dim=-1)  # [P,K]

    idx_map = torch.arange(P, device=device, dtype=torch.float32).view(1, 1, gy, gx)
    idx_pad = F.pad(idx_map, (r, r, r, r), mode="constant", value=-1.0)
    cand_idx = F.unfold(idx_pad, kernel_size=(kw, kw), stride=1).squeeze(0).transpose(0, 1).contiguous().long()  # [P,K]

    valid = cand_idx >= 0
    cand_idx = cand_idx.clamp(min=0)
    dist = dist.masked_fill(~valid, float("inf"))

    topk = min(max_group_size, K)
    _, order = torch.topk(dist, k=topk, dim=1, largest=False)
    group_idx = cand_idx.gather(1, order)  # [P,topk]
    return group_idx


def _hard_threshold_groups(groups: torch.Tensor, sigma: float, lambda_hard: float):
    x = groups.permute(0, 2, 3, 4, 5, 6, 1).contiguous()  # [B,u,v,c,k,k,N]
    coeff = torch.fft.fftn(x, dim=(1, 2, 4, 5, 6), norm="ortho")
    thr = lambda_hard * sigma
    mask = coeff.abs() >= thr
    coeff = coeff * mask
    est = torch.fft.ifftn(coeff, dim=(1, 2, 4, 5, 6), norm="ortho").real
    est = est.permute(0, 6, 1, 2, 3, 4, 5).contiguous()  # [B,N,u,v,c,k,k]

    nz = mask.sum(dim=(1, 2, 3, 4, 5, 6)).to(groups.dtype)
    weight = 1.0 / torch.clamp(nz, min=1.0)
    return est, weight


def _wiener_groups(noisy_groups: torch.Tensor, basic_groups: torch.Tensor, sigma: float):
    y = noisy_groups.permute(0, 2, 3, 4, 5, 6, 1).contiguous()   # [B,u,v,c,k,k,N]
    b = basic_groups.permute(0, 2, 3, 4, 5, 6, 1).contiguous()

    Y = torch.fft.fftn(y, dim=(1, 2, 4, 5, 6), norm="ortho")
    Bc = torch.fft.fftn(b, dim=(1, 2, 4, 5, 6), norm="ortho")

    power = Bc.abs() ** 2
    wien = power / (power + sigma * sigma + 1e-12)
    est = torch.fft.ifftn(wien * Y, dim=(1, 2, 4, 5, 6), norm="ortho").real
    est = est.permute(0, 6, 1, 2, 3, 4, 5).contiguous()

    weight = 1.0 / torch.clamp((wien ** 2).sum(dim=(1, 2, 3, 4, 5, 6)).real.to(noisy_groups.dtype), min=1e-12)
    return est, weight


def _aggregate_chunk_scatter(
    out_sum_flat: torch.Tensor,   # [u,v,c,HW]
    out_wgt_flat: torch.Tensor,   # [u,v,1,HW]
    est_group: torch.Tensor,      # [B,N,u,v,c,k,k]
    group_idx: torch.Tensor,      # [B,N]
    positions: torch.Tensor,      # [P,2]
    disparities: torch.Tensor,    # [P,u,v,2]
    h: int,
    w: int,
    patch_size: int,
    weights: torch.Tensor,        # [B]
):
    device = est_group.device
    B, N, U, V, C, K, _ = est_group.shape

    pos = positions[group_idx]          # [B,N,2]
    disp = disparities[group_idx]       # [B,N,U,V,2]

    yy = pos[..., 0].unsqueeze(-1).unsqueeze(-1) + disp[..., 0]  # [B,N,U,V]
    xx = pos[..., 1].unsqueeze(-1).unsqueeze(-1) + disp[..., 1]  # [B,N,U,V]

    yy = yy.clamp(0, h - K)
    xx = xx.clamp(0, w - K)

    ky = torch.arange(K, device=device, dtype=torch.long).view(1, 1, 1, 1, K, 1)
    kx = torch.arange(K, device=device, dtype=torch.long).view(1, 1, 1, 1, 1, K)

    lin = (yy[..., None, None] + ky) * w + (xx[..., None, None] + kx)  # [B,N,U,V,K,K]
    idx_uv = lin.permute(2, 3, 0, 1, 4, 5).contiguous().view(U, V, -1)  # [U,V,M]

    w_group = weights.view(B, 1, 1, 1, 1, 1)

    for cc in range(C):
        src = (est_group[:, :, :, :, cc] * w_group).permute(2, 3, 0, 1, 4, 5).contiguous().view(U, V, -1)
        out_sum_flat[:, :, cc].scatter_add_(2, idx_uv, src)

    src_w = w_group.expand(B, N, U, V, K, K).permute(2, 3, 0, 1, 4, 5).contiguous().view(U, V, -1)
    out_wgt_flat[:, :, 0].scatter_add_(2, idx_uv, src_w)


def _lfbm5d_stage_fast(
    noisy_lf: torch.Tensor,
    match_lf: torch.Tensor,
    sigma: float,
    patch_size: int,
    stride: int,
    disp_radius: int,
    sim_radius: int,
    max_group_size: int,
    lambda_hard: float,
    stage: str,
    search_chunk: int,
    group_pool: int,
    transform_chunk: int,
):
    device = noisy_lf.device
    U, V, C, H, W = noisy_lf.shape

    positions, gy, gx = _make_patch_positions(H, W, patch_size, stride, device=device)
    P = positions.shape[0]


    match_patches4d, shift_ids, offsets = _search_shifts_and_gather(
        match_lf, positions, patch_size, disp_radius, search_chunk=search_chunk
    )  


    if stage == "wiener":
        noisy_patches4d = _gather_with_shift_ids(
            noisy_lf, positions, patch_size, disp_radius, shift_ids, offsets, gather_chunk=search_chunk
        )
    else:
        noisy_patches4d = match_patches4d


    group_idx = _group_similar_vectorized(
        match_patches4d,
        gy=gy,
        gx=gx,
        stride=stride,
        sim_radius=sim_radius,
        max_group_size=max_group_size,
        group_pool=group_pool,
    )  # [P,N]

    disparities = offsets[shift_ids]  

    out_sum = torch.zeros_like(noisy_lf)
    out_wgt = torch.zeros((U, V, 1, H, W), device=device, dtype=noisy_lf.dtype)

    out_sum_flat = out_sum.view(U, V, C, H * W)
    out_wgt_flat = out_wgt.view(U, V, 1, H * W)

    for s in range(0, P, transform_chunk):
        e = min(s + transform_chunk, P)
        idx_chunk = group_idx[s:e]  # [B,N]

        basic_groups = match_patches4d[idx_chunk]  # [B,N,U,V,C,K,K]

        if stage == "hard":
            est_group, weights = _hard_threshold_groups(basic_groups, sigma=sigma, lambda_hard=lambda_hard)
        elif stage == "wiener":
            noisy_groups = noisy_patches4d[idx_chunk]
            est_group, weights = _wiener_groups(noisy_groups, basic_groups, sigma=sigma)
        else:
            raise ValueError(f"Unknown stage: {stage}")

        _aggregate_chunk_scatter(
            out_sum_flat=out_sum_flat,
            out_wgt_flat=out_wgt_flat,
            est_group=est_group,
            group_idx=idx_chunk,
            positions=positions,
            disparities=disparities,
            h=H,
            w=W,
            patch_size=patch_size,
            weights=weights,
        )


    out = out_sum / (out_wgt + 1e-8)

    low_mask = out_wgt < 1e-6  # [U,V,1,H,W]

    if low_mask.any():
        pool_k = 5
        pad = pool_k // 2

        sum_2d = out_sum.contiguous().view(U * V * C, 1, H, W)
        wgt_2d = out_wgt.expand(U, V, C, H, W).contiguous().view(U * V * C, 1, H, W)

        sum_pool = F.avg_pool2d(sum_2d, kernel_size=pool_k, stride=1, padding=pad)
        wgt_pool = F.avg_pool2d(wgt_2d, kernel_size=pool_k, stride=1, padding=pad)

        fill = (sum_pool / (wgt_pool + 1e-8)).view(U, V, C, H, W)

        low_mask_c = low_mask.expand(U, V, C, H, W)
        out = torch.where(low_mask_c, fill, out)


        still_low = (wgt_pool.view(U, V, C, H, W) < 1e-6) & low_mask_c
        out = torch.where(still_low, match_lf, out)

    return out


@torch.no_grad()
def LFBM5D_denoiser(
    lf: torch.Tensor,
    sigma: float = 25 / 255.0,
    patch_size: int = 8,
    stride_hard: int = 4,
    stride_wiener: int = 4,
    disp_radius_hard: int = 2,
    disp_radius_wiener: int = 2,
    sim_radius_hard: int = 12,
    sim_radius_wiener: int = 12,
    max_group_size_hard: int = 8,
    max_group_size_wiener: int = 8,
    lambda_hard: float = 2.7,
    search_chunk: int = 4096,
    group_pool: int = 4,
    transform_chunk: int = 32,
    clip_output: bool = False,
):
    _check_lf_tensor(lf)


    squeeze_channel = False
    if lf.ndim == 4:
        lf = lf.unsqueeze(2)
        squeeze_channel = True

    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if lf.shape[-2] < patch_size or lf.shape[-1] < patch_size:
        raise ValueError("patch_size must be <= h and <= w")

    basic = _lfbm5d_stage_fast(
        noisy_lf=lf,
        match_lf=lf,
        sigma=sigma,
        patch_size=patch_size,
        stride=stride_hard,
        disp_radius=disp_radius_hard,
        sim_radius=sim_radius_hard,
        max_group_size=max_group_size_hard,
        lambda_hard=lambda_hard,
        stage="hard",
        search_chunk=search_chunk,
        group_pool=group_pool,
        transform_chunk=transform_chunk,
    )

    denoised = _lfbm5d_stage_fast(
        noisy_lf=lf,
        match_lf=basic,
        sigma=sigma,
        patch_size=patch_size,
        stride=stride_wiener,
        disp_radius=disp_radius_wiener,
        sim_radius=sim_radius_wiener,
        max_group_size=max_group_size_wiener,
        lambda_hard=lambda_hard,
        stage="wiener",
        search_chunk=search_chunk,
        group_pool=group_pool,
        transform_chunk=transform_chunk,
    )

    if clip_output:
        denoised = denoised.clamp(0.0, 1.0)

    if squeeze_channel:
        denoised = denoised.squeeze(2)

    return denoised
