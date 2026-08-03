import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.base.base_ops import LayerNorm2d
from einops import rearrange, repeat
from methods.base.base_ops import LayerNorm2d, resize_to

def get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    elif act_name == "gelu":
        return nn.GELU()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "idy":
        return nn.Identity()
    else:
        raise NotImplementedError


class ConvGNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, act=True):
        if padding is None:
            padding = (kernel_size // 2) * dilation
        super().__init__()
        self.add_module("conv", nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias))
        self.add_module("gn", nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch))
        if act:
            self.add_module("relu", nn.ReLU(inplace=True))


class LNConvAct(nn.Sequential):
    def __init__(self, in_planes, out_planes, k, s=1, p=0, d=1, g=1, bias=False, act_name="relu"):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.add_module("ln", LayerNorm2d(in_planes))
        self.add_module("conv", nn.Conv2d(in_planes, out_planes, k, s, p, d, g, bias=bias))
        if act_name is not None:
            self.add_module(name=act_name, module=get_act_fn(act_name=act_name))


class ConvProj(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        mlp_times: float = 1,
        norm_layer: nn.Module = LayerNorm2d,
        bias: bool = False,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(mlp_times * in_features)

        self.norm = norm_layer(in_features) if norm_layer else nn.Identity()
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=bias)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class ConvMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        mlp_times: float = 1,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = LayerNorm2d,
        bias: bool = False,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(mlp_times * in_features)

        self.norm = norm_layer(in_features) if norm_layer else nn.Identity()
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features, bias=bias)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x



class SemanticGuidanceBlock(nn.Module):
    def __init__(self, feat_dim, guide_dim, num_heads=4):
        super().__init__()
        self.feat_dim = feat_dim 
        self.guide_dim = guide_dim

        self.guide_norm = nn.LayerNorm(guide_dim)
        self.feat_norm = LayerNorm2d(feat_dim)

        self.num_heads = num_heads
        self.attn_scale = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.g_proj = nn.Linear(guide_dim, feat_dim, bias=False)
        self.qkv_proj = nn.Conv2d(feat_dim, 3 * feat_dim, 1, bias=False)
        self.o_proj = nn.Conv2d(feat_dim, feat_dim, 1, bias=True)

        self.norm_ffn = ConvMlp(feat_dim, feat_dim)

    def masked_pooling(self, image_feat, mask):                         
        obj_embs = mask * resize_to(image_feat, tgt_hw=mask.shape[-2:])
        obj_embs = obj_embs.sum((2, 3)) / mask.sum((2, 3))  # b d
        return obj_embs

    def forward(self, img_feats, cls_embs, cls_logits=None, seg_logits=None, verbose=False):
        """
        img_feats: B,C,H,W
        cls_embs: Nc,D
        cls_logits: B,Nc
        seg_logits: B,1,H,W
        """
        normed_img_feats = self.feat_norm(img_feats)
        _, _, H, W = normed_img_feats.shape
        normed_cls_embs = self.guide_norm(cls_embs)

        g = self.g_proj(normed_cls_embs)  # Nc,D
        qkv = self.qkv_proj(normed_img_feats)  # B,D,H,W

        g = rearrange(g, "nc (nh hd) -> nc nh hd", nh=self.num_heads)
        qkv = rearrange(qkv, "b (ng nh hd) h w -> ng b nh hd (h w)", ng=3, nh=self.num_heads)
        q, k, v = qkv.unbind(0)

        guide_map = torch.einsum("nhd, bhdl -> bhnl", F.normalize(g, dim=-1), F.normalize(q, dim=-2))
        guide_map = guide_map - guide_map.min(dim=-1, keepdim=True).values
        guide_map = guide_map / guide_map.max(dim=-1, keepdim=True).values
        guide_map = (guide_map.softmax(dim=2) * guide_map).sum(dim=2).unsqueeze(2)

        if cls_logits is not None:
            cls_logits = repeat(cls_logits, "b nc -> b nh nc", nh=self.num_heads)
            cls_obj_embs = torch.einsum("nhd, bhn -> bhd", g, cls_logits)

            seg_obj_embs = self.masked_pooling(k, seg_logits.sigmoid())  # b d
            seg_obj_embs = rearrange(seg_obj_embs, "b (nh hd) -> b nh hd", nh=self.num_heads)
            obj_embs = cls_obj_embs + seg_obj_embs

            c_norm_obj_embs = F.normalize(obj_embs, dim=-1)  # b nh hd
            c_norm_k = F.normalize(k, dim=-2)  # b nh hd hw
            aux_guide_map = torch.einsum("bhd, bhdl -> bhl", c_norm_obj_embs, c_norm_k)  # b,nh,hw
            aux_guide_map = aux_guide_map.unsqueeze(2)  # b,nh,1,hw
            #
            aux_guide_map = aux_guide_map - aux_guide_map.min(dim=-1, keepdim=True).values
            aux_guide_map = aux_guide_map / aux_guide_map.max(dim=-1, keepdim=True).values
            guide_map = guide_map * aux_guide_map
        v = guide_map * v

        hw_norm_q = F.normalize(q, dim=-1)  # b nh hd1 hw
        hw_norm_k = F.normalize(k, dim=-1)  # b nh hd2 hw
        attn = hw_norm_q @ hw_norm_k.transpose(-1, -2)  # b nh hd1 hw @ b nh hw hd2 => b nh hd1 hd2
        attn = self.attn_scale * attn
        qkv = attn.softmax(dim=-1) @ v  # b nh hd1 hw 
        qkv = rearrange(qkv, "b nh hd (h w) -> b (nh hd) h w", h=H, w=W)

        img_feats = img_feats + self.o_proj(qkv)
        img_feats = img_feats + self.norm_ffn(img_feats)
        return img_feats


class StructureEnhancementBlock(nn.Module):
    def __init__(self, dim, num_heads=4, num_aux=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_aux = num_aux

        self.q_norm = LayerNorm2d(dim)
        self.kv_norm = LayerNorm2d(dim * num_aux)

        self.q_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.kv_proj = nn.Conv2d(dim * num_aux, 2 * dim, 1, bias=False)
        self.o_proj = nn.Conv2d(dim, dim, 1, bias=True)

        self.ffn = ConvMlp(dim, dim)

        self.attn_scale = nn.Parameter(torch.zeros(num_heads, 1, 1))

        self.gate_conv = nn.Sequential(
            nn.Conv2d(num_aux, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, img_feat, aux_feats):
        """
        img_feat: (B, C, H, W)
        aux_feats: list of (B, C, H, W) features (e.g., [edge_feat, depth_feat])
        """
        assert isinstance(aux_feats, (list, tuple)) and len(aux_feats) == self.num_aux
        B, C, H, W = img_feat.shape

        aux_mean = torch.cat(
            [f.mean(1, keepdim=True) for f in aux_feats],  # B,1,H,W
            dim=1
        )  # B, num_aux, H, W
        gate = self.gate_conv(aux_mean)  # B,1,H,W

        q = self.q_proj(self.q_norm(img_feat))  # B, dim, H, W
        kv = self.kv_proj(self.kv_norm(torch.cat(aux_feats, dim=1)))  # B, 2*dim, H, W
        kv = rearrange(kv, "b (ng nh hd) h w -> ng b nh hd (h w)", ng=2, nh=self.num_heads)

        k, v = kv.unbind(0)  # k: B, nh, hd, hw
        q = rearrange(q, "b (nh hd) h w -> b nh hd (h w)", nh=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = q @ k.transpose(-1, -2)  # (b, nh, hd, hd)
        attn = attn * self.attn_scale.sigmoid()
        attn = attn.softmax(dim=-1)

        out = attn @ v  # (b, nh, hd, hw)
        out = rearrange(out, "b nh hd (h w) -> b (nh hd) h w", h=H, w=W)
        out = out * gate

        img_feat = img_feat + self.o_proj(out)
        img_feat = img_feat + self.ffn(img_feat)

        return img_feat




# reference from CSGNet: https://github.com/ycyinchao/CSGNet/blob/main/lib/models/detectors/Net.py
class CrossLayerFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_prev = ConvGNReLU(dim, dim, kernel_size=3, stride=1, dilation=1, bias=False)
        self.conv_curr = ConvGNReLU(dim, dim, kernel_size=3, stride=1, dilation=1, bias=False)
        self.fuse = nn.Conv2d(2 * dim, dim, 1, bias=True)
        self.adaptive_conv = nn.Conv2d(dim, 1, 1, bias=False)
        self.spatial_conv = nn.Conv2d(3, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fi, fli):
        fli = F.interpolate(fli, size=fi.shape[-2:], mode="bilinear", align_corners=False)
        prev_feat = self.conv_prev(fi)
        curr_feat = self.conv_curr(fli)
        fused = self.fuse(torch.cat([prev_feat, curr_feat], dim=1))

        avg_out = fused.mean(dim=1, keepdim=True)
        max_out, _ = fused.max(dim=1, keepdim=True)
        adaptive_out = self.adaptive_conv(fused)
        sa = torch.cat([avg_out, max_out, adaptive_out], dim=1)
        sa_map = self.sigmoid(self.spatial_conv(sa))
        return fused * (1 + sa_map)

# reference from CSGNet: https://github.com/ycyinchao/CSGNet/blob/main/lib/models/detectors/Net.py
class SemanticRefinement(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_semantic = ConvGNReLU(dim, dim, kernel_size=3, stride=1, dilation=1, bias=False)
        self.conv_visual = ConvGNReLU(dim, dim, kernel_size=3, stride=1, dilation=1, bias=False)
        self.fuse = nn.Conv2d(2 * dim, dim, 1, bias=True)
        self.gate = nn.Conv2d(dim, dim, 1)

    def forward(self, semantic_feat, visual_feat):
        semantic_feat = F.interpolate(semantic_feat, size=visual_feat.shape[-2:], mode="bilinear", align_corners=False)
        sem = self.conv_semantic(semantic_feat)
        vis = self.conv_visual(visual_feat)
        fusion = self.fuse(torch.cat([sem, vis], dim=1))
        gate = torch.tanh(self.gate(fusion))
        return fusion * (1.0 + gate)


# reference from SDRNet: https://github.com/whyandbecause/SDRNet/blob/main/lib/SDRNet.py
class DetailEnhance(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_d1 = ConvGNReLU(dim, dim, kernel_size=3, stride=1, dilation=1, bias=False, act=False)
        self.conv_d2 = ConvGNReLU(dim, dim, kernel_size=3, stride=1, dilation=2, bias=False, act=False)
        self.conv_d4 = ConvGNReLU(dim, dim, kernel_size=3, stride=1, dilation=4, bias=False, act=False)

    def forward(self, x):
        x1 = self.conv_d1(x)
        x2 = self.conv_d2(x1)
        x3 = self.conv_d4(x2)
        return x + 0.5 * (x - x1) + 0.5 * (x1 - x2) + 0.25 * (x2 - x3)


class GlobalLocalizationHead(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fg5_conv3 = ConvGNReLU(3 * dim, dim, kernel_size=3, stride=1, dilation=1, bias=False)
        self.fg5_conv5 = ConvGNReLU(dim, dim, kernel_size=5, stride=1, dilation=1, bias=False)

        self.f5_conv = ConvGNReLU(2 * dim, dim, kernel_size=3, stride=1, dilation=1, bias=False)
        self.f4_conv = ConvGNReLU(2 * dim, dim, kernel_size=3, stride=1, dilation=1, bias=False)
        self.f3_conv = ConvGNReLU(2 * dim, dim, kernel_size=3, stride=1, dilation=1, bias=False)
        self.f2_conv = ConvGNReLU(2 * dim, dim, kernel_size=3, stride=1, dilation=1, bias=False)

        self.fg4_conv = ConvGNReLU(dim, dim, kernel_size=3, stride=1, dilation=1, bias=False)
        self.fg3_conv = ConvGNReLU(dim, dim, kernel_size=3, stride=1, dilation=1, bias=False)
        self.fg2_conv = ConvGNReLU(dim, dim, kernel_size=3, stride=1, dilation=1, bias=False)
        self.fg1_conv = ConvGNReLU(dim, dim, kernel_size=3, stride=1, dilation=1, bias=False)

        self.pred_heads = nn.ModuleList([nn.Conv2d(dim, 1, 3, 1, 1) for _ in range(5)])

    def forward(self, image_res2, image_res3, image_res4, image_res5, out_hw=None):
        r2, r3, r4, r5 = image_res2, image_res3, image_res4, image_res5

        r3_up = F.interpolate(r3, size=r5.shape[-2:], mode="bilinear", align_corners=False)
        r4_up = F.interpolate(r4, size=r5.shape[-2:], mode="bilinear", align_corners=False)
        concat = torch.cat([r3_up, r4_up, r5], dim=1)
        x = self.fg5_conv3(concat)
        fg5 = self.fg5_conv5(x)

        f5 = self.f5_conv(torch.cat([fg5, r5], dim=1))
        fg5_up_f5 = F.interpolate(fg5, size=f5.shape[-2:], mode="bilinear", align_corners=False)
        fused = fg5_up_f5 * f5 + f5
        fg4 = self.fg4_conv(fused)

        fg5_up_f4 = F.interpolate(fg5, size=r4.shape[-2:], mode="bilinear", align_corners=False)
        f4 = self.f4_conv(torch.cat([fg5_up_f4, r4], dim=1))
        fg4_up_f4 = F.interpolate(fg4, size=f4.shape[-2:], mode="bilinear", align_corners=False)
        fused = fg4_up_f4 * f4 + f4
        fg3 = self.fg3_conv(fused)

        fg5_up_f3 = F.interpolate(fg5, size=r3.shape[-2:], mode="bilinear", align_corners=False)
        f3 = self.f3_conv(torch.cat([fg5_up_f3, r3], dim=1))
        fg3_up_f3 = F.interpolate(fg3, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        fused = fg3_up_f3 * f3 + f3
        fg2 = self.fg2_conv(fused)

        fg5_up_f2 = F.interpolate(fg5, size=r2.shape[-2:], mode="bilinear", align_corners=False)
        f2 = self.f2_conv(torch.cat([fg5_up_f2, r2], dim=1))
        fg2_up_f2 = F.interpolate(fg2, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        fused = fg2_up_f2 * f2 + f2
        fg1 = self.fg1_conv(fused)

        fgs = [fg5, fg4, fg3, fg2, fg1]

        preds = []
        for i, fg in enumerate(fgs):
            p = self.pred_heads[i](fg)
            if out_hw is not None:
                p = F.interpolate(p, size=out_hw, mode="bilinear", align_corners=False)
            else:
                p = F.interpolate(p, scale_factor=4, mode="bilinear", align_corners=False)
            preds.append(p)

        return fgs[-1], fgs, preds[-1], preds
    


class AdaptivePartition(nn.Module):
    def __init__(self, gamma=0.4, threshold=0.2):
        super().__init__()
        self.gamma = gamma
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, loc_prob):
        b, _, h, w = loc_prob.shape
        boxes = []
        for idx in range(b):
            prob = loc_prob[idx, 0]
            fg = prob > self.threshold

            if fg.any():
                ys, xs = torch.where(fg)
                y1, y2 = ys.min().item(), ys.max().item()
                x1, x2 = xs.min().item(), xs.max().item()
            else:
                y1, x1, y2, x2 = 0, 0, h - 1, w - 1

            box_h = y2 - y1 + 1
            box_w = x2 - x1 + 1
            in_box = prob[y1 : y2 + 1, x1 : x2 + 1]
            rt = in_box.sum().item() / max(float(box_h * box_w), 1.0)
            scale = 1.0 + self.gamma * rt

            new_h = min(int(round(box_h * scale)), h)
            new_w = min(int(round(box_w * scale)), w)
            cy = 0.5 * (y1 + y2)
            cx = 0.5 * (x1 + x2)

            ny1 = int(round(cy - 0.5 * (new_h - 1)))
            ny2 = ny1 + new_h - 1
            nx1 = int(round(cx - 0.5 * (new_w - 1)))
            nx2 = nx1 + new_w - 1

            ny1 = max(0, min(ny1, h - new_h))
            nx1 = max(0, min(nx1, w - new_w))
            ny2 = ny1 + new_h - 1
            nx2 = nx1 + new_w - 1

            boxes.append([ny1, nx1, ny2, nx2])
            bg = torch.ones((1, h, w), dtype=loc_prob.dtype, device=loc_prob.device)
            bg[:, ny1 : ny2 + 1, nx1 : nx2 + 1] = 0.0

        boxes = torch.tensor(boxes, dtype=torch.long, device=loc_prob.device)

        return {
            "boxes": boxes,
        }


class ContrastiveBackgroundSuppression(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.global_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.local_proj = nn.Conv2d(dim, dim, 1, bias=False)

        self.out = nn.Sequential(
            ConvGNReLU(dim, dim, kernel_size=3, stride=1, dilation=1, bias=False),
            ConvGNReLU(dim, dim, kernel_size=3, stride=1, dilation=1, bias=False, act=False)
        )

    def forward(self, local_feat, bg_proto):
        bg = self.global_proj(bg_proto)
        local = self.local_proj(local_feat)

        bg_norm = F.normalize(bg, dim=1)
        local_norm = F.normalize(local, dim=1)
        sim = (local_norm * bg_norm).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)

        gate = (1.0 - sim) * 0.5
        gate = gate.clamp(0.0, 1.0)

        Lr = local_feat * gate
        ret_feat = self.out(Lr) + local_feat

        return ret_feat, gate


class ContextualBackgroundAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.bg_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.ctx_proj = nn.Conv2d(dim, dim, 1, bias=False)

        self.adapter_in = ConvGNReLU(dim * 2 + 1, dim, kernel_size=3, stride=1, dilation=1, bias=False)
        self.adapter_out = nn.Conv2d(dim, dim, 1, bias=True)
        self.scale = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, feat, bg_mask, bg_proto):
        if bg_mask.shape[-2:] != feat.shape[-2:]:
            bg_mask = F.interpolate(bg_mask, size=feat.shape[-2:], mode='nearest')
        if bg_proto.shape[-2:] != feat.shape[-2:]:
            bg_proto = F.interpolate(bg_proto, size=feat.shape[-2:], mode='nearest')
        bg_proto_proj = self.bg_proj(bg_proto)
        ctx_proj = self.ctx_proj(feat)
        x = torch.cat([ctx_proj, bg_proto_proj, bg_mask], dim=1)
        delta = self.adapter_out(self.adapter_in(x))
        return feat + self.scale.sigmoid() * delta + self.shift * bg_mask
    
