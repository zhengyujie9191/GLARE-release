import open_clip
import torch
import torch.nn.functional as F
from loguru import logger
from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from torch import nn
from torch.utils.checkpoint import checkpoint

from methods.base.base_ops import PixelNormalizer, rescale_2x, resize_to
from methods.glare.modules import (
    LNConvAct,
    ConvProj,
    StructureEnhancementBlock,
    SemanticGuidanceBlock,
    CrossLayerFusion,
    DetailEnhance,
    AdaptivePartition,
    SemanticRefinement,
    GlobalLocalizationHead,
    ContextualBackgroundAdapter,
    ContrastiveBackgroundSuppression,
)
from methods.glare.utils import (
    crop_and_resize_batch,
    paste_to_full,
    boxes_to_bg_mask,
    compute_bg_proto,
)
from methods.glare.loss import (
    edge_dice_loss,
    localization_loss,
    seg_loss,
)

from .prompts import get_prompt_template_by_name


## CLIP-ConvNeXt-XXL
# model_name="convnext_xxlarge"
# pretrained="laion2b_s34b_b82k_augreg_soup"

## CLIP-ConvNeXt-L
# model_name="convnext_large_d_320"
# pretrained="laion2b_s29b_b131k_ft_soup"


class ConvNeXtCLIP(nn.Module):
    def __init__(
        self,
        model_name="convnext_xxlarge",
        pretrained="laion2b_s34b_b82k_augreg_soup",
        template_set="camoprompts",
    ):
        super().__init__()
        self.clip_model, _, self.preprocess_val = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.mean = OPENAI_DATASET_MEAN
        self.std = OPENAI_DATASET_STD
        self.text_tokenizer = open_clip.get_tokenizer(model_name)

        self.template_set = get_prompt_template_by_name(template_set)
        logger.info(f"Create the CLIP ({model_name + '-' + pretrained}) with template_set {self.template_set}")

        model_name = model_name.lower()
        assert "convnext_" in model_name
        self.model_type = "convnext"
        if "_base" in model_name:
            self.feat_chans = [128, 128, 256, 512, 1024]
        elif "_large" in model_name:
            self.feat_chans = [192, 192, 384, 768, 1536]
        elif "_xxlarge" in model_name:
            self.feat_chans = [384, 384, 768, 1536, 3072]

        self.dim_latent = self.clip_model.text_projection.shape[-1]
        self.out_strides = {"stem": 2, "res2": 4, "res3": 8, "res4": 16, "res5": 32, "emb": -1}
        self.out_chans = {
            "stem": self.feat_chans[0],
            "res2": self.feat_chans[1],
            "res3": self.feat_chans[2],
            "res4": self.feat_chans[3],
            "res5": self.feat_chans[4],
            "emb": self.dim_latent,
        }

    def output_shape(self):
        return {
            name: dict(channels=self.out_chans[name], stride=self.out_strides[name])
            for name in ["stem", "res2", "res3", "res4", "res5", "emb"]
        }

    @property
    def device(self):
        for param in self.clip_model.parameters():
            return param.device

    @torch.no_grad()
    def get_text_embs(self, text_list, normalize=True):
        """对输入的所有类别名称都使用一套模板构建平均嵌入
        return: NumberofClasses,D
        """
        
        # reference for templates: https://github.com/mlfoundations/open_clip/blob/91f6cce16b7bee90b3b5d38ca305b5b3b67cc200/src/training/imagenet_zeroshot_data.py
        text_tokens = self.text_tokenizer(text_list).to(self.device)

        # list -> TD
        cast_dtype = self.clip_model.transformer.get_cast_dtype()
        x = self.clip_model.token_embedding(text_tokens).to(cast_dtype)  # [num_temp, n_ctx, d_model]
        #
        x = x + self.clip_model.positional_embedding.to(cast_dtype)  # 80,77,768  77,768

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]

        # take feats from the eot embedding (eot_token is the highest number in each sequence)
        text_embs = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.clip_model.text_projection

        if normalize:
            text_embs = F.normalize(text_embs, dim=-1)  # Nc,768
        return text_embs

    @torch.no_grad()
    def get_text_embs_by_template(self, text_list):
        """对输入的所有类别名称都使用一套模板构建平均嵌入
        return: NumberofClasses,D
        """
        
        text_embs = []
        for text in text_list:
            # reference for templates: https://github.com/mlfoundations/open_clip/blob/91f6cce16b7bee90b3b5d38ca305b5b3b67cc200/src/training/imagenet_zeroshot_data.py
            text_tokens = self.text_tokenizer([template.format(text) for template in self.template_set]).to(
                self.device
            )

            # list -> TD
            cast_dtype = self.clip_model.transformer.get_cast_dtype()
            x = self.clip_model.token_embedding(text_tokens).to(cast_dtype)  # [num_temp, n_ctx, d_model]
            #
            x = x + self.clip_model.positional_embedding.to(cast_dtype)  # 80,77,768  77,768

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]

            # take feats from the eot embedding (eot_token is the highest number in each sequence) => Nc,768
            text_emb = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.clip_model.text_projection
            text_embs.append(text_emb)
        text_embs = torch.stack(text_embs, dim=0)  # Nc,Nt,768

        text_embs /= text_embs.norm(dim=-1, keepdim=True)
        text_embs = text_embs.mean(1)
        text_embs /= text_embs.norm(dim=-1, keepdim=True)
        return text_embs  # Nc,768

    @torch.no_grad()
    def visual_feats_to_embs(self, x, normalize: bool = True):
        """
        将图像特征转换为图像嵌入向量
        """
        x = self.clip_model.visual.trunk.head(x)
        x = self.clip_model.visual.head(x)
        return F.normalize(x, dim=-1) if normalize else x

    @torch.no_grad()
    def get_visual_feats(self, x):
        
        out = {}
        x = self.clip_model.visual.trunk.stem(x)
        out["stem"] = x.contiguous()  # os4
        for i in range(4):
            x = self.clip_model.visual.trunk.stages[i](x)
            out[f"res{i + 2}"] = x.contiguous()  # res 2 (os4), 3 (os8), 4 (os16), 5 (os32)

        x = self.clip_model.visual.trunk.norm_pre(x)
        out["clip_vis_dense"] = x.contiguous()
        return out


class GLARE(nn.Module):
    def __init__(
        self,
        iterations=2,
        template_set="camoprompts",
        use_checkpointing=False,
        adapt_gamma=0.4,
        adapt_threshold=0.2,
        loc_loss_weight=1.0,
        seg_loss_weight=1.0,
        edge_loss_weight=0.5,
    ):
        super().__init__()
        assert iterations in [1, 2, 3]
        self.iterations = iterations
        logger.info(f"Current Iteration: {iterations}")

        self.clip = ConvNeXtCLIP(template_set=template_set)
        self.normalizer = PixelNormalizer(mean=self.clip.mean, std=self.clip.std)
        self.test_class_embs = None
        self.train_class_embs = None

        # self.encoder_dims = [192, 384, 768, 1536]
        self.encoder_dims = [384, 768, 1536, 3072]
        self.hid_dim = 128

        self.tra1 = ConvProj(self.encoder_dims[0], self.hid_dim)
        self.tra2 = ConvProj(self.encoder_dims[0], self.hid_dim)
        self.tra3 = ConvProj(self.encoder_dims[1], self.hid_dim)
        self.tra4 = ConvProj(self.encoder_dims[2], self.hid_dim)
        self.tra5 = ConvProj(self.encoder_dims[3], self.hid_dim)

        self.tra1_glo = ConvProj(self.encoder_dims[0], self.hid_dim)
        self.tra2_glo = ConvProj(self.encoder_dims[1], self.hid_dim)
        self.tra3_glo = ConvProj(self.encoder_dims[2], self.hid_dim)
        self.tra4_glo = ConvProj(self.encoder_dims[3], self.hid_dim)
        self.global_head = GlobalLocalizationHead(self.hid_dim)

        self.cba_3 = ContextualBackgroundAdapter(self.hid_dim)
        self.cba_2 = ContextualBackgroundAdapter(self.hid_dim)
        self.cba_1 = ContextualBackgroundAdapter(self.hid_dim)


        self.dec5 = SemanticGuidanceBlock(feat_dim=self.hid_dim, guide_dim=self.clip.dim_latent)
        self.dec4 = SemanticGuidanceBlock(feat_dim=self.hid_dim, guide_dim=self.clip.dim_latent)
        self.dec3 = SemanticGuidanceBlock(feat_dim=self.hid_dim, guide_dim=self.clip.dim_latent)
        self.dec2 = SemanticGuidanceBlock(feat_dim=self.hid_dim, guide_dim=self.clip.dim_latent)
        self.dec1 = SemanticGuidanceBlock(feat_dim=self.hid_dim, guide_dim=self.clip.dim_latent)

        self.cod_head = nn.ModuleList([
            nn.Sequential(LNConvAct(self.hid_dim, self.hid_dim, 3, 1, 1, act_name="relu"), nn.Conv2d(self.hid_dim, 1, 3, 1, 1))
            for _ in range(3)
        ])

        self.clf_54 = CrossLayerFusion(self.hid_dim)
        self.clf_43 = CrossLayerFusion(self.hid_dim)
        self.clf_32 = CrossLayerFusion(self.hid_dim)
        self.clf_21 = CrossLayerFusion(self.hid_dim)

        self.srm4 = SemanticRefinement(self.hid_dim)
        self.srm3 = SemanticRefinement(self.hid_dim)
        self.srm2 = SemanticRefinement(self.hid_dim)
        self.srm1 = SemanticRefinement(self.hid_dim)

        self.mrg3 = StructureEnhancementBlock(self.hid_dim, num_aux=1)
        self.mrg2 = StructureEnhancementBlock(self.hid_dim, num_aux=1)
        self.mrg1 = StructureEnhancementBlock(self.hid_dim, num_aux=1)

        self.edg_stems = nn.ModuleList([nn.Sequential(LNConvAct(self.hid_dim, self.hid_dim, 3, 1, 1, act_name="relu"), nn.Conv2d(self.hid_dim, self.hid_dim, 3, 1, 1)) for _ in range(3)])
        self.de = DetailEnhance(self.hid_dim)
        self.edg_heads = nn.ModuleList([LNConvAct(self.hid_dim, 1, 3, 1, 1, act_name="idy") for _ in range(3)])

        self.partitioner = AdaptivePartition(gamma=adapt_gamma, threshold=adapt_threshold)
        self.cbs_3 = ContrastiveBackgroundSuppression(self.hid_dim)
        self.cbs_2 = ContrastiveBackgroundSuppression(self.hid_dim)
        self.cbs_1 = ContrastiveBackgroundSuppression(self.hid_dim)

        self.mask_temperature = nn.Parameter(torch.tensor(1.0))
        self.cls_confidence_net = nn.Sequential(
            nn.Linear(self.hid_dim + 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.loc_loss_weight = float(loc_loss_weight)
        self.seg_loss_weight = float(seg_loss_weight)
        self.edge_loss_weight = float(edge_loss_weight)


        self.use_checkpointing = use_checkpointing
        
    def _maybe_checkpoint(self, fn, *args):
        if getattr(self, "use_checkpointing", False) and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def get_visual_feats(self, image):
        image = self.normalizer(image)
        image_feats = self.clip.get_visual_feats(image)
        image_stem = rescale_2x(image_feats["stem"])  # 192, 80, 80 -> 192, 160, 160
        image_res5 = image_feats["res5"]
        image_res4 = image_feats["res4"]
        image_res3 = image_feats["res3"]
        image_res2 = image_feats["res2"]

        image_stem = self.tra1(image_stem)
        image_res2 = self.tra2(image_res2)
        image_res3 = self.tra3(image_res3)
        image_res4 = self.tra4(image_res4)
        image_res5 = self.tra5(image_res5)

        image_res2_glo = self.tra1_glo(image_feats["res2"])
        image_res3_glo = self.tra2_glo(image_feats["res3"])
        image_res4_glo = self.tra3_glo(image_feats["res4"])
        image_res5_glo = self.tra4_glo(image_feats["res5"])

        image_deep = image_feats["clip_vis_dense"]
        return image_stem, image_res2, image_res3, image_res4, image_res5, image_deep, image_res2_glo, image_res3_glo, image_res4_glo, image_res5_glo

    def _compute_mask_stats(self, mask_prob):
        b = mask_prob.shape[0]
        mask_flat = mask_prob.view(b, -1)
        mean_act = mask_flat.mean(dim=1, keepdim=True)
        std_act = mask_flat.std(dim=1, keepdim=True)
        eps = 1e-7
        entropy = -(mask_flat * torch.log(mask_flat + eps) + (1 - mask_flat) * torch.log(1 - mask_flat + eps))
        entropy = entropy.mean(dim=1, keepdim=True) / 0.69314718056
        coverage = (mask_flat > 0.5).float().mean(dim=1, keepdim=True)
        return torch.cat([mean_act, std_act, entropy, coverage], dim=1)

    def _sharpen_mask(self, mask_prob):
        eps = 1e-7
        mask_logits = torch.log(mask_prob + eps) - torch.log(1 - mask_prob + eps)
        temperature = F.softplus(self.mask_temperature) + 0.1
        return torch.sigmoid(mask_logits / temperature)

    def _predict_local_path_confidence(self, mask_prob, loc_feat):
        if loc_feat is None:
            stats = self._compute_mask_stats(mask_prob)
            conf = (1.0 - stats[:, 2:3]).clamp(0.0, 1.0)
            return conf

        stats = self._compute_mask_stats(mask_prob)
        loc_pooled = F.adaptive_avg_pool2d(loc_feat, output_size=1).flatten(1)
        return self.cls_confidence_net(torch.cat([loc_pooled, stats], dim=1))

    def _compute_cls_logits(self, image_embs, normed_class_embs):
        if normed_class_embs.ndim == 2:
            class_logits = image_embs @ normed_class_embs.T
        else:
            class_logits = torch.einsum("bd,bnd->bn", image_embs, normed_class_embs)
        return self.clip.clip_model.logit_scale.exp() * class_logits

    def _global_cls_logits(self, image_deep, normed_class_embs):
        global_vec = F.adaptive_avg_pool2d(image_deep, output_size=1)
        global_embs = self.clip.visual_feats_to_embs(global_vec, normalize=True)
        return self._compute_cls_logits(global_embs, normed_class_embs)

    def map_classifier(self, logits, image_deep, normed_class_embs, loc_feat=None):
        prob = logits.sigmoid()
        sharp_prob = self._sharpen_mask(prob)

        image_feats = resize_to(image_deep, tgt_hw=prob.shape[-2:])
        local_vec = (sharp_prob * image_feats).sum(dim=(-1, -2)) / sharp_prob.sum(dim=(-1, -2)).clamp_min(1e-6)
        local_vec = local_vec[..., None, None]
        local_embs = self.clip.visual_feats_to_embs(local_vec, normalize=True)
        local_logits = self._compute_cls_logits(local_embs, normed_class_embs)

        global_logits = self._global_cls_logits(image_deep, normed_class_embs)

        if loc_feat is not None and loc_feat.shape[-2:] != prob.shape[-2:]:
            loc_feat = resize_to(loc_feat, tgt_hw=prob.shape[-2:])
        local_conf = self._predict_local_path_confidence(prob, loc_feat)

        fused_logits = local_conf * local_logits + (1.0 - local_conf) * global_logits

        return fused_logits


    def body_local(self, image, normed_class_embs, global_context=None, bg_mask=None):
        image_stem, image_res2, image_res3, image_res4, image_res5, image_deep, _, _, _, _ = self.get_visual_feats(image)
        refined_stem_shared = image_stem
        
        x5 = self._maybe_checkpoint(self.dec5, image_res5, normed_class_embs)
        x4 = self._maybe_checkpoint(self.clf_54, image_res4, x5)
        refined_stem_shared = self._maybe_checkpoint(self.srm4, x4, refined_stem_shared)
        x4 = self._maybe_checkpoint(self.dec4, x4, normed_class_embs)

        x3 = self._maybe_checkpoint(self.clf_43, image_res3, x4)
        refined_stem_shared = self._maybe_checkpoint(self.srm3, x3, refined_stem_shared)

        logits = {}
        cls_logits = None
        seg_logits = None

        prev_seg_logits = None

        for i in range(1, self.iterations + 1):
            global_bg_mask_clean = bg_mask
            local_bg_mask_iter = None
            bg_proto = compute_bg_proto(global_context, global_bg_mask_clean) if (global_context is not None and global_bg_mask_clean is not None) else None            
            if i > 1 and prev_seg_logits is not None:
                temperature = 0.3
                pred_fg_prob = torch.sigmoid(prev_seg_logits / temperature).detach()
                confidence = (pred_fg_prob - 0.5).abs() * 2
                local_bg_mask_iter = (1.0 - pred_fg_prob) * confidence
                        
            x3 = self._maybe_checkpoint(self.dec3, x3, normed_class_embs, cls_logits, seg_logits)
            
            if global_context is not None:
                if local_bg_mask_iter is not None:
                    local_bg_mask_3 = F.interpolate(local_bg_mask_iter, size=x3.shape[-2:], mode='nearest')
                    x3 = self.cba_3(x3, local_bg_mask_3, bg_proto=bg_proto)
            
                x3, _ = self.cbs_3(local_feat=x3, bg_proto=bg_proto)

            edg_feat3 = self._maybe_checkpoint(self.de, self.edg_stems[2](x3))
            aux_feats3 = [edg_feat3]
            x3 = self._maybe_checkpoint(self.mrg3, x3, aux_feats3)

            x2 = self._maybe_checkpoint(self.clf_32, image_res2, x3)
            refined_stem_shared = self._maybe_checkpoint(self.srm2, x2, refined_stem_shared)

            x2 = self._maybe_checkpoint(self.dec2, x2, normed_class_embs, cls_logits, seg_logits)
                
            if global_context is not None:
                if local_bg_mask_iter is not None:
                    local_bg_mask_2 = F.interpolate(local_bg_mask_iter, size=x2.shape[-2:], mode='nearest')
                    x2 = self.cba_2(x2, local_bg_mask_2, bg_proto=bg_proto)
            
                x2, _ = self.cbs_2(local_feat=x2, bg_proto=bg_proto)

            edg_feat2 = self._maybe_checkpoint(self.de, self.edg_stems[1](x2))
            aux_feats2 = [edg_feat2]
            x2 = self._maybe_checkpoint(self.mrg2, x2, aux_feats2)

            x1 = self._maybe_checkpoint(self.clf_21, refined_stem_shared, x2)
            refined_stem_shared = self._maybe_checkpoint(self.srm1, x1, refined_stem_shared)

            x1 = self._maybe_checkpoint(self.dec1, x1, normed_class_embs, cls_logits, seg_logits)

            if global_context is not None:
                if local_bg_mask_iter is not None:
                    local_bg_mask_1 = F.interpolate(local_bg_mask_iter, size=x1.shape[-2:], mode='nearest')
                    x1 = self.cba_1(x1, local_bg_mask_1, bg_proto=bg_proto)
            
                x1, _ = self.cbs_1(local_feat=x1, bg_proto=bg_proto)
                
            edg_feat1 = self._maybe_checkpoint(self.de, self.edg_stems[0](x1))
            aux_feats1 = [edg_feat1]
            x1 = self._maybe_checkpoint(self.mrg1, x1, aux_feats1)

            seg_logits = self._maybe_checkpoint(self.cod_head[0], rescale_2x(x1))

            seg3 = self._maybe_checkpoint(self.cod_head[2], x3)
            seg2 = self._maybe_checkpoint(self.cod_head[1], x2)
            seg1 = self._maybe_checkpoint(self.cod_head[0], x1)

            logits[f"seg{i}-3"] = seg3
            logits[f"seg{i}-2"] = seg2
            logits[f"seg{i}-1"] = seg1
            logits[f"edg{i}-3"] = self._maybe_checkpoint(self.edg_heads[2], edg_feat3)
            logits[f"edg{i}-2"] = self._maybe_checkpoint(self.edg_heads[1], edg_feat2)
            logits[f"edg{i}-1"] = self._maybe_checkpoint(self.edg_heads[0], edg_feat1)

            if i != self.iterations:
                cls_logits = self._maybe_checkpoint(self.map_classifier, seg_logits, image_deep, normed_class_embs, x1)

            logits[f"seg{i}"] = seg_logits
            logits[f"global_bg_mask{i}_3"] = global_bg_mask_clean
            logits[f"global_bg_mask{i}_2"] = global_bg_mask_clean
            logits[f"global_bg_mask{i}_1"] = global_bg_mask_clean

            prev_seg_logits = seg_logits.detach()

        return logits

    def _forward_pipeline(self, image, normed_class_embs):
        image_hw = image.shape[-2:]
        _, _, _, _, _, image_deep, image_res2_glo, image_res3_glo, image_res4_glo, image_res5_glo = self.get_visual_feats(image)
        fused, _, loc_logits, loc_preds = self.global_head(image_res2_glo, image_res3_glo, image_res4_glo, image_res5_glo)
        loc_side_logits = loc_preds

        with torch.no_grad():
            partition = self.partitioner(loc_logits.sigmoid())
        loc_boxes = partition["boxes"]

        boxes = loc_boxes
        bg_mask = boxes_to_bg_mask(boxes, image_hw)

        local_image = crop_and_resize_batch(image, boxes, image_hw)

        local_logits_dict = self.body_local(
            local_image,
            normed_class_embs,
            global_context=fused,
            bg_mask=bg_mask.detach(),
        )

        local_seg_logits = local_logits_dict[f"seg{self.iterations}"]
        seg_logits = paste_to_full(local_seg_logits, boxes, image_hw, base_logits=None)

        cls_logits = self._maybe_checkpoint(self.map_classifier, seg_logits, image_deep, normed_class_embs, fused)
        
        return {
            "loc": loc_logits,
            "loc_sides": loc_side_logits,
            "seg": seg_logits,
            "local": local_logits_dict,
            "boxes": boxes,
            "cls": cls_logits,
        }

    def train_forward(self, data, gt_classes, class_names: list, **kwargs):
        image = data["image"]

        if self.train_class_embs is None:
            self.train_class_embs = self.clip.get_text_embs_by_template(class_names)
        normed_class_embs = self.train_class_embs  # Nc,D

        mask = data["mask"]

        outputs = self._forward_pipeline(image, normed_class_embs)

        losses = []
        loss_str = []

        with torch.no_grad():
            edge_ks = 5
            eroded_mask = -F.max_pool2d(-mask, kernel_size=edge_ks, stride=1, padding=edge_ks // 2)
            dilated_mask = F.max_pool2d(mask, kernel_size=edge_ks, stride=1, padding=edge_ks // 2)
            edge = dilated_mask - eroded_mask
            edge = edge.gt(0).float()

        loc_loss, _ = localization_loss(outputs["loc_sides"], mask)
        losses.append(self.loc_loss_weight * loc_loss)
        loss_str.append(f"loc: {loc_loss.item():.5f}")

        local_mask = crop_and_resize_batch(mask, outputs["boxes"], image.shape[-2:])
        local_edge = crop_and_resize_batch(edge, outputs["boxes"], image.shape[-2:])

        local_logits = outputs["local"]
        for i in range(1, self.iterations + 1):
            iter_weight = 1.0 / (2 ** (self.iterations - i))

            s3 = seg_loss(logits=resize_to(local_logits[f"seg{i}-3"], local_mask.shape[2:]), mask=local_mask)
            s2 = seg_loss(logits=resize_to(local_logits[f"seg{i}-2"], local_mask.shape[2:]), mask=local_mask)
            s1 = seg_loss(logits=resize_to(local_logits[f"seg{i}-1"], local_mask.shape[2:]), mask=local_mask)
            losses.append(self.seg_loss_weight * iter_weight * s3)
            losses.append(self.seg_loss_weight * iter_weight * s2)
            losses.append(self.seg_loss_weight * iter_weight * s1)
            loss_str.append(f"seg3: {s3.item():.5f}")
            loss_str.append(f"seg2: {s2.item():.5f}")
            loss_str.append(f"seg1: {s1.item():.5f}")

            e_dice3 = edge_dice_loss(logits=resize_to(local_logits[f"edg{i}-3"], local_edge.shape[2:]), edge=local_edge)
            e_dice2 = edge_dice_loss(logits=resize_to(local_logits[f"edg{i}-2"], local_edge.shape[2:]), edge=local_edge)
            e_dice1 = edge_dice_loss(logits=resize_to(local_logits[f"edg{i}-1"], local_edge.shape[2:]), edge=local_edge)
            losses.append(self.edge_loss_weight * iter_weight * e_dice3)
            losses.append(self.edge_loss_weight * iter_weight * e_dice2)
            losses.append(self.edge_loss_weight * iter_weight * e_dice1)
            loss_str.append(f"e_dice3: {e_dice3.item():.5f}")
            loss_str.append(f"e_dice2: {e_dice2.item():.5f}")
            loss_str.append(f"e_dice1: {e_dice1.item():.5f}")
            
        if outputs.get("seg", None) is not None:
            global_seg_loss = seg_loss(logits=outputs["seg"], mask=mask)
            losses.append(self.seg_loss_weight * global_seg_loss)
            loss_str.append(f"seg_global: {global_seg_loss.item():.5f}")

        return dict(
            loss=sum(losses),
            loss_str=" ".join(loss_str),
        )


    def test_forward(self, data, class_names: list, **kwargs):
        image = data["image"]

        if self.test_class_embs is None:
            self.test_class_embs = self.clip.get_text_embs_by_template(class_names)
        normed_class_embs = self.test_class_embs

        outputs = self._forward_pipeline(image, normed_class_embs)
        seg_logits = outputs["seg"]


        map_cls_logits = outputs["cls"]
        cls_id_per_image = torch.argmax(map_cls_logits, dim=-1)
        pred_classes = [class_names[i] for i in cls_id_per_image]

        return dict(
            prob=seg_logits.sigmoid(),
            classes=pred_classes,
        )

    def forward(self, *arg, **kwargs):
        if self.training:
            return self.train_forward(*arg, **kwargs)
        else:
            return self.test_forward(*arg, **kwargs)

    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if name.startswith("clip."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            else:
                param.requires_grad = True
                param_groups["retrained"].append(param)
        logger.info(
            f"Parameter Groups:{{"
            f"Pretrained: {len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, "
            f"ReTrained: {len(param_groups['retrained'])}}}"
        )
        return param_groups
