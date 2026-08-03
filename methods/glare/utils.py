import torch
import torch.nn.functional as F

from methods.base.base_ops import resize_to


def _normalize_box(box, h_max, w_max):
	y1, x1, y2, x2 = box.tolist()
	y1 = int(max(0, min(round(float(y1)), h_max - 1)))
	x1 = int(max(0, min(round(float(x1)), w_max - 1)))
	y2 = int(max(0, min(round(float(y2)), h_max - 1)))
	x2 = int(max(0, min(round(float(x2)), w_max - 1)))
	if y2 < y1:
		y1, y2 = y2, y1
	if x2 < x1:
		x1, x2 = x2, x1
	return y1, x1, y2, x2


def crop_and_resize_batch(tensor, boxes, out_hw):
	crops = []
	h_max, w_max = tensor.shape[-2], tensor.shape[-1]
	for i in range(tensor.shape[0]):
		y1, x1, y2, x2 = _normalize_box(boxes[i], h_max, w_max)
		crop = tensor[i : i + 1, :, y1 : y2 + 1, x1 : x2 + 1]
		crops.append(resize_to(crop, tgt_hw=out_hw))
	return torch.cat(crops, dim=0)


def paste_to_full(local_logits, boxes, full_hw, base_logits=None):
	b = local_logits.shape[0]
	if base_logits is None:
		full = local_logits.new_full((b, 1, full_hw[0], full_hw[1]), -20.0)
	else:
		full = base_logits.clone()

	h_max, w_max = full_hw
	for i in range(b):
		y1, x1, y2, x2 = _normalize_box(boxes[i], h_max, w_max)
		patch = resize_to(local_logits[i : i + 1], tgt_hw=(y2 - y1 + 1, x2 - x1 + 1))
		full[i : i + 1, :, y1 : y2 + 1, x1 : x2 + 1] = patch
	return full


def boxes_to_bg_mask(boxes, full_hw):
	h_max, w_max = int(full_hw[0]), int(full_hw[1])
	bg_masks = []
	for i in range(boxes.shape[0]):
		y1, x1, y2, x2 = _normalize_box(boxes[i], h_max, w_max)
		bg = torch.ones((1, h_max, w_max), dtype=torch.float32, device=boxes.device)
		bg[:, y1 : y2 + 1, x1 : x2 + 1] = 0.0
		bg_masks.append(bg)
	return torch.stack(bg_masks, dim=0)


def draw_boxes_on_image(image, boxes, color=(1.0, 0.0, 0.0), thickness=4):
	vis = image.detach().clone()
	b, c, h, w = vis.shape

	if c == 1:
		vis = vis.repeat(1, 3, 1, 1)
	elif c > 3:
		vis = vis[:, :3, :, :]

	draw_color = torch.tensor(color, dtype=vis.dtype, device=vis.device).view(1, 3, 1, 1)
	t = max(1, int(thickness))

	for i in range(min(b, boxes.shape[0])):
		y1, x1, y2, x2 = _normalize_box(boxes[i], h, w)
		vis[i : i + 1, :, y1 : min(h, y1 + t), x1 : x2 + 1] = draw_color
		vis[i : i + 1, :, max(0, y2 - t + 1) : y2 + 1, x1 : x2 + 1] = draw_color
		vis[i : i + 1, :, y1 : y2 + 1, x1 : min(w, x1 + t)] = draw_color
		vis[i : i + 1, :, y1 : y2 + 1, max(0, x2 - t + 1) : x2 + 1] = draw_color

	return vis.clamp(0.0, 1.0)


def compute_bg_proto(global_context, bg_mask, eps=1e-6):
	bg_mask_resized = F.interpolate(bg_mask.float(), size=global_context.shape[-2:], mode="nearest")
	denom = bg_mask_resized.sum(dim=(-1, -2), keepdim=True).clamp_min(eps)
	bg_proto = (global_context * bg_mask_resized).sum(dim=(-1, -2), keepdim=True) / denom
	return bg_proto
