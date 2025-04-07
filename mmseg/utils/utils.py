import contextlib

import numpy as np
import torch
import torch.nn.functional as F


@contextlib.contextmanager
def np_local_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def downscale_label_ratio(gt, scale_factor, min_ratio, n_classes, ignore_index=255):
    assert scale_factor > 1
    bs, orig_c, orig_h, orig_w = gt.shape
    assert orig_c == 1
    trg_h, trg_w = orig_h // scale_factor, orig_w // scale_factor
    ignore_substitute = n_classes

    out = gt.clone()  # otw. next line would modify original gt
    out[out == ignore_index] = ignore_substitute
    out = F.one_hot(out.squeeze(1), num_classes=n_classes + 1).permute(0, 3, 1, 2)
    assert list(out.shape) == [bs, n_classes + 1, orig_h, orig_w], out.shape
    out = F.avg_pool2d(out.float(), kernel_size=scale_factor)
    gt_ratio, out = torch.max(out, dim=1, keepdim=True)
    out[out == ignore_substitute] = ignore_index
    out[gt_ratio < min_ratio] = ignore_index
    assert list(out.shape) == [bs, 1, trg_h, trg_w], out.shape
    return out


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.T
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return np.stack(b, axis=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.T
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return np.stack(b, axis=-1)


def box_noise(boxes, box_noise_scale=0.0, a_min=0, a_max=256):
    known_bbox_expand = box_xyxy_to_cxcywh(boxes)

    diff = np.zeros_like(known_bbox_expand)
    diff[:, :2] = known_bbox_expand[:, 2:] / 2
    diff[:, 2:] = known_bbox_expand[:, 2:]
    known_bbox_expand += (
        (np.random.rand(*known_bbox_expand.shape) * 2 - 1.0) * diff * box_noise_scale
    )
    boxes = box_cxcywh_to_xyxy(known_bbox_expand)
    boxes = np.clip(boxes, a_min=a_min, a_max=a_max)

    return boxes
