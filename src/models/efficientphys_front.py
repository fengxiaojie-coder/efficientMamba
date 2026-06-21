"""Faithful port of EfficientPhys front-end (up to `d10` embedding).

This module implements the Motion/Appearance dual-branch network with
TSM and attention gating as in the original `model.py` from the
external EfficientPhys repository. It exposes `EfficientPhysFront` with
`forward_features(frames: Tensor[B,T,6,H,W]) -> Tensor[B,T,nb_dense]`.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor


class Attention_mask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, roi_map: Optional[Tensor] = None) -> Tensor:
        """Normalize attention map and optionally apply an ROI bias map.

        x: attention logits / pre-sigmoid or post-sigmoid map shaped (nt,1,H,W)
        roi_map: optional spatial bias map shaped (nt,1,H,W) with values in [0,1]
        """
        if roi_map is not None:
            # apply multiplicative bias to encourage attention inside ROI
            x = x * (1.0 + 0.8 * roi_map)
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / (xsum + 1e-8) * xshape[2] * xshape[3] * 0.5


class TSM(nn.Module):
    def __init__(self, n_segment: int = 10, fold_div: int = 3):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x: Tensor) -> Tensor:
        # x is flattened: (nt, c, h, w) where nt = batch * n_segment
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div if self.fold_div > 0 else 0
        out = torch.zeros_like(x)
        if fold > 0:
            out[:, :-1, :fold] = x[:, 1:, :fold]
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        else:
            out = x
        return out.view(nt, c, h, w)


class EfficientPhysFront(nn.Module):
    """Port of the EfficientPhys front until the embedding `d10`.

    Expects input frames shaped `[B, T, C, H, W]` where `C==6` (diff+raw)
    like the original implementation. Returns `[B, T, nb_dense]`.
    """

    def __init__(self, in_channels: int = 6, nb_filters1: int = 32, nb_filters2: int = 64,
                 nb_dense: int = 128, kernel_size: int = 3, frame_depth: int = 128, img_size: int = 72):
        super().__init__()
        # in_channels here is the total channels per frame (6 when using both
        # motion+appearance). Internal branch convs expect 3 channels each.
        self.in_channels = in_channels
        self.branch_in = 3
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        self.kernel_size = kernel_size
        self.frame_depth = frame_depth
        self.img_size = img_size

        # TSM modules
        self.TSM_1 = TSM(n_segment=self.frame_depth)
        self.TSM_2 = TSM(n_segment=self.frame_depth)
        self.TSM_3 = TSM(n_segment=self.frame_depth)
        self.TSM_4 = TSM(n_segment=self.frame_depth)

        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.branch_in, self.nb_filters1, kernel_size=self.kernel_size, padding=1, bias=True)
        self.motion_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=1, bias=True)
        self.motion_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)

        # Appearance branch convs
        self.apperance_conv1 = nn.Conv2d(self.branch_in, self.nb_filters1, kernel_size=self.kernel_size, padding=1, bias=True)
        self.apperance_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.apperance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=1, bias=True)
        self.apperance_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)

        # Attention gating
        self.apperance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, bias=True)
        self.attn_mask_2 = Attention_mask()

        # Pooling / dropout
        self.avg_pooling_1 = nn.AvgPool2d((2, 2))
        self.avg_pooling_2 = nn.AvgPool2d((2, 2))
        self.avg_pooling_3 = nn.AvgPool2d((2, 2))
        self.dropout_1 = nn.Dropout(0.25)
        self.dropout_2 = nn.Dropout(0.25)
        self.dropout_3 = nn.Dropout(0.25)
        self.dropout_4 = nn.Dropout(0.5)

        # Final dense matching original sizes used by EfficientPhys
        if self.img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif self.img_size == 72:
            self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)
        elif self.img_size == 96:
            self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
        else:
            raise Exception('Unsupported image size')

    def forward_features(self, frames: Tensor, roi_map: Optional[Tensor] = None, return_attention: bool = False) -> Tensor | tuple[Tensor, dict]:
        """Compute embeddings from a video clip.

        frames: [B, T, C, H, W] where C==6 (diff then raw)
        roi_map: optional spatial bias map
        return_attention: if True, also return attention maps g1 and g2 for visualization
        returns: [B, T, nb_dense] or ([B, T, nb_dense], {'g1': Tensor, 'g2': Tensor})
        """
        if frames.dim() != 5:
            raise ValueError('Expected frames [B,T,C,H,W]')

        B, T, C, H, W = frames.shape
        if C not in (3, 6):
            raise ValueError('Expected C==6 (both) or C==3')
        if T != self.frame_depth:
            raise ValueError(
                f'Expected T=={self.frame_depth} to match frame_depth, got T=={T}. '
                'Please keep preprocessing clip_len and model frame_depth consistent.'
            )

        # reshape to (nt, c, h, w) consistent with original implementation
        nt = B * T
        x = frames.reshape(nt, C, H, W)

        # split channels like original: first 3 are diff/motion, next 3 are raw/appearance
        if C == 6:
            diff_input = x[:, :3, :, :]
            raw_input = x[:, 3:, :, :]
        else:
            # if only 3 channels provided, treat as both (duplicate) to keep API usable
            diff_input = x
            raw_input = x

        # Motion stream
        diff_input = self.TSM_1(diff_input)
        d1 = torch.tanh(self.motion_conv1(diff_input))
        d1 = self.TSM_2(d1)
        d2 = torch.tanh(self.motion_conv2(d1))

        # Appearance stream
        r1 = torch.tanh(self.apperance_conv1(raw_input))
        r2 = torch.tanh(self.apperance_conv2(r1))

        # gating 1
        g1_raw = torch.sigmoid(self.apperance_att_conv1(r2))
        # prepare roi map expanded to nt if provided (resize to g1 spatial size)
        roi_map_nt1 = None
        if roi_map is not None:
            B_tmp, T_tmp = frames.shape[0], frames.shape[1]
            _, _, H1, W1 = g1_raw.shape
            roi_resized1 = torch.nn.functional.interpolate(roi_map, size=(H1, W1), mode='bilinear', align_corners=False)
            roi_map_nt1 = roi_resized1.unsqueeze(1).expand(-1, T_tmp, -1, -1, -1).reshape(-1, 1, H1, W1)
        g1 = self.attn_mask_1(g1_raw, roi_map=roi_map_nt1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)

        d4 = self.TSM_3(d4)
        d5 = torch.tanh(self.motion_conv3(d4))
        d5 = self.TSM_4(d5)
        d6 = torch.tanh(self.motion_conv4(d5))

        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))

        # gating 2
        g2_raw = torch.sigmoid(self.apperance_att_conv2(r6))
        # prepare roi_map for g2 spatial size if available
        roi_map_nt2 = None
        if roi_map is not None:
            _, _, H2, W2 = g2_raw.shape
            roi_resized2 = torch.nn.functional.interpolate(roi_map, size=(H2, W2), mode='bilinear', align_corners=False)
            roi_map_nt2 = roi_resized2.unsqueeze(1).expand(-1, T, -1, -1, -1).reshape(-1, 1, H2, W2)
        g2 = self.attn_mask_2(g2_raw, roi_map=roi_map_nt2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.reshape(d8.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))

        # reshape back to [B, T, D]
        out = d10.view(B, T, -1)

        if return_attention:
            attn_dict = {
                'g1': g1.view(B, T, 1, g1.shape[-2], g1.shape[-1]),
                'g2': g2.view(B, T, 1, g2.shape[-2], g2.shape[-1]),
            }
            return out, attn_dict
        return out


def smoke_test() -> tuple[tuple[int, ...], tuple[int, ...]]:
    model = EfficientPhysFront(in_channels=6, frame_depth=16, img_size=72)
    dummy = torch.randn(2, 16, 6, 72, 72)
    features = model.forward_features(dummy)
    return tuple(features.shape), tuple(dummy.shape)


if __name__ == '__main__':
    feats_shape, inp_shape = smoke_test()
