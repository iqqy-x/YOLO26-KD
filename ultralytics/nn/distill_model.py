# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from contextlib import ExitStack
from math import isfinite
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules.head import Detect
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import dist2bbox, make_anchors
from ultralytics.utils.torch_utils import autocast, copy_attr

from .tasks import load_checkpoint


class CanAttentionMatrix(nn.Module):
    """Compute the official CanKD dot-product affinity between student queries and teacher keys."""

    def __init__(self, in_channels: int, inter_channels: int, dimension: int = 3, sub_sample: bool = True) -> None:
        """Build query/key embeddings and optional teacher-key pooling as in CanKD's attentionMatrix."""
        super().__init__()
        assert dimension in [1, 2, 3]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=2)
        self.theta = conv_nd(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        if sub_sample:
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        """Return dot products divided by the number of teacher reference positions, without softmax."""
        assert y_s.shape == y_t.shape
        batch_size = y_s.size(0)
        theta_y_s = self.theta(y_s).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        phi_y_t = self.phi(y_t).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_y_s, phi_y_t)
        return f / f.size(-1)


class CanNonLocalAttention(nn.Module):
    """Transform student features with teacher values and the official CanKD residual connection."""

    def __init__(
        self,
        in_channels: int,
        inter_channels: int | None = None,
        dimension: int = 3,
        sub_sample: bool = True,
        bn_layer: bool = True,
    ) -> None:
        """Build the Can block, including the zero-initialized output projection or BatchNorm scale."""
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.att = CanAttentionMatrix(in_channels, inter_channels, dimension, sub_sample)
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=2)
            bn = nn.BatchNorm1d
        self.g = conv_nd(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(inter_channels, in_channels, kernel_size=1, stride=1, padding=0), bn(in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(inter_channels, in_channels, kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)

    def forward(self, y_s: torch.Tensor, y_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the enhanced student and unchanged teacher features."""
        assert y_s.shape == y_t.shape
        batch_size = y_s.size(0)
        f_div_C = self.att(y_s, y_t)
        g_y_t = self.g(y_t).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        y_s_ = torch.matmul(f_div_C, g_y_t).permute(0, 2, 1).contiguous()
        y_s_ = y_s_.view(batch_size, self.inter_channels, *y_s.size()[2:])
        return self.W(y_s_) + y_s, y_t


class CanKDLoss(nn.Module):
    """Port of CanKD's NLKD_IN_Loss: instance-normalized MSE / 2 with trainable cross-attention.

    The operations and state-dict layout follow ``CanKD/mmrazor/models/losses/nlkd_IN_loss.py`` from
    https://github.com/tori-hotaru/CanKD (Apache-2.0). Alignment is external to this loss, as in the official configs.
    """

    def __init__(
        self,
        in_channels: int,
        inter_channels: int | None = None,
        dimension: int = 2,
        sub_sample: bool = True,
        bn_layer: bool = True,
        tau: float = 1.0,
        loss_weight: float = 1.0,
    ) -> None:
        """Initialize with the official defaults; tau is retained but unused by the original computation."""
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels else in_channels // 2
        if self.inter_channels == 0:
            self.inter_channels = 1
        self.tau = tau
        self.loss_weight = loss_weight
        self.non_local_att = CanNonLocalAttention(in_channels, self.inter_channels, dimension, sub_sample, bn_layer)
        self.in_norm = nn.InstanceNorm2d(in_channels, affine=False)

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalize each sample/channel exactly as the official InstanceNorm2d layer."""
        return self.in_norm(feat)

    def forward(self, s_feature: torch.Tensor, t_feature: torch.Tensor) -> torch.Tensor:
        """Compute the official mean MSE / 2, applying loss_weight exactly once."""
        assert s_feature.shape == t_feature.shape
        z_s, z_t = self.non_local_att(s_feature, t_feature)
        n_z_s, n_z_t = self.norm(z_s), self.norm(z_t)
        loss = F.mse_loss(n_z_s, n_z_t) / 2
        return loss * self.loss_weight


class CWDLoss(nn.Module):
    """Channel-wise KL distillation, following Equation (4) of https://arxiv.org/abs/2011.13256.

    Each channel defines a distribution over spatial positions. Alignment is external, and normalization uses the
    aligned channel count, as in TorchDistiller's CriterionCWD with norm_type="channel" and divergence="kl".

    Attributes:
        temperature (float): Positive softmax temperature.
        loss_weight (float): Multiplier applied once to the channel-averaged KL loss.
    """

    def __init__(self, temperature: float = 1.0, loss_weight: float = 1.0):
        """Initialize the spatial softmax temperature and distillation weight."""
        super().__init__()
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("cwd_temperature must be finite and greater than 0")
        self.temperature = temperature
        self.loss_weight = loss_weight

    def forward(self, s_feature: torch.Tensor, t_feature: torch.Tensor) -> torch.Tensor:
        """Return weighted KL(teacher || student), summed spatially and averaged over samples and aligned channels."""
        if s_feature.ndim != 4 or s_feature.shape != t_feature.shape:
            raise ValueError("CWD requires aligned student/teacher features with matching (N, C, H, W) shapes")
        n, c = s_feature.shape[:2]
        student_log = (s_feature.float().flatten(2) / self.temperature).log_softmax(-1)
        teacher_prob = (t_feature.detach().float().flatten(2) / self.temperature).softmax(-1)
        loss = F.kl_div(student_log, teacher_prob, reduction="sum")
        return loss * (self.temperature**2) * self.loss_weight / (n * c)


class BTKDLoss(nn.Module):
    """BTKD++ detection distillation with probability rectification and Gaussian box distributions.

    Follows Sections 3.3-4.1 and Appendix A of https://doi.org/10.1007/s11263-026-02924-x. Anchor-free
    boxes use image-normalized xyxy coordinates in [0, 1], rather than anchor-relative offset ranges.
    """

    def __init__(self):
        """Use temperature 2, IoU threshold 0.65, 128 bins, and a Gaussian width of two bins."""
        super().__init__()
        self.temperature = 2.0
        self.iou_threshold = 0.65
        self.gamma = 0.0
        self.register_buffer("centers", (torch.arange(128, dtype=torch.float32) + 0.5) / 128)

    @staticmethod
    def rectify(prob: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Rectify the wrong/GT probability pair while preserving its mass and all other entries."""
        wrong = prob.argmax(-1, keepdim=True)
        labels = labels.unsqueeze(-1)
        p_gt, p_wrong = prob.gather(-1, labels), prob.gather(-1, wrong)
        mass = p_gt + p_wrong
        scale = mass / (1 + mass)
        rectified = prob.clone()
        rectified.scatter_(-1, labels, torch.where(wrong != labels, (p_gt + 1) * scale, p_gt))
        rectified.scatter_(-1, wrong, torch.where(wrong != labels, p_wrong * scale, p_wrong))
        return rectified

    def distill(
        self, student_log: torch.Tensor, teacher_prob: torch.Tensor, labels: torch.Tensor, easy: torch.Tensor
    ) -> torch.Tensor:
        """Return separately normalized easy/hard KL losses with the linear curriculum weights."""
        target = torch.where(easy.unsqueeze(-1), teacher_prob, self.rectify(teacher_prob, labels))
        kl = F.kl_div(student_log, target, reduction="none").sum(-1)
        hard = ~easy
        return (1 - self.gamma) * (kl * easy).sum() / easy.sum().clamp(min=1) + self.gamma * (
            kl * hard
        ).sum() / hard.sum().clamp(min=1)

    @staticmethod
    def decode(preds: dict, head: Detect, foreground: torch.Tensor) -> torch.Tensor:
        """Decode only assigned positive boxes to pixel xyxy, retaining student coordinate gradients."""
        anchors, strides = make_anchors(preds["feats"], head.stride, 0.5)
        batch_size = foreground.shape[0]
        anchors = anchors.unsqueeze(0).expand(batch_size, -1, -1)[foreground].float()
        strides = strides.unsqueeze(0).expand(batch_size, -1, -1)[foreground].float()
        distances = preds["boxes"].permute(0, 2, 1)[foreground].float()
        if head.reg_max > 1:
            distances = (
                distances.reshape(-1, 4, head.reg_max)
                .softmax(-1)
                .matmul(torch.arange(head.reg_max, device=distances.device, dtype=torch.float32))
            )
        return dist2bbox(distances, anchors, xywh=False) * strides

    def forward(
        self,
        student: dict,
        teacher: dict,
        assigned: tuple,
        student_head: Detect,
        teacher_head: Detect,
        image_shape: tuple[int, int],
    ) -> torch.Tensor:
        """Distill class and four coordinate distributions at the native assignment's positive grid positions."""
        if [f.shape[-2:] for f in student["feats"]] != [f.shape[-2:] for f in teacher["feats"]]:
            raise ValueError("BTKD++ requires matching teacher/student feature grids")
        foreground = assigned[3].bool()
        labels, gt_boxes = assigned[0][foreground].long(), assigned[1][foreground].float()
        s_scores = student["scores"].permute(0, 2, 1)[foreground].float()
        t_scores = teacher["scores"].detach().permute(0, 2, 1)[foreground].float()
        s_boxes = self.decode(student, student_head, foreground)
        if student_head.nc == 1:
            # A binary background/foreground distribution avoids a degenerate one-class softmax.
            s_scores = torch.cat((torch.zeros_like(s_scores), s_scores), -1)
        with torch.no_grad():
            t_boxes = self.decode(teacher, teacher_head, foreground)
            if student_head.nc == 1:
                t_scores = torch.cat((torch.zeros_like(t_scores), t_scores), -1)
                labels = torch.ones_like(labels)
            easy = (t_scores.argmax(-1) == labels) & (
                bbox_iou(t_boxes, gt_boxes, xywh=False).squeeze(-1) >= self.iou_threshold
            )
        cls_loss = self.distill(
            (s_scores / self.temperature).log_softmax(-1),
            (t_scores / self.temperature).softmax(-1),
            labels,
            easy,
        )
        height, width = image_shape
        scale = s_boxes.new_tensor((width, height, width, height))
        s_boxes, t_boxes, gt_boxes = ((boxes / scale).clamp(0, 1) for boxes in (s_boxes, t_boxes, gt_boxes))
        # The paper specifies sigma in bins, not in image-normalized coordinate units.
        s_log = (-0.5 * ((s_boxes.unsqueeze(-1) - self.centers.float()) / (2 / 128)).square()).log_softmax(-1)
        t_prob = (-0.5 * ((t_boxes.unsqueeze(-1) - self.centers.float()) / (2 / 128)).square()).softmax(-1)
        gt_bins = (gt_boxes * 128).long().clamp(max=127)
        # Treat the four coordinates independently, summing their mean distillation losses.
        box_loss = self.distill(s_log, t_prob, gt_bins, easy.unsqueeze(-1).expand_as(gt_bins)) * 4
        return cls_loss + box_loss


class FeatureHook:
    """Picklable forward hook that stores layer output into a shared dict."""

    def __init__(self, feat_dict: dict, idx: int) -> None:
        """Initialize the hook with the shared feature dict and the key (feature level) to store outputs under."""
        self.feat_dict = feat_dict
        self.idx = idx

    def __call__(self, module: nn.Module, inputs: tuple, output) -> None:
        """Store the layer's forward output into the shared feature dict under its index.

        The output is a tensor for neck layers but a tuple/dict for the Detect head, so it is left untyped.
        """
        self.feat_dict[self.idx] = output


class DistillationModel(nn.Module):
    """YOLO knowledge distillation model.

    This class wraps a teacher-student pair for knowledge distillation training. Features are extracted from both models
    via forward hooks for distillation.

    Attributes:
        teacher_model (nn.Module): Frozen teacher model providing features.
        student_model (nn.Module): Trainable student model being distilled.
        feats_idx (list): Student layer indices for feature extraction.
        teacher_feats_idx (list): Teacher layer indices at the same feature levels.
        projector (nn.ModuleList): Per-level MLP projectors, or CanKD/CWD channel aligners and losses.
        distill_method (str): Distillation method: "sl2", "cankd", "cwd", or "btkd" (BTKD++ detection).
        dis (float): Distillation loss weight factor.

    Methods:
        get_distill_layers: Auto-detect distillation feature layers from the Detect head.
        forward: Run the student model, or compute the combined loss when given a training batch.
        loss: Compute combined detection and distillation loss.
        loss_sl2: Compute score-weighted L2 distillation loss for a feature pair.
        decouple_outputs: Normalize teacher/student head outputs across train/val formats.
        fuse: Fuse and return the student model for inference and export.
        train: Set training mode while keeping teacher frozen.

    Examples:
        Train a student model with knowledge distillation from a larger teacher (the trainer builds the
        DistillationModel internally when the ``distill_model`` argument is set)
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")
        >>> model.train(data="coco8.yaml", distill_model="yolo26s.pt")
    """

    def __init__(self, teacher_model: str | Path | nn.Module, student_model: nn.Module):
        """Initialize the distillation model with teacher, student, and feature extraction hooks.

        Args:
            teacher_model (str | Path | nn.Module): Teacher model checkpoint path or module.
            student_model (nn.Module): Student model module to be trained.
        """
        super().__init__()
        ch = student_model.yaml.get("channels", 3)
        if isinstance(teacher_model, (str, Path)):
            teacher_model = load_checkpoint(teacher_model)[0]
            if teacher_model.yaml.get("channels", 3) != ch:
                weights = teacher_model
                teacher_model = type(weights)(weights.yaml.copy(), ch=ch, nc=weights.yaml["nc"], verbose=False)
                teacher_model.load(weights)
        device = next(student_model.parameters()).device
        self.teacher_model = teacher_model.to(device)
        self._freeze_teacher()
        self.student_model = student_model
        self.distill_method = getattr(student_model.args, "distill_method", "sl2")
        if self.distill_method not in {"sl2", "cankd", "cwd", "btkd"}:
            raise ValueError("distill_method must be 'sl2', 'cankd', 'cwd', or 'btkd'")
        self.feats_idx = self.get_distill_layers(student_model)
        self.teacher_feats_idx = self.get_distill_layers(teacher_model)
        if len(self.feats_idx) != len(self.teacher_feats_idx):
            raise ValueError("Teacher and student detection heads must have the same number of feature levels")

        # Hook-based feature capture: identical for teacher and student
        self._teacher_feats: dict[int, torch.Tensor] = {}
        self._student_feats: dict[int, torch.Tensor] = {}
        self._teacher_hooks: list = []
        self._student_hooks: list = []
        if self.distill_method == "btkd":
            student_head, teacher_head = student_model.model[-1], teacher_model.model[-1]
            if type(student_head) is not Detect or type(teacher_head) is not Detect:
                raise ValueError("BTKD++ currently supports object detection with Detect heads only")
            if student_head.nc != teacher_head.nc or student_model.names != teacher_model.names:
                raise ValueError("BTKD++ requires identical teacher/student classes and class order")
            if not torch.equal(student_head.stride, teacher_head.stride):
                raise ValueError("BTKD++ requires matching teacher/student detection strides")
            dual_head = getattr(student_head, "one2one_cv2", None) is not None
            if dual_head != (getattr(teacher_head, "one2one_cv2", None) is not None):
                raise ValueError("BTKD++ requires matching teacher/student prediction branches")
            heads = ("cv2", "cv3", "one2one_cv2", "one2one_cv3") if dual_head else ("cv2", "cv3")
            if any(getattr(head, name, None) is None for head in (student_head, teacher_head) for name in heads):
                raise ValueError("BTKD++ requires unfused teacher/student detection heads")
            copy_attr(self, student_model)
            self.dis = student_model.args.dis
            # Keep the existing checkpoint/resume contract without trainable KD layers.
            self.projector = nn.ModuleList()
            self.btkd = BTKDLoss().to(device)
            return
        if self.distill_method == "cwd":
            if type(student_model.model[-1]) is not Detect or type(teacher_model.model[-1]) is not Detect:
                raise ValueError("CWD currently supports object detection with Detect heads only")
            if not torch.equal(student_model.model[-1].stride, teacher_model.model[-1].stride):
                raise ValueError("CWD requires matching teacher/student feature strides")
        self._register_feature_hooks()

        # Get feature dimensions via dummy forward pass (hooks capture outputs)
        imgsz = student_model.args.imgsz
        student_model.eval()
        with torch.no_grad():
            im = torch.zeros(2, ch, imgsz, imgsz, device=device)
            teacher_model(im)
            student_model(im)
        student_model.train()
        teacher_output = [self._teacher_feats[i] for i in range(len(self.feats_idx))]
        student_output = [self._student_feats[i] for i in range(len(self.feats_idx))]

        copy_attr(self, student_model)
        self.dis = self.student_model.args.dis
        projectors = []
        for student_out, teacher_out in zip(student_output[:-1], teacher_output[:-1]):
            student_dim = self.decouple_outputs(student_out).shape[1]
            teacher_dim = self.decouple_outputs(teacher_out).shape[1]
            if self.distill_method in {"cankd", "cwd"}:
                align = nn.Conv2d(student_dim, teacher_dim, 1) if student_dim != teacher_dim else nn.Identity()
                if self.distill_method == "cankd" and isinstance(align, nn.Conv2d):
                    # Match the official ConvModuleConnector with norm_cfg=None and act_cfg=None.
                    nn.init.kaiming_normal_(align.weight, a=0, mode="fan_out", nonlinearity="relu")
                    nn.init.constant_(align.bias, 0)
                projectors.append(
                    nn.ModuleDict(
                        {
                            "align": align,
                            "loss": CanKDLoss(in_channels=teacher_dim, loss_weight=self.dis)
                            if self.distill_method == "cankd"
                            else CWDLoss(temperature=student_model.args.cwd_temperature, loss_weight=self.dis),
                        }
                    )
                )
                continue
            projectors.append(
                nn.Sequential(
                    nn.Conv2d(student_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(teacher_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
                )
            )
        self.projector = nn.ModuleList(projectors).to(device)

    def __getstate__(self):
        """Return a copy of state for pickling without captured features or hook handles.

        Clears the feature dicts in place (rather than replacing the attributes) because the registered
        FeatureHooks share these exact dict objects; otherwise a deepcopy/pickle of a mid-training model would
        still reach the hook-held tensors (which carry grad_fn and cannot be deep-copied).
        """
        self._teacher_feats.clear()
        self._student_feats.clear()
        state = self.__dict__.copy()
        state["_teacher_hooks"] = []
        state["_student_hooks"] = []
        return state

    def __setstate__(self, state):
        """Clear stale features and hooks, and re-register forward hooks after unpickling."""
        state.setdefault("teacher_feats_idx", state["feats_idx"])  # checkpoints saved before teacher_feats_idx
        state.setdefault("distill_method", "sl2")
        self.__dict__.update(state)
        self._teacher_feats = {}
        self._student_feats = {}
        self._register_feature_hooks()

    def _remove_feature_hooks(self) -> None:
        """Remove any previously registered feature-capture hooks."""
        for handle in self._student_hooks:
            handle.remove()
        self._student_hooks.clear()
        if self.teacher_model is not None:
            for handle in self._teacher_hooks:
                handle.remove()
            self._teacher_hooks.clear()

    @staticmethod
    def _clear_feature_hooks(module: nn.Module) -> None:
        """Remove any FeatureHook instances from a module's forward hooks."""
        for handle_id, hook in list(module._forward_hooks.items()):
            if isinstance(hook, FeatureHook):
                del module._forward_hooks[handle_id]

    def _register_feature_hooks(self) -> None:
        """Register feature-capture hooks, removing stale FeatureHook instances first."""
        self._remove_feature_hooks()
        if self.distill_method == "btkd":
            return
        for i, (s, t) in enumerate(zip(self.feats_idx, self.teacher_feats_idx)):  # features stored by level i
            self._clear_feature_hooks(self.student_model.model[s])
            self._student_hooks.append(
                self.student_model.model[s].register_forward_hook(FeatureHook(self._student_feats, i))
            )
            if self.teacher_model is not None:
                self._clear_feature_hooks(self.teacher_model.model[t])
                self._teacher_hooks.append(
                    self.teacher_model.model[t].register_forward_hook(FeatureHook(self._teacher_feats, i))
                )

    @staticmethod
    def get_distill_layers(model: nn.Module) -> list[int]:
        """Auto-detect distillation feature layers from the model's Detect head.

        Returns the Detect head's input layer indices plus the head layer index itself.
        E.g. YOLO26 -> [16, 19, 22, 23], YOLOv8 -> [15, 18, 21, 22].

        Args:
            model (nn.Module): Model whose `model` layers contain a Detect head.

        Returns:
            (list[int]): Detect head input layer indices followed by the head layer index.

        Raises:
            ValueError: If the model has no Detect head.
        """
        for m in model.model:
            if isinstance(m, Detect):
                return [*list(m.f), m.i]
        raise ValueError("No Detect head found in model")

    def _freeze_teacher(self):
        """Keep teacher fixed for distillation."""
        if self.teacher_model is None:
            return
        self.teacher_model.eval()
        for v in self.teacher_model.parameters():
            if v.requires_grad:
                v.requires_grad = False

    def train(self, mode: bool = True):
        """Set model train mode while keeping teacher frozen in eval mode."""
        super().train(mode)
        self._freeze_teacher()
        return self

    def forward(self, x, *args, **kwargs):
        """Run the student model, or compute the combined loss when given a training batch.

        Args:
            x (torch.Tensor | dict): Input image tensor, or a batch dict with images and labels for loss computation.
            *args (Any): Additional positional arguments passed to `loss()` or the student `predict()`.
            **kwargs (Any): Additional keyword arguments passed to `loss()` or the student `predict()`.

        Returns:
            (Any): Loss tuple if x is a dict, otherwise student model predictions.
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.student_model.predict(x, *args, **kwargs)

    def fuse(self, verbose: bool = True, imgsz: int | list[int] = 640):
        """Fuse and return the student model, dropping the training-only distillation wrapper.

        Args:
            verbose (bool): Whether to print model information after fusion.
            imgsz (int | list[int]): Input image size used for FLOPs calculation.

        Returns:
            (nn.Module): The fused student model.
        """
        self._remove_feature_hooks()
        return self.student_model.fuse(verbose=verbose, imgsz=imgsz)

    def loss(self, batch, preds=None):
        """Compute combined detection and distillation loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Student predictions, used only in validation mode.

        Returns:
            loss (torch.Tensor): Student loss with the distillation loss appended.
            loss_items (dict[str, torch.Tensor]): Detached loss components, including `dis_loss`.
        """
        loss_distill = torch.zeros(1, device=batch["img"].device)
        if not self.training:  # for loss calculation during validation while training
            if preds is None:
                preds = self.student_model(batch["img"])
            regular_loss, loss_items = self.student_model.loss(batch, preds)
            loss_items["dis_loss"] = loss_distill.detach()
            return torch.cat([regular_loss, loss_distill]), loss_items

        # Clear feature dicts before forward passes
        self._teacher_feats.clear()
        self._student_feats.clear()

        with torch.no_grad():
            teacher_preds = self.teacher_model(batch["img"])  # hooks capture teacher features for feature KD
        preds = self.student_model(batch["img"])  # hooks capture student features

        if self.distill_method == "btkd":
            return self.loss_btkd(batch, preds, teacher_preds)

        regular_loss, loss_items = self.student_model.loss(batch, preds)
        n = len(self.feats_idx) - 1  # neck levels; level n is the Detect head
        if self.distill_method == "sl2":
            teacher_head_feat = self._teacher_feats[n]
            teacher_scores = (
                self.decouple_outputs(teacher_head_feat, branch="one2many")["scores"]
                + self.decouple_outputs(teacher_head_feat, branch="one2one")["scores"]
            ) / 2
            # neck feature sizes vary per batch (e.g. multi_scale), so split scores by the live teacher feats
            neck_feats = [self._teacher_feats[i] for i in range(n)]
            parts = torch.split(teacher_scores, [f.shape[-2] * f.shape[-1] for f in neck_feats], dim=-1)
            teacher_scores = tuple(p.sigmoid().max(dim=1, keepdim=True).values for p in parts)
        for i in range(n):
            teacher_feat = self.decouple_outputs(self._teacher_feats[i])
            student_feat = self.decouple_outputs(self._student_feats[i])
            if self.distill_method in {"cankd", "cwd"}:
                level = self.projector[i]
                # Keep alignment and feature distillation losses in FP32 under AMP.
                with autocast(enabled=False, device=student_feat.device.type):
                    loss_distill += level["loss"](level["align"](student_feat.float()), teacher_feat.float())
            else:
                student_feat = self.projector[i](student_feat)
                loss_distill += (
                    self.loss_sl2(student_feat, teacher_feat, feat_idx=i, teacher_scores=teacher_scores) * self.dis
                )

        loss_items["dis_loss"] = loss_distill.detach()
        loss_distill = loss_distill * batch["img"].shape[0]
        return torch.cat([regular_loss, loss_distill]), loss_items

    def loss_btkd(self, batch: dict, preds: dict, teacher_preds: dict | tuple) -> tuple:
        """Observe native assignments once per head and add BTKD++ without modifying the native criterion."""
        if getattr(self.student_model, "criterion", None) is None:
            self.student_model.criterion = self.student_model.init_criterion()
        criterion = self.student_model.criterion
        branches = (
            (("one2many", criterion.one2many, criterion.o2m), ("one2one", criterion.one2one, criterion.o2o))
            if "one2many" in preds
            else ((None, criterion, 1.0),)
        )
        assigned = {}
        with ExitStack() as hooks:
            for i, (_, branch_loss, _) in enumerate(branches):
                hooks.callback(branch_loss.assigner.register_forward_hook(FeatureHook(assigned, i)).remove)
            regular_loss, loss_items = self.student_model.loss(batch, preds)
        teacher_preds = teacher_preds[1] if isinstance(teacher_preds, tuple) else teacher_preds
        loss_distill = regular_loss.new_zeros(1)
        with autocast(enabled=False, device=batch["img"].device.type):
            for i, (branch, _, weight) in enumerate(branches):
                loss_distill += (
                    weight
                    * self.dis
                    * self.btkd(
                        preds[branch] if branch else preds,
                        teacher_preds[branch] if branch else teacher_preds,
                        assigned[i],
                        self.student_model.model[-1],
                        self.teacher_model.model[-1],
                        batch["img"].shape[-2:],
                    )
                )
        loss_items["dis_loss"] = loss_distill.detach()
        return torch.cat((regular_loss, loss_distill * batch["img"].shape[0])), loss_items

    def loss_sl2(
        self, student_feat: torch.Tensor, teacher_feat: torch.Tensor, feat_idx: int, teacher_scores: tuple
    ) -> torch.Tensor:
        """Compute score-weighted L2 distillation loss for a feature pair.

        Args:
            student_feat (torch.Tensor): Student feature tensor of shape (N, C, H, W).
            teacher_feat (torch.Tensor): Teacher feature tensor of shape (N, C, H, W).
            feat_idx (int): Index of the feature level for selecting teacher scores.
            teacher_scores (tuple): Tuple of score tensors for each feature level.

        Returns:
            (torch.Tensor): The computed score-weighted L2 loss.
        """
        teacher_score = teacher_scores[feat_idx]
        n, c = student_feat.shape[:2]
        student_feat = student_feat.view(n, c, -1)
        teacher_feat = teacher_feat.view(n, c, -1)
        mse = F.mse_loss(student_feat, teacher_feat, reduction="none")
        weighted_mse = (mse * teacher_score).sum() / (teacher_score.sum() * c + 1e-9)
        return weighted_mse

    @property
    def criterion(self):
        """Get the criterion from the student model."""
        return self.student_model.criterion

    @criterion.setter
    def criterion(self, value) -> None:
        """Set value for student criterion."""
        self.student_model.criterion = value

    def init_criterion(self):
        """Initialize the loss criterion via the student model."""
        return self.student_model.init_criterion()

    @property
    def end2end(self):
        """Expose student end-to-end mode for validator/predictor control."""
        return getattr(self.student_model, "end2end", False)

    @end2end.setter
    def end2end(self, value):
        """Forward end-to-end mode update to the student model."""
        self.student_model.end2end = value

    def set_head_attr(self, **kwargs):
        """Forward head-attribute updates (e.g. max_det, agnostic_nms, end2end) to the student model."""
        self.student_model.set_head_attr(**kwargs)

    def decouple_outputs(self, preds, branch: str = "one2one"):
        """Decouple outputs for teacher/student models.

        This method handles different output formats from YOLO models, including
        tuple outputs (train/val mode), dict outputs with branches (one2one/one2many),
        and direct tensor outputs.

        Args:
            preds (torch.Tensor | tuple | dict): Model predictions in various formats.
            branch (str): Which branch to extract from dict outputs ("one2one" or "one2many").

        Returns:
            (torch.Tensor | dict): The decoupled predictions.
        """
        if isinstance(preds, tuple):  # decouple for val mode
            preds = preds[1]
        if isinstance(preds, dict) and branch in preds:
            preds = preds[branch]
        return preds
