import torch
import torch.nn.functional as F
from torch import nn
from nets import resnet
from nets.roi_pooling import MultiScaleRoIAlign
from nets.common import FrozenBatchNorm2d
from losses.commons import IOULoss
from utils.boxs_utils import box_iou
from torchvision.ops.boxes import batched_nms

default_cfg = {
    "num_cls": 80,
    "backbone": "resnet18",
    "pretrained": True,
    "reduction": False,
    "norm_layer": None,
    "fpn_channel": 256,
    "fpn_bias": True,
    "anchor_sizes": [32., 64., 128., 256., 512.],
    "anchor_scales": [2 ** 0, ],
    "anchor_ratios": [0.5, 1.0, 2.0],
    "strides": [4., 8., 16., 32., 64.],
    "rpn_pre_nms_top_n_train": 2000,
    "rpn_post_nms_top_n_train": 2000,
    "rpn_pre_nms_top_n_test": 1000,
    "rpn_post_nms_top_n_test": 1000,
    "rpn_fg_iou_thresh": 0.7,
    "rpn_bg_iou_thresh": 0.3,
    "rpn_nms_thresh": 0.7,
    "rpn_batch_size_per_image": 256,
    "rpn_positive_fraction": 0.5,

    "box_fg_iou_thresh": 0.5,
    "box_bg_iou_thresh": 0.5,
    "box_batch_size_per_image": 512,
    "box_positive_fraction": 0.25,
    "box_score_thresh": 0.05,
    "box_nms_thresh": 0.5,
    "box_detections_per_img": 100
}


class BoxCoder(object):
    def __init__(self, weights=None):
        super(BoxCoder, self).__init__()
        if weights is None:
            weights = [0.1, 0.1, 0.2, 0.2]
        self.weights = torch.tensor(data=weights, requires_grad=False)

    def encoder(self, anchors, gt_boxes):
        """
        :param gt_boxes:[box_num, 4]
        :param anchors: [box_num, 4]
        :return:
        """
        if self.weights.device != anchors.device:
            self.weights = self.weights.to(anchors.device)
        anchors_wh = anchors[..., [2, 3]] - anchors[..., [0, 1]]
        anchors_xy = anchors[..., [0, 1]] + 0.5 * anchors_wh
        gt_wh = (gt_boxes[..., [2, 3]] - gt_boxes[..., [0, 1]]).clamp(min=1.0)
        gt_xy = gt_boxes[..., [0, 1]] + 0.5 * gt_wh
        delta_xy = (gt_xy - anchors_xy) / anchors_wh
        delta_wh = (gt_wh / anchors_wh).log()

        delta_targets = torch.cat([delta_xy, delta_wh], dim=-1) / self.weights

        return delta_targets

    def decoder(self, predicts, anchors):
        """
        :param predicts: [anchor_num, 4] or [bs, anchor_num, 4]
        :param anchors: [anchor_num, 4]
        :return: [anchor_num, 4] (x1,y1,x2,y2)
        """
        if self.weights.device != anchors.device:
            self.weights = self.weights.to(anchors.device)
        anchors_wh = anchors[:, [2, 3]] - anchors[:, [0, 1]]
        anchors_xy = anchors[:, [0, 1]] + 0.5 * anchors_wh
        scale_reg = predicts * self.weights
        scale_reg[..., :2] = anchors_xy + scale_reg[..., :2] * anchors_wh
        scale_reg[..., 2:] = scale_reg[..., 2:].exp() * anchors_wh
        scale_reg[..., :2] -= (0.5 * scale_reg[..., 2:])
        scale_reg[..., 2:] = scale_reg[..., :2] + scale_reg[..., 2:]

        return scale_reg


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        'BELOW_LOW_THRESHOLD': int,
        'BETWEEN_THRESHOLDS': int,
    }

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
                matched_vals < self.high_threshold
        )
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    @staticmethod
    def set_low_quality_matches_(matches, all_matches, match_quality_matrix):
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        gt_pred_pairs_of_highest_quality = torch.nonzero(
            match_quality_matrix == highest_quality_foreach_gt[:, None], as_tuple=False
        )
        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


class FPN(nn.Module):
    def __init__(self, in_channels, out_channel, bias=True):
        super(FPN, self).__init__()
        self.latent_layers = list()
        self.out_layers = list()
        for channels in in_channels:
            self.latent_layers.append(nn.Conv2d(channels, out_channel, 1, 1, bias=bias))
            self.out_layers.append(nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=bias))
        self.latent_layers = nn.ModuleList(self.latent_layers)
        self.out_layers = nn.ModuleList(self.out_layers)
        self.max_pooling = nn.MaxPool2d(1, 2)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, xs):
        num_layers = len(xs)
        for i in range(num_layers):
            xs[i] = self.latent_layers[i](xs[i])
        for i in range(num_layers):
            layer_idx = num_layers - i - 1
            if i == 0:
                xs[layer_idx] = self.out_layers[layer_idx](xs[layer_idx])
            else:
                d_l = nn.UpsamplingBilinear2d(size=xs[layer_idx].shape[-2:])(xs[layer_idx + 1])
                xs[layer_idx] = self.out_layers[layer_idx](d_l + xs[layer_idx])
        xs.append(self.max_pooling(xs[-1]))
        return xs


class AnchorGenerator(object):
    def __init__(self, anchor_sizes, anchor_scales, anchor_ratios, strides):
        self.anchor_sizes = anchor_sizes
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.strides = strides
        self.anchor_per_grid = len(self.anchor_ratios) * len(self.anchor_scales)
        assert len(anchor_sizes) == len(strides)

    @staticmethod
    def __get_anchor_delta(anchor_size, anchor_scales, anchor_ratios):
        """
        :param anchor_size:
        :param anchor_scales: list
        :param anchor_ratios: list
        :return: [len(anchor_scales) * len(anchor_ratio),4]
        """
        scales = torch.tensor(anchor_scales).float()
        ratio = torch.tensor(anchor_ratios).float()
        scale_size = (scales * anchor_size)
        w = (scale_size[:, None] * ratio[None, :].sqrt()).view(-1) / 2
        h = (scale_size[:, None] / ratio[None, :].sqrt()).view(-1) / 2
        delta = torch.stack([-w, -h, w, h], dim=1)
        return delta

    def build_anchors(self, feature_maps):
        """
        :param feature_maps:
        :return: list(anchor) anchor:[all,4] (x1,y1,x2,y2)
        """
        assert len(self.anchor_sizes) == len(feature_maps)
        anchors = list()
        for stride, size, feature_map in zip(self.strides, self.anchor_sizes, feature_maps):
            # 9*4
            anchor_delta = self.__get_anchor_delta(size, self.anchor_scales, self.anchor_ratios)
            _, _, ny, nx = feature_map.shape
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            # h,w,4
            grid = torch.stack([xv, yv, xv, yv], 2).float()
            anchor = (grid[:, :, None, :] + 0.5) * stride + anchor_delta[None, None, :, :]
            anchor = anchor.view(-1, 4)
            anchors.append(anchor.to(feature_map.device))
        return anchors

    def __call__(self, feature_maps):
        anchors = self.build_anchors(feature_maps)
        return anchors


class RPNHead(nn.Module):
    def __init__(self, in_channel, anchor_per_grid):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.cls = nn.Conv2d(in_channel, anchor_per_grid, 1, 1)
        self.box = nn.Conv2d(in_channel, anchor_per_grid * 4, 1, 1)
        for l in self.children():
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        cls = []
        box = []
        bs = x[0].shape[0]
        for feature in x:
            t = F.relu(self.conv(feature))
            cls.append(self.cls(t).permute(0, 2, 3, 1).contiguous().view(bs, -1, 1))
            box.append(self.box(t).permute(0, 2, 3, 1).contiguous().view(bs, -1, 4))
        return cls, box


class RPN(nn.Module):
    def __init__(self, rpn_head,
                 rpn_pre_nms_top_n,
                 rpn_post_nms_top_n,
                 rpn_fg_iou_thresh=0.7,
                 rpn_bg_iou_thresh=0.3,
                 rpn_nms_thresh=0.7,
                 rpn_batch_size_per_image=256,
                 rpn_positive_fraction=0.5,
                 iou_type="giou"
                 ):
        super(RPN, self).__init__()

        self.rpn_head = rpn_head
        self.rpn_pre_nms_top_n = rpn_pre_nms_top_n
        self.rpn_post_nms_top_n = rpn_post_nms_top_n
        self.rpn_nms_thresh = rpn_nms_thresh
        self.rpn_batch_size_per_image = rpn_batch_size_per_image
        self.rpn_positive_fraction = rpn_positive_fraction
        self.proposal_matcher = Matcher(
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            allow_low_quality_matches=True
        )
        self.box_coder = BoxCoder([1., 1., 1., 1.])

        self.bce = nn.BCEWithLogitsLoss()
        self.box_loss = IOULoss(iou_type=iou_type)

    def get_pre_nms_top_n(self):
        return self.rpn_pre_nms_top_n['train'] if self.training else self.rpn_pre_nms_top_n['test']

    def get_post_nms_top_n(self):
        return self.rpn_post_nms_top_n['train'] if self.training else self.rpn_post_nms_top_n['test']

    def filter_proposals(self, proposals, objectness, anchor_nums_per_level, valid_size):
        """
        :param proposals:[bs,anchor_nums,4]
        :param objectness:[bs,anchor_nums,1]
        :param anchor_nums_per_level:list()
        :param valid_size:[bs,2](w,h)
        :return:
        """
        bs = proposals.shape[0]
        device = proposals.device
        objectness = objectness.squeeze(-1)
        levels = torch.cat([torch.full((n,), idx, dtype=torch.int64, device=device)
                            for idx, n in enumerate(anchor_nums_per_level)], dim=0)[None, :].repeat(bs, 1)
        anchor_idx = list()
        offset = 0
        for ob in objectness.split(anchor_nums_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.get_pre_nms_top_n(), num_anchors)
            _, top_k = ob.topk(pre_nms_top_n, dim=1)
            anchor_idx.append(top_k + offset)
            offset += num_anchors
        anchor_idx = torch.cat(anchor_idx, dim=1)
        batch_idx = torch.arange(bs, device=device)[:, None]
        objectness = objectness[batch_idx, anchor_idx]
        levels = levels[batch_idx, anchor_idx]
        proposals = proposals[batch_idx, anchor_idx]

        final_boxes = list()
        final_scores = list()
        for boxes, scores, lvl, img_shape in zip(proposals, objectness, levels, valid_size):
            width, height = img_shape
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(min=0, max=width)
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(min=0, max=height)
            keep = ((boxes[..., 2] - boxes[..., 0]) > 1e-3) & ((boxes[..., 3] - boxes[..., 1]) > 1e-3)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            keep = batched_nms(boxes, scores, lvl, self.rpn_nms_thresh)
            keep = keep[:self.get_post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def balanced_positive_negative_sampler(self, match_idx):
        sample_size = self.rpn_batch_size_per_image
        positive_fraction = self.rpn_positive_fraction
        positive = torch.nonzero(match_idx >= 0, as_tuple=False).squeeze(1)
        negative = torch.nonzero(match_idx == Matcher.BELOW_LOW_THRESHOLD, as_tuple=False).squeeze(1)
        num_pos = int(sample_size * positive_fraction)
        num_pos = min(positive.numel(), num_pos)
        num_neg = sample_size - num_pos
        num_neg = min(negative.numel(), num_neg)

        perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

        pos_idx_per_image = positive[perm1]
        neg_idx_per_image = negative[perm2]
        return pos_idx_per_image, neg_idx_per_image

    def compute_loss(self, objectness, proposal, anchors_all, targets):
        gt_boxes = targets['target'].split(targets['batch_len'])
        batch_idx = list()
        anchor_idx = list()
        gt_idx = list()
        for idx, gt in enumerate(gt_boxes):
            if len(gt) == 0:
                match_idx = torch.full_like(anchors_all[:, 0], fill_value=Matcher.BELOW_LOW_THRESHOLD).long()
            else:
                gt_anchor_iou = box_iou(gt[:, 1:], anchors_all)
                match_idx = self.proposal_matcher(gt_anchor_iou)
            positive_idx, negative_idx = self.balanced_positive_negative_sampler(match_idx)
            batch_idx.append(([idx] * len(positive_idx), [idx] * len(negative_idx)))
            gt_idx.append(match_idx[positive_idx].long())
            anchor_idx.append((positive_idx, negative_idx))
        all_batch_idx = sum([sum(item, []) for item in batch_idx], [])
        all_anchor_idx = torch.cat([torch.cat(item) for item in anchor_idx])
        all_cls_target = torch.tensor(sum([[1] * len(item[0]) + [0] * len(item[1])
                                           for item in anchor_idx], []),
                                      device=objectness.device, dtype=objectness.dtype)
        all_cls_predicts = objectness[all_batch_idx, all_anchor_idx]

        cls_loss = self.bce(all_cls_predicts, all_cls_target[:, None])
        all_positive_batch = sum([item[0] for item in batch_idx], [])
        all_positive_anchor = torch.cat([item[0] for item in anchor_idx])
        all_predict_box = proposal[all_positive_batch, all_positive_anchor]
        all_gt_box = torch.cat([i[j][:, 1:] for i, j in zip(gt_boxes, gt_idx)], dim=0)
        box_loss = self.box_loss(all_predict_box, all_gt_box).sum() / len(all_gt_box)
        return cls_loss, box_loss

    def forward(self, xs, anchors, valid_size, targets=None):
        objectness, pred_bbox_delta = self.rpn_head(xs)
        anchors_num_per_layer = [len(anchor) for anchor in anchors]
        anchors_all = torch.cat([anchor for anchor in anchors], dim=0)
        objectness = torch.cat([obj for obj in objectness], dim=1)
        pred_bbox_delta = torch.cat([delta for delta in pred_bbox_delta], dim=1)
        proposals = self.box_coder.decoder(pred_bbox_delta, anchors_all)
        boxes, scores = self.filter_proposals(proposals.detach(),
                                              objectness.detach(),
                                              anchors_num_per_layer,
                                              valid_size)
        losses = dict()
        if self.training:
            assert targets is not None
            cls_loss, box_loss = self.compute_loss(objectness, proposals, anchors_all, targets)
            losses['rpn_cls_loss'] = cls_loss
            losses['rpn_box_loss'] = box_loss
        return boxes, losses


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FasterRCNNSimplePredictor(nn.Module):
    def __init__(self, in_channels, num_cls=80):
        super(FasterRCNNSimplePredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_cls + 1)
        self.bbox_pred = nn.Linear(in_channels, 4)

    def forward(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


class ROIHead(nn.Module):
    def __init__(self,
                 box_head,
                 roi_pooling,
                 box_predict,
                 box_fg_iou_thresh=0.5,
                 box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512,
                 box_positive_fraction=0.25,
                 box_detections_per_img=100,
                 box_score_thresh=0.05,
                 box_nms_thresh=0.5,
                 iou_type="giou"):
        super(ROIHead, self).__init__()
        self.box_head = box_head
        self.roi_pooling = roi_pooling
        self.box_predict = box_predict
        self.box_fg_iou_thresh = box_fg_iou_thresh
        self.box_bg_iou_thresh = box_bg_iou_thresh
        self.box_batch_size_per_image = box_batch_size_per_image
        self.box_positive_fraction = box_positive_fraction
        self.box_detections_per_img = box_detections_per_img
        self.box_score_thresh = box_score_thresh
        self.box_nms_thresh = box_nms_thresh

        self.proposal_matcher = Matcher(
            self.box_fg_iou_thresh,
            self.box_bg_iou_thresh,
            allow_low_quality_matches=False
        )
        self.box_coder = BoxCoder()
        self.box_loss = IOULoss(iou_type=iou_type)
        self.ce = nn.CrossEntropyLoss()

    def balanced_positive_negative_sampler(self, match_idx):
        sample_size = self.box_batch_size_per_image
        positive_fraction = self.box_positive_fraction
        positive = torch.nonzero(match_idx >= 0, as_tuple=False).squeeze(1)
        negative = torch.nonzero(match_idx == Matcher.BELOW_LOW_THRESHOLD, as_tuple=False).squeeze(1)
        num_pos = int(sample_size * positive_fraction)
        num_pos = min(positive.numel(), num_pos)
        num_neg = sample_size - num_pos
        num_neg = min(negative.numel(), num_neg)

        perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

        pos_idx_per_image = positive[perm1]
        neg_idx_per_image = negative[perm2]
        return pos_idx_per_image, neg_idx_per_image

    def select_training_samples(self, proposals, gt_boxes):
        proposals = [torch.cat([p, g[:, 1:]]) for p, g in zip(proposals, gt_boxes)]
        proposals_ret = list()
        proposal_idx = list()
        for idx, p, g in zip(range(len(proposals)), proposals, gt_boxes):
            if len(g) == 0:
                match_idx = torch.full_like(p[:, 0], fill_value=Matcher.BELOW_LOW_THRESHOLD).long()
            else:
                gt_anchor_iou = box_iou(g[:, 1:], p)
                match_idx = self.proposal_matcher(gt_anchor_iou)
            positive_idx, negative_idx = self.balanced_positive_negative_sampler(match_idx)
            proposal_idx.append((positive_idx, negative_idx, match_idx[positive_idx].long()))
            proposals_ret.append(p[torch.cat([positive_idx, negative_idx])])
        return proposals_ret, proposal_idx

    def compute_loss(self, cls_predicts, box_predicts, proposal_idx, gt_boxes):
        assert proposal_idx is not None and gt_boxes is not None
        all_cls_idx = list()
        positive_mask = list()
        target_boxes = list()
        for prop_idx, gt_box in zip(proposal_idx, gt_boxes):
            p, n, g = prop_idx
            p_cls = gt_box[g][:, 0]
            n_cls = torch.full((len(n),), -1., device=p_cls.device, dtype=p_cls.dtype)
            all_cls_idx.append(p_cls)
            all_cls_idx.append(n_cls)
            mask = torch.zeros((len(p) + len(n),), device=p_cls.device).bool()
            mask[:len(p)] = True
            positive_mask.append(mask)
            target_boxes.append(gt_box[g][:, 1:])

        all_cls_idx = (torch.cat(all_cls_idx) + 1).long()
        positive_mask = torch.cat(positive_mask)
        target_boxes = torch.cat(target_boxes)
        box_loss = self.box_loss(box_predicts[positive_mask], target_boxes).sum() / len(target_boxes)
        cls_loss = self.ce(cls_predicts, all_cls_idx)
        return cls_loss, box_loss

    def post_process(self, cls_predicts, box_predicts, valid_size):
        predicts = list()
        for cls, box, wh in zip(cls_predicts, box_predicts, valid_size):
            box[..., [0, 2]] = box[..., [0, 2]].clamp(min=0, max=wh[0])
            box[..., [1, 3]] = box[..., [1, 3]].clamp(min=0, max=wh[1])
            scores = cls.softmax(dim=-1)
            scores = scores[:, 1:]
            labels = torch.arange(scores.shape[-1], device=cls.device)
            labels = labels.view(1, -1).expand_as(scores)
            boxes = box.unsqueeze(1).repeat(1, scores.shape[-1], 1).reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            inds = torch.nonzero(scores > self.box_score_thresh, as_tuple=False).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            keep = ((boxes[..., 2] - boxes[..., 0]) > 1e-2) & ((boxes[..., 3] - boxes[..., 1]) > 1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            keep = batched_nms(boxes, scores, labels, self.box_nms_thresh)
            keep = keep[:self.box_detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            pred = torch.cat([boxes, scores[:, None], labels[:, None]], dim=-1)
            predicts.append(pred)
        return predicts

    def forward(self, feature, proposals, valid_size, targets=None):
        feature_dict = dict()
        proposal_idx = None
        gt_boxes = None
        for i in range(len(feature) - 1):
            feature_dict['{:d}'.format(i)] = feature[i]
        if self.training:
            assert targets is not None
            gt_boxes = targets['target'].split(targets['batch_len'])
            proposals, proposal_idx = self.select_training_samples(proposals, gt_boxes)
        hw_size = [(s[1], s[0]) for s in valid_size]
        box_features = self.roi_pooling(feature_dict, proposals, hw_size)
        box_features = self.box_head(box_features)
        cls_predict, box_predicts = self.box_predict(box_features)
        box_predicts = self.box_coder.decoder(box_predicts, torch.cat(proposals))
        losses = dict()
        predicts = None
        if self.training:
            assert proposal_idx is not None and gt_boxes is not None
            cls_loss, box_loss = self.compute_loss(cls_predict, box_predicts, proposal_idx, gt_boxes)
            losses['roi_cls_loss'] = cls_loss
            losses['roi_box_loss'] = box_loss
        else:
            if cls_predict.dtype == torch.float16:
                cls_predict = cls_predict.float()
            if box_predicts.dtype == torch.float16:
                box_predicts = box_predicts.float()

            batch_nums = [len(p) for p in proposals]
            cls_predict = cls_predict.split(batch_nums, dim=0)
            box_predicts = box_predicts.split(batch_nums, dim=0)
            predicts = self.post_process(cls_predict, box_predicts, valid_size)
        return predicts, losses


class FasterRCNN(nn.Module):
    def __init__(self, **kwargs):
        super(FasterRCNN, self).__init__()
        self.cfg = {**default_cfg, **kwargs}
        self.backbone = getattr(resnet, self.cfg['backbone'])(pretrained=self.cfg['pretrained'],
                                                              reduction=self.cfg['reduction'])
        self.fpn = FPN(in_channels=self.backbone.inner_channels,
                       out_channel=self.cfg['fpn_channel'],
                       bias=self.cfg['fpn_bias'])
        self.anchor_generator = AnchorGenerator(
            anchor_sizes=self.cfg['anchor_sizes'],
            anchor_scales=self.cfg['anchor_scales'],
            anchor_ratios=self.cfg['anchor_ratios'],
            strides=self.cfg['strides']
        )
        self.anchors = None

        rpn_head = RPNHead(self.cfg['fpn_channel'], anchor_per_grid=self.anchor_generator.anchor_per_grid)
        rpn_pre_nms_top_n = {"train": self.cfg['rpn_pre_nms_top_n_train'],
                             "test": self.cfg['rpn_pre_nms_top_n_test']}
        rpn_post_nms_top_n = {"train": self.cfg['rpn_post_nms_top_n_train'],
                              "test": self.cfg['rpn_post_nms_top_n_test']}
        self.rpn = RPN(rpn_head=rpn_head,
                       rpn_pre_nms_top_n=rpn_pre_nms_top_n,
                       rpn_post_nms_top_n=rpn_post_nms_top_n,
                       rpn_fg_iou_thresh=self.cfg['rpn_fg_iou_thresh'],
                       rpn_bg_iou_thresh=self.cfg['rpn_bg_iou_thresh'],
                       rpn_nms_thresh=self.cfg['rpn_nms_thresh'],
                       rpn_batch_size_per_image=self.cfg['rpn_batch_size_per_image'],
                       rpn_positive_fraction=self.cfg['rpn_positive_fraction'])
        roi = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )
        resolution = roi.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(
            self.cfg['fpn_channel'] * resolution ** 2,
            representation_size)
        box_predict = FasterRCNNSimplePredictor(
            representation_size, num_cls=self.cfg['num_cls']
        )
        self.roi_head = ROIHead(box_head=box_head,
                                roi_pooling=roi,
                                box_predict=box_predict,
                                box_fg_iou_thresh=self.cfg['box_fg_iou_thresh'],
                                box_bg_iou_thresh=self.cfg['box_bg_iou_thresh'],
                                box_batch_size_per_image=self.cfg['box_batch_size_per_image'],
                                box_positive_fraction=self.cfg['box_positive_fraction'],
                                box_detections_per_img=self.cfg['box_detections_per_img'],
                                box_score_thresh=self.cfg['box_score_thresh'],
                                box_nms_thresh=self.cfg['box_nms_thresh'],
                                )

    def forward(self, x, valid_size, targets=None):
        if self.anchors:
            anchor_num = sum([a.shape[0] for a in self.anchors]) / self.anchor_generator.anchor_per_grid
        else:
            anchor_num = -1
        xs = self.backbone(x)
        xs = self.fpn(xs)
        xs_resolution = sum([i.shape[-2] * i.shape[-1] for i in xs])
        if xs_resolution != anchor_num:
            self.anchors = self.anchor_generator(xs)
        boxes, rpn_losses = self.rpn(xs, self.anchors, valid_size, targets)
        predicts, roi_losses = self.roi_head(xs, boxes, valid_size, targets)
        losses = {}
        losses.update(rpn_losses)
        losses.update(roi_losses)
        if self.training:
            return losses
        else:
            return predicts


if __name__ == '__main__':
    inp = torch.rand(size=(2, 3, 640, 640))
    size = torch.tensor([[640, 640], [640, 640]], dtype=torch.float)
    net = FasterRCNN().eval()
    net(inp, valid_size=size)
    net(inp, valid_size=size)
