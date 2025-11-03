# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Model head modules
"""

import math
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.yolo.utils.tal import dist2bbox, make_anchors

from .block import DFL, Proto
from .conv import Conv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init_

__all__ = ['Detect', 'Segment', 'Pose', 'Classify', 'RTDETRDecoder','Detect_Efficient']


class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        #print("nc",nc)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = Conv(in_dim, in_dim, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge  # Extract edges
        edge = self.out_conv(edge)  # Enhance edges
        return x + edge  # Add back enhanced edges

class GSConvE(nn.Module):
    '''
    GSConv enhancement for representation learning: generate various receptive-fields and
    texture-features only in one Conv module
    https://github.com/AlanLi1997/slim-neck-by-gsconv
    '''
    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, d, act)
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_, c_, 3, 1, 1, bias=False),
            nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, bias=False),
            nn.GELU()
        )

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        y = torch.cat((x1, x2), dim=1)
        # shuffle
        y = y.reshape(y.shape[0], 2, y.shape[1] // 2, y.shape[2], y.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])


class SEBlock(nn.Module):
    """ Squeeze-and-Excitation Block for Channel Attention. """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        scale = x.mean((2, 3), keepdim=True)  # Global Average Pooling
        scale = torch.sigmoid(self.fc2(F.relu(self.fc1(scale))))
        return x * scale  # Apply attention


# 🔥 Feature Pyramid with Shared Convolution & Dilation
class FeaturePyramidSharedConv(nn.Module):
    def __init__(self, c1, c2, dilations=[1, 3, 5]):
        super().__init__()

        c_ = c1 // 2  # Reduce channels first
        self.cv1 = nn.Conv2d(c1, c_, kernel_size=1, stride=1)  # Reduce channels
        self.cv2 = nn.Conv2d(c_ * (1 + len(dilations)), c2, kernel_size=1, stride=1)  # Expand channels after dilation
        self.dilations = dilations

        # 🔥 Shared Convolution: Depthwise + Pointwise + SE Attention
        self.share_conv = nn.Sequential(
            nn.Conv2d(c_, c_, kernel_size=3, stride=1, padding=1, groups=c_, bias=False),  # Depthwise
            nn.Conv2d(c_, c_, kernel_size=1, bias=False),  # Pointwise
            SEBlock(c_)  # SE Attention
        )

    def forward(self, x):
        y = [self.cv1(x)]  # Reduce channels first

        # 🔥 Multi-Dilation Feature Extraction (Uses same weights)
        for dilation in self.dilations:
            y.append(F.conv2d(y[-1], weight=self.share_conv[0].weight, bias=None, dilation=dilation,
                              padding=(dilation * (3 - 1) + 1) // 2, groups=y[-1].shape[1]))

        return self.cv2(torch.cat(y, 1))  # Concatenate all feature maps & pass through 1x1 conv
 
class Detect_Efficient(nn.Module):
    """YOLOv8 Efficient Detection Head with Feature Pyramid Shared Convolution."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc: int = 80, ch: tuple = (), reg_max: int = 16, dilations: list = [1, 3, 5]):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)  # Number of detection layers
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.dilations = dilations
        self.stride = torch.zeros(self.nl)  # strides computed during build

        self.anchors = torch.empty(0)
        self.shape = None
        self.dynamic = False
        self.export = False

        # 🔥 Shared Convolution with Multi-Dilation Feature Extraction
        self.feature_pyramid = nn.ModuleList([
            FeaturePyramidSharedConv(x, x, dilations=self.dilations) for x in ch
        ])

        # 🔥 Detection Heads
        self.regression_head = nn.ModuleList(nn.Conv2d(x, 4 * self.reg_max, 1) for x in ch)
        self.classification_head = nn.ModuleList(nn.Conv2d(x, self.nc, 1) for x in ch)

        # 🔥 Bounding Box Transformation
        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

    def forward(self, x):
        batch_size, _, _, _ = x[0].shape  # BCHW
        shape = x[0].shape  # BCHW

        for i in range(self.nl):
            # 🔥 Apply Feature Pyramid Shared Conv
            x[i] = self.feature_pyramid[i](x[i])

            x[i] = torch.cat((self.regression_head[i](x[i]), self.classification_head[i](x[i])), dim=1)

        if self.training:
            return x  # Return raw outputs for loss computation


        # 🔥 Update Anchors if Needed
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = x[0].shape

        # 🔥 Concatenate outputs across detection layers
        x_cat = torch.cat([xi.view(batch_size, self.no, -1) for xi in x], dim=2)

        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), dim=1)

        # 🔥 Decode Box Predictions
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        predictions = torch.cat((dbox, cls.sigmoid()), dim=1)

        return predictions if self.export else (predictions, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.regression_head, m.classification_head, m.stride):  # from
            a.bias.data[:] = 1.0  # box
            b.bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class Segment(Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, ch)
        ###### Jiayuan changed self.nm to self.nc
        self.npr = 32  # intermediate convolutional feature dimension
        self.cv1 = Conv(ch[0], self.npr, k=3)
        self.edge_enhancers = EdgeEnhancer(ch[0])
        self.upsample = nn.ConvTranspose2d(self.npr, self.npr//2, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(self.npr//2, self.npr//4, k=3)
        self.cv3 = Conv(self.npr//4, self.nc+1) ###### self.nc+1 means add the background
        self.sigmoid = nn.Sigmoid()
        # self.detect = Detect.forward
        #
        # c4 = max(ch[0] // 4, self.nm)
        # self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.cv3(self.cv2(self.upsample(self.cv1(self.edge_enhancers(x[0]))))) # mask protos
        if self.training:
            return p
        return p
        # bs = p.shape[0]  # batch size
        #
        # mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        # x = self.detect(self, x)
        # if self.training:
        #     return x, mc, p
        # return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))

"""class Segment(Detect):

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, ch)
        self.npr = 32  # intermediate convolutional feature dimension
        self.edge_enhancer = EdgeEnhancer(ch[0])  # Edge enhancement module
        self.cv1 = Conv(ch[0], self.npr, k=3)
        self.upsample = nn.ConvTranspose2d(self.npr, self.npr//2, 2, 2, 0, bias=True)
        self.cv2 = Conv(self.npr//2, self.npr//4, k=3)
        self.cv3 = Conv(self.npr//4, self.nc+1)  # self.nc+1 means add the background
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x[0] = self.edge_enhancer(x[0])  # Apply edge enhancement
        p = self.cv3(self.cv2(self.upsample(self.cv1(x[0]))))  # Mask protos
        if self.training:
            return p
        return p"""
class Pose(Detect):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 1:2].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3].sigmoid_()  # inplace sigmoid
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class RTDETRDecoder(nn.Module):

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hidden_dim=256,
            num_queries=300,
            strides=(8, 16, 32),  # TODO
            nl=3,
            num_decoder_points=4,
            nhead=8,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.,
            act=nn.ReLU(),
            eval_idx=-1,
            # training args
            num_denoising=100,
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False):
        super().__init__()
        assert len(ch) <= nl
        assert len(strides) == len(ch)
        for _ in range(nl - len(strides)):
            strides.append(strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = strides
        self.nl = nl
        self.nc = nc
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers

        # backbone feature projection
        self._build_input_proj_layer(ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, act, nl,
                                                          num_decoder_points)
        self.decoder = DeformableTransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(nc, hidden_dim)
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.enc_score_head = nn.Linear(hidden_dim, nc)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hidden_dim, nc) for _ in range(num_decoder_layers)])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(num_decoder_layers)])

        self._reset_parameters()

    def forward(self, feats, gt_meta=None):
        # input projection and embedding
        memory, spatial_shapes, _ = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training:
            raise NotImplementedError
            # denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
            #     get_contrastive_denoising_training_group(gt_meta,
            #                                 self.num_classes,
            #                                 self.num_queries,
            #                                 self.denoising_class_embed.weight,
            #                                 self.num_denoising,
            #                                 self.label_noise_ratio,
            #                                 self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask = None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(target,
                                              init_ref_points_unact,
                                              memory,
                                              spatial_shapes,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask)
        if not self.training:
            out_logits = out_logits.sigmoid_()
        return out_bboxes, out_logits  # enc_topk_bboxes, enc_topk_logits, dn_meta

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)

    def _build_input_proj_layer(self, ch):
        self.input_proj = nn.ModuleList()
        for in_channels in ch:
            self.input_proj.append(
                nn.Sequential(nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1, bias=False),
                              nn.BatchNorm2d(self.hidden_dim)))
        in_channels = ch[-1]
        for _ in range(self.nl - len(ch)):
            self.input_proj.append(
                nn.Sequential(nn.Conv2D(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
                              nn.BatchNorm2d(self.hidden_dim)))
            in_channels = self.hidden_dim

    def _generate_anchors(self, spatial_shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(end=h, dtype=torch.float32),
                                            torch.arange(end=w, dtype=torch.float32),
                                            indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)

            valid_WH = torch.tensor([h, w]).to(torch.float32)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = torch.concat(anchors, 1)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors.to(device=device, dtype=dtype), valid_mask.to(device=device)

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.nl > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.nl):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0]
        for feat in proj_feats:
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()
        return feat_flatten, spatial_shapes, level_start_index

    def _get_decoder_input(self, memory, spatial_shapes, denoising_class=None, denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        anchors, valid_mask = self._generate_anchors(spatial_shapes, dtype=memory.dtype, device=memory.device)
        memory = torch.where(valid_mask, memory, 0)
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)  # (bs, h*w, nc)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors  # (bs, h*w, 4)

        # (bs, topk)
        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)
        # extract region proposal boxes
        # (bs, topk_ind)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)
        topk_ind = topk_ind.view(-1)

        # Unsigmoided
        reference_points_unact = enc_outputs_coord_unact[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        enc_topk_bboxes = torch.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat([denoising_bbox_unact, reference_points_unact], 1)
        if self.training:
            reference_points_unact = reference_points_unact.detach()
        enc_topk_logits = enc_outputs_class[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        else:
            target = output_memory[batch_ind, topk_ind].view(bs, self.num_queries, -1)
            if self.training:
                target = target.detach()
        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits
