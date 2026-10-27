import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Int8Bias, Uint8ActPerTensorFloat
import utils_cifar10 as utils
import warnings

warnings.filterwarnings("ignore", message=".*Named tensors.*")
warnings.filterwarnings("ignore", message=".*__torch_function__.*")

class nnet_cifar10(nn.Module):
    def __init__(
        self, num_classes=10, cfg=None, cls_cfg=None, input_size=32,
        act_bit_width=4, w_bit_width=4, in_bit_width=8, kernel_size=3,
        dropout_p=0.3, narrow_range=True, unsigned_requant=False
    ):
        super().__init__()
        self.unsigned_requant = bool(unsigned_requant)
        self.act_bit_width = int(act_bit_width)
        self.w_bit_width   = int(w_bit_width)
        self.kernel_size   = int(kernel_size)
        self.narrow_range  = narrow_range
        self.cfg = cfg if cfg is not None else utils.ARC_CONFIGS["VGG9-C10-ZAMA"]["cfg"]
        self.cls_cfg = cls_cfg if cls_cfg is not None else utils.ARC_CONFIGS["VGG9-C10-ZAMA"]["cls_cfg"]

        self.quant_input = qnn.QuantIdentity(bit_width=in_bit_width, return_quant_tensor=True)
        self.features, last_ch, final_spatial = self._make_layers(self.cfg, input_size, act_bit_width)
        
        self.flatten = nn.Flatten(start_dim=1)
        self.requant_flat = self._requant(act_bit_width)
        self.flatten_dim = int(last_ch * final_spatial * final_spatial)
        self.classifier = self._make_classifier(self.flatten_dim, self.cls_cfg, num_classes, act_bit_width, dropout_p)

    def _make_classifier(self, in_features, cls_cfg, num_classes, act_bit_width, dropout_p):
        layers = []
        curr_in = int(in_features)
        drop = nn.Identity() if dropout_p <= 0 else nn.Dropout(dropout_p)

        for h_dim in cls_cfg:
            layers += [
                qnn.QuantLinear(curr_in, h_dim, bias=True, weight_bit_width=self.w_bit_width,
                                weight_quant=Int8WeightPerTensorFloat, bias_quant=Int8Bias,
                                weight_narrow_range=self.narrow_range, return_quant_tensor=True),
                # Adjusted eps for 2-bit quantization stability
                nn.BatchNorm1d(h_dim, eps=1e-4), 
                qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True),
                drop,
            ]
            curr_in = h_dim

        layers += [
            qnn.QuantLinear(curr_in, num_classes, bias=True, weight_bit_width=self.w_bit_width,
                            weight_quant=Int8WeightPerTensorFloat, bias_quant=Int8Bias,
                            weight_narrow_range=self.narrow_range, return_quant_tensor=False)
        ]
        return nn.Sequential(*layers)

    def _make_layers(self, cfg, current_size, act_bit_width):
        layers = []
        in_channels = 3

        for x in cfg:
            if x == "M" or x == "A":
                Pool = nn.MaxPool2d if x == "M" else nn.AvgPool2d
                layers += [
                    Pool(kernel_size=2, stride=2),
                    self._requant(act_bit_width),
                ]
                current_size //= 2
            else:
                x = int(x)
                layers += [
                    # Strict padding=0 for proper spatial reduction in FHE
                    qnn.QuantConv2d(in_channels, x, kernel_size=self.kernel_size, padding=0, bias=False,
                                    weight_bit_width=self.w_bit_width, weight_quant=Int8WeightPerTensorFloat,
                                    weight_narrow_range=self.narrow_range, return_quant_tensor=True),
                    # Adjusted eps for 2-bit quantization stability
                    nn.BatchNorm2d(x, eps=1e-4), 
                    qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True),
                ]
                in_channels = x
                # Calculate new spatial dimension mathematically
                current_size = current_size - self.kernel_size + 1 
                
        return nn.Sequential(*layers), in_channels, current_size

    def _requant(self, bits):
        if self.unsigned_requant:
            return qnn.QuantIdentity(act_quant=Uint8ActPerTensorFloat, bit_width=bits, return_quant_tensor=True)
        return qnn.QuantIdentity(bit_width=bits, return_quant_tensor=True)

    def clip_weights(self):
        max_w = 2 ** (self.w_bit_width - 1) - 1
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, (qnn.QuantConv2d, qnn.QuantLinear)):
                    m.weight.data.clamp_(-max_w, max_w)

    def forward(self, x):
        x = self.quant_input(x)
        x = self.features(x)
        x = x.value if hasattr(x, "value") else x
        x = self.flatten(x)
        x = self.requant_flat(x)
        return self.classifier(x)