import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn import Parameter

from .quant_utils import *

def round_pass(input: torch.tensor):
    """
    Args:
        input: input tensor

    Returns:
        rounded tensor with STE for backward
    """
    y = torch.round(input)
    y_grad = input
    return (y - y_grad).detach() + y_grad

def floor_pass(input: torch.tensor):
    """
    Args:
        input: input tensor

    Returns:
        rounded tensor with STE for backward
    """
    y = torch.floor(input)
    y_grad = input
    return (y - y_grad).detach() + y_grad

class Quantizer():
    def __init__(self, N_bits: int, dtype: torch.dtype , signed: bool = True, symmetric: bool = True):
        super().__init__()
        if N_bits is None:
            self.N_bits = None
            return
        self.N_bits = N_bits
        self.signed = signed
        self.symmetric = symmetric
        # self.eps = torch.iinfo(dtype).eps
        # self.minimum_range = torch.iinfo(dtype).eps

        if self.signed:
            self.Qn = - 2 ** (self.N_bits - 1)
            self.Qp = 2 ** (self.N_bits - 1) - 1
        else:
            self.Qn = 0
            self.Qp = 2 ** self.N_bits - 1

    def __call__(self, x):  
        return self.forward(x)

    def forward(self, x): 
        if self.N_bits is None:
            return None

        if self.symmetric:
            max_x = x.abs().max().detach()
            scale = max_x / self.Qp
            x = x * scale 
            x = round_pass(x.clamp_(self.Qn, self.Qp)) 
            
        else: #Asymmetric
            min_x = x.min().detach()
            max_x = x.max().detach()
            range_x = (max_x - min_x).detach().clamp_(min=self.minimum_range)
            scale = range_x / (self.Qp - self.Qn)

            zero_point = torch.round((min_x / scale) - self.Qn)

            x = (x / scale) + zero_point
            x = round_pass(x.clamp_(self.Qn, self.Qp))

        return x, scale


class Quantized_Linear(nn.Linear):
    def __init__(self, weight_quantize_module: Quantizer, act_quantize_module: Quantizer, grad_quantize_module: Quantizer,
                 in_features, out_features, bias=True):
        super(Quantized_Linear, self).__init__(in_features, out_features, bias=bias)
        self.weight_quantize_module = weight_quantize_module
        self.act_quantize_module = act_quantize_module
        self.grad_quantize_module = grad_quantize_module

    def forward(self, input, s_x):
        return _quantize_global.apply(input, s_x, self.weight, self.bias, self.weight_quantize_module,
                                      self.act_quantize_module, self.grad_quantize_module)
    

class _quantize_global(torch.autograd.Function):
    def forward(ctx, x_3D, s_x, w_2D, bias=None, w_qmodule=None, a_qmodule=None, g_qmodule=None):
        x_2D = x_3D.view(-1, x_3D.size(-1)) #reshape to 2D
        s_x_expanded = s_x.view(1, -1).expand_as(x_2D) if s_x.dim() == 1 else s_x
        x_2D = x_2D * s_x_expanded #dequantize 

        weight_quant, s_weight_quant = w_qmodule(w_2D)
        input_quant, s_input_quant = a_qmodule(x_2D)

        ctx.save_for_backward = x_2D, w_2D, s_input_quant, s_weight_quant

        ctx.g_qmodule = g_qmodule
        
        output = input_quant.matmul(weight_quant.t())

        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        s_o = s_weight_quant * s_input_quant 

        # return F.linear(input_quant, weight=weight_quant, bias=self.bias) \
        #        * bias_scaling_factor, bias_scaling_factor

        return output.view(*x_3D.size()[:-1], -1) * s_o
    
    @staticmethod
    def backward(ctx, g_3D):
        g_2D = g_3D.reshape(-1, g_3D.size(-1))
        grad_X = grad_W = grad_bias = None 
        x, w, s_x, s_w = ctx.save_for_backward

        if ctx.g_qmodule is not None:
            g_2D_quant, s_g_2D_quant = ctx.g_qmodule(g_2D)


        if ctx.g_qmodule is not None: #Forward & Backward Quantizaiton
            grad_X = torch.matmul(g_2D_quant, w.to(g_2D_quant.dtype)).view(*g_2D_quant.size()[:-1], -1)
            grad_X = grad_X * s_g_2D_quant * s_w

            grad_W = torch.matmul(g_2D_quant.t(), x.to(g_2D_quant.dtype))
            grad_W = grad_W * s_g_2D_quant * s_x

            grad_bias = g_2D.sum(dim=0)

        else: #Only Forward Quantization
            grad_X = torch.matmul(g_2D, w.to(g_2D.dtype)).view(*g_2D.size()[:-1], -1)
            grad_X = grad_X * s_w

            grad_W = torch.matmul(g_2D.t(), x.to(g_2D.dtype))
            grad_W = grad_W * s_x

            grad_bias = g_2D.sum(dim=0)

        return grad_X, None, grad_W, grad_bias, None, None, None
        


class QuantAct(nn.Module):
    def __init__(self, 
                 N_bits: int, 
                 dtype: torch.dtype , 
                 signed: bool = True, 
                 symmetric: bool = True):
        super(QuantAct, self).__init__()
        self.quantizer = Quantizer(N_bits=N_bits, dtype=dtype, signed=signed, symmetric=symmetric)

    def forward(self, x):
        q_x, s_qx = self.quantizer(x)
        return q_x, s_qx
    

# class IntLayerNorm(nn.LayerNorm):
#     """
#     Implementation of I-LayerNorm
#     Class to quantize given LayerNorm layer
#     """
#     def __init__(self, 
#                 normalized_shape, 
#                 eps=1e-5,
#                 elementwise_affine=True):
#         super(IntLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)
#         self.dim_sqrt = None
#         self.register_buffer('norm_scaling_factor', torch.zeros(1))
#         self.register_buffer('bias_integer', torch.zeros_like(self.bias))

#     def forward(self, x): #input: full-precision 
#         if self.dim_sqrt is None:
#             n = torch.tensor(x.shape[2], dtype=torch.float)
#             self.dim_sqrt = torch.sqrt(n).cuda()

#         # Normalization: computes mean and variance(std)

#         # x_int = x / scaling_factor
#         meant = round_pass.apply(x.mean(axis=2, keepdim=True))
#         y = x - meant
#         y_sq = y ** 2
#         var = torch.sum(y_sq, axis=2, keepdim=True)
#         rstd = 1 / var.sqrt()

#         x_hat = (x - meant) * rstd

#         ln_output = x_hat * self.weight + self.bias
        

#         # # Integer Iteration
#         # k = 2 ** 16
#         # for _ in range(10):
#         #     k_1 = floor_pass.apply((k + floor_pass.apply(q_var/k))/2)
#         #     k = k_1
#         # std_int = k

#         # factor = floor_pass.apply((2 ** 31-1) / std_int)
#         # y_int = floor_pass.apply(y_int * factor / 2)
#         # scaling_factor = self.dim_sqrt / 2 ** 30

#         # # scaling and shifting
#         # bias = self.bias.data.detach() / (self.weight.data.detach())
#         # bias_int = floor_pass.apply(bias / scaling_factor)

#         # self.bias_integer = bias_int

#         # y_int = y_int + bias_int
#         # scaling_factor = scaling_factor * self.weight
#         # x = y_int * scaling_factor
#         # self.norm_scaling_factor = scaling_factor
#         return ln_output

#이건 가능한 안쓰도록 
class IntGELU(nn.Module):
    """
    Implementation of ShiftGELU
    Class to quantize given GELU layer
    """

    def __init__(self, output_bit=8):
        super(IntGELU, self).__init__()
        self.output_bit = output_bit

        self.n = 23  # sufficiently large integer
        #The minimum value for ensuring accuracy (varies depending on models)

        self.register_buffer('act_scaling_factor', torch.zeros(1))

    # def fix(self):
    #     pass

    # def unfix(self):
    #     pass

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int + floor_pass.apply(x_int / 2) - floor_pass.apply(x_int / 2 ** 4)

        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor)
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_pass.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r/2 - x0_int
        exp_int = torch.clamp(floor_pass.apply(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = scaling_factor / 2 ** self.n

        return exp_int, scaling_factor

    def forward(self, x, scaling_factor=None):
        pre_x_int = x / scaling_factor
        scaling_factor_sig = scaling_factor * 1.702

        x_int_max, _ = pre_x_int.max(dim=-1, keepdim=True)
        x_int = pre_x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor_sig) # e^(x-x_max)

        exp_int_max, _ = self.int_exp_shift(-x_int_max, scaling_factor_sig)  # e^(-x_max)
        exp_int_sum = exp_int + exp_int_max

        exp_int_sum.clamp_max_(2**31-1)
        factor = floor_pass.apply((2 ** 31-1) / exp_int_sum)
        sigmoid_int = floor_pass.apply(exp_int * factor / 2 ** (31-self.output_bit+1))
        sigmoid_scaling_factor = torch.Tensor([1 / 2 ** (self.output_bit-1)]).cuda()

        x_int = pre_x_int * sigmoid_int
        scaling_factor = scaling_factor * sigmoid_scaling_factor
        self.act_scaling_factor = scaling_factor
        return x_int * scaling_factor, scaling_factor


#이건 가능한 안쓰도록 
class IntSoftmax(nn.Module):
    """
    Implementation of Shiftmax
    Class to quantize given Softmax layer
    """

    def __init__(self, output_bit=8):
        super(IntSoftmax, self).__init__()
        self.output_bit = output_bit

        self.n = 15  # sufficiently large integer
        #The minimum value for ensuring accuracy (varies depending on models)

        self.register_buffer('act_scaling_factor', torch.zeros(1))

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int + floor_pass.apply(x_int / 2) - floor_pass.apply(x_int / 2 ** 4)

        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor)
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_pass.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r/2 - x0_int
        exp_int = torch.clamp(floor_pass.apply(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = scaling_factor / 2 ** self.n
        return exp_int, scaling_factor

    def forward(self, x, scaling_factor):
        x_int = x / scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)

        exp_int_sum.clamp_max_(2**31-1)
        factor = floor_pass.apply((2**31-1) / exp_int_sum)
        exp_int = floor_pass.apply(exp_int * factor / 2 ** (31-self.output_bit+1))
        scaling_factor = torch.Tensor([1 / 2 ** (self.output_bit-1)]).cuda()

        self.act_scaling_factor = scaling_factor
        return exp_int * scaling_factor, scaling_factor

#이건 가능한 안쓰도록 
class QuantMatMul(nn.Module):
    """
    Class to quantize weights of given matmul layer
    """
    def __init__(self):
        super(QuantMatMul, self).__init__()
        self.register_buffer('act_scaling_factor', torch.zeros(1))
    
    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, A, pre_act_scaling_factor_A, B, pre_act_scaling_factor_B):
        A_int = A / pre_act_scaling_factor_A
        B_int = B / pre_act_scaling_factor_B
        act_scaling_factor = pre_act_scaling_factor_A * pre_act_scaling_factor_B
        self.act_scaling_factor = act_scaling_factor
        return (A_int @ B_int) * act_scaling_factor, act_scaling_factor


# class QuantConv2d(nn.Conv2d):
#     """
#     Class to quantize weights of given convolutional layer
#     Parameters:
#     ----------
#     weight_bit : int, default 4
#         Bitwidth for quantized weights.
#     bias_bit : int, default None
#         Bitwidth for quantized bias.
#     full_precision_flag : bool, default False
#         If True, use fp32 and skip quantization
#     quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
#         The mode for quantization.
#     per_channel : bool, default False
#         Whether to use channel-wise quantization.
#     fix_flag : bool, default False
#         Whether the module is in fixed mode or not.
#     weight_percentile : float, default 0
#         The percentile to setup quantization range, 0 means no use of percentile, 99.9 means to cut off 0.1%.
#     """

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride=1,
#                  padding=0,
#                  dilation=1,
#                  groups=1,
#                  bias=True,
#                  weight_bit=8,
#                  bias_bit=32,
#                  quant_mode="symmetric",
#                  per_channel=True,
#                  weight_percentile=0):
#         super(QuantConv2d, self).__init__(in_channels=in_channels,
#                                           out_channels=out_channels,
#                                           kernel_size=kernel_size,
#                                           stride=stride,
#                                           padding=padding,
#                                           dilation=dilation,
#                                           groups=groups,
#                                           bias=bias
#                                           )
#         self.weight_bit = weight_bit
#         self.quant_mode = quant_mode
#         self.per_channel = per_channel
#         self.weight_percentile = weight_percentile
#         self.bias_bit = bias_bit
#         self.quantize_bias = (False if bias_bit is None else True)

#         self.register_buffer('conv_scaling_factor', torch.zeros(self.out_channels))
#         self.register_buffer('weight_integer', torch.zeros_like(self.weight))
#         self.register_buffer('bias_integer', torch.zeros_like(self.bias))

#     def __repr__(self):
#         s = super(QuantConv2d, self).__repr__()
#         s = "(" + s + " weight_bit={}, quant_mode={})".format(self.weight_bit, self.quant_mode)
#         return s

#     def fix(self):
#         pass

#     def unfix(self):
#         pass

#     def forward(self, x, pre_act_scaling_factor=None):
#         if self.quant_mode == "symmetric":
#             self.weight_function = SymmetricQuantFunction.apply
#         elif self.quant_mode == "asymmetric":
#             raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
#         else:
#             raise ValueError("unknown quant mode: {}".format(self.quant_mode))

#         with torch.no_grad():
#             w = self.weight
#             if self.per_channel:
#                 v = w.reshape(w.shape[0], -1)
#                 cur_min = v.min(axis=1).values
#                 cur_max = v.max(axis=1).values
#                 self.min_val = cur_min
#                 self.max_val = cur_max
#             else:
#                 raise Exception('For weight, we only support per_channel quantization.')

#             self.conv_scaling_factor = symmetric_linear_quantization_params(
#                 self.weight_bit, self.min_val, self.max_val)

#         self.weight_integer = self.weight_function(
#             self.weight, self.weight_bit, self.conv_scaling_factor, True)
#         bias_scaling_factor = self.conv_scaling_factor * pre_act_scaling_factor
#         self.bias_integer = self.weight_function(
#             self.bias, self.bias_bit, bias_scaling_factor, True)

#         pre_act_scaling_factor = pre_act_scaling_factor.view(1, -1, 1, 1)
#         x_int = x / pre_act_scaling_factor
#         correct_output_scale = bias_scaling_factor.view(1, -1, 1, 1)

#         return (F.conv2d(x_int, self.weight_integer, self.bias_integer, self.stride, self.padding,
#                          self.dilation, self.groups) * correct_output_scale, correct_output_scale)

