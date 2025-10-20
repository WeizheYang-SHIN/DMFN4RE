# coding: utf-8
# _*_ coding: utf-8 _*_
# @Time : 2024/5/21 14:46 
# @Author : wz.yang 
# @File : binirilize_lbcnn_model.py
# @desc :

import torch
import torch.nn as nn
import torch.nn.functional as F
from .data_utils import *
from transformers import BertTokenizer, BertModel

class ConvLBP(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, sparsity=0.5):
        super().__init__(in_channels, out_channels, kernel_size, padding=1, bias=False)
        self.sparsity = sparsity
        self.initialize_weights()

    def initialize_weights(self):
        weights = self.weight.data
        # matrix_proba = torch.FloatTensor(weights.shape).fill_(0.5)
        # binary_weights = torch.bernoulli(matrix_proba) * 2 - 1
        # mask_inactive = torch.rand(matrix_proba.shape) > self.sparsity
        # binary_weights.masked_fill_(mask_inactive, 0)
        # self.original_weight = nn.Parameter(binary_weights.clone())  # 保存原始二值权重作为参数
        self.weight.data.copy_(weights)
        self.weight.requires_grad_(True)
        # 添加一个历史趋势变量，用于记录权重变化趋势
        self.prev_mean = None
        self.prev_std = None
        # self.weight.data = self.quantize_weights(self.weight.data)

    # def update_weight(self):
    #     # 将权重更新为量化后的{-1, 0, 1}
    #     with torch.no_grad():
    #         # updated_weight = self.original_weight.data.clone()
    #         updated_weight = self.weight.data
    #         quantized_weight = torch.where(updated_weight > 0, torch.ones_like(updated_weight),
    #                                        torch.where(updated_weight < 0, -torch.ones_like(updated_weight),
    #                                                    torch.zeros_like(updated_weight)))
    #         self.weight.data.copy_(quantized_weight)
    #         self.original_weight = quantized_weight

    # def update_weight(self):
    #     # 将权重更新为量化后的{-1, 0, 1}，基于与初始化权重的差异
    #     with torch.no_grad():
    #         # 计算当前权重与原始权重的差异
    #         weight_diff = self.weight.data - self.original_weight.data
    #         # 根据差异的大小进行量化
    #         quantized_weight = torch.where((weight_diff >= 0.01), torch.ones_like(weight_diff),
    #                                        torch.where((weight_diff <= -0.01), -torch.ones_like(weight_diff),
    #                                                    torch.zeros_like(weight_diff)))
    #         # 更新self.weight，这里直接用量化后的差异加上原始权重进行更新
    #         # self.weight.data.copy_(self.original_weight.data + quantized_weight)
    #         self.weight.data.copy_(quantized_weight)
    #         self.original_weight.data.copy_(quantized_weight)

    # 自适应量化阈值
    # def update_weight(self):
    #     with torch.no_grad():
    #         weight_diff = self.weight.data - self.original_weight.data
    #         abs_diff_mean = torch.mean(torch.abs(weight_diff)).item()
    #         threshold = 0.01 * abs_diff_mean  # 动态阈值
    #         quantized_weight = torch.where((weight_diff >= threshold), torch.ones_like(weight_diff),
    #                                        torch.where((weight_diff <= -threshold), -torch.ones_like(weight_diff),
    #                                                    torch.zeros_like(weight_diff)))
    #         self.weight.data.copy_(quantized_weight)
    #         self.original_weight.data.copy_(quantized_weight)

    # 多比特量化
    # def update_weight(self):
    #     with torch.no_grad():
    #         weight_diff = self.weight.data - self.original_weight.data
    #         # 使用三个等级的量化 {-2, 0, 2}
    #         quantized_weight = torch.where((weight_diff >= 0.02), 2 * torch.ones_like(weight_diff),
    #                                        torch.where((weight_diff <= -0.02), -2 * torch.ones_like(weight_diff),
    #                                                    torch.zeros_like(weight_diff)))
    #         self.weight.data.copy_(quantized_weight)
    #         self.original_weight.data.copy_(quantized_weight)

    # 基于权重分布的标准差进行量化
    # def update_weight(self):
    #     with torch.no_grad():
    #         # 计算当前权重的均值和标准差
    #         weight_mean = torch.mean(self.weight.data)
    #         weight_std = torch.std(self.weight.data)
    #
    #         # 定义量化策略的阈值，这里以均值±标准差为界，根据实际情况调整
    #         lower_bound = weight_mean - weight_std
    #         upper_bound = weight_mean + weight_std
    #
    #         # 利用当前权重的分布动态量化
    #         quantized_weight = torch.where(self.weight.data > upper_bound, torch.ones_like(self.weight.data),
    #                                        torch.where(self.weight.data < lower_bound,
    #                                                    -torch.ones_like(self.weight.data),
    #                                                    torch.zeros_like(self.weight.data)))
    #
    #         self.weight.data.copy_(quantized_weight)
    #         self.original_weight.data.copy_(quantized_weight)

    # def update_weight(self):
    #     with torch.no_grad():
    #         # 计算当前权重的均值和标准差
    #         weight_mean = torch.mean(self.weight.data)
    #         weight_std = torch.std(self.weight.data)
    #
    #         # 初始化历史趋势（首次运行）
    #         if self.prev_mean is None or self.prev_std is None:
    #             self.prev_mean = weight_mean
    #             self.prev_std = weight_std
    #             # 首次运行时直接量化，后续迭代将基于变化调整
    #             lower_bound = weight_mean - weight_std
    #             upper_bound = weight_mean + weight_std
    #         else:
    #             # 动态调整量化界限，基于前后两次迭代的均值和标准差变化
    #             delta_mean = (weight_mean - self.prev_mean) / weight_mean  # 相对变化量
    #             delta_std = (weight_std - self.prev_std) / weight_std
    #
    #             # 调整量化界限，这里以变化率作为调整因子，具体调整逻辑可以根据需求定制
    #             lower_bound = weight_mean - (weight_std * (1 + delta_std))
    #             upper_bound = weight_mean + (weight_std * (1 + delta_std))
    #
    #             # 更新历史趋势
    #             self.prev_mean = weight_mean
    #             self.prev_std = weight_std
    #
    #         # 量化操作
    #         quantized_weight = torch.where(self.weight.data > upper_bound, torch.ones_like(self.weight.data),
    #                                        torch.where(self.weight.data < lower_bound,
    #                                                    -torch.ones_like(self.weight.data),
    #                                                    torch.zeros_like(self.weight.data)))
    #
    #         # 应用量化后的权重
    #         self.weight.data.copy_(quantized_weight)
    #         self.original_weight.data.copy_(quantized_weight)

    # def update_weight(self):
    #     with torch.no_grad():
    #         # 计算当前权重的梯度（这里简化处理，实际训练中应从优化器获取）
    #         # 注意：实际应用中，这一步应当在反向传播后，优化器.step()之前执行
    #         # gradient = torch.randn_like(self.weight)  # 示例，实际应使用真实的梯度
    #
    #         # 确保在此前已经进行了loss.backward()操作并计算了梯度
    #         if self.weight.grad is not None:  # 检查梯度是否存在，避免初次未计算的情况
    #             gradient = self.weight.grad.clone()  # 使用真实的梯度，克隆一份防止外部修改影响
    #             # 接下来的逻辑保持不变，使用gradient进行量化区间的动态调整...
    #         else:
    #             gradient = torch.randn_like(self.weight)  # 示例，实际应使用真实的梯度
    #             print("Gradient is not available. Please ensure that backward has been called before updating.")
    #         # 移动数据到CPU上计算直方图
    #         gradient_cpu = gradient.flatten().to('cpu')
    #         # 动态调整量化区间
    #         if self.prev_weight is None or self.prev_gradient is None:
    #             lower_bound = self.weight - self.sparsity * torch.abs(self.weight)
    #             upper_bound = self.weight + self.sparsity * torch.abs(self.weight)
    #         else:
    #             prev_gradient_cpu = self.prev_gradient.flatten().to('cpu')
    #             hist_current, _ = torch.histogram(gradient_cpu, bins=self.histogram_bins, range=(-1, 1))
    #             hist_prev, _ = torch.histogram(prev_gradient_cpu, bins=self.histogram_bins,
    #                                            range=(-1, 1)) if self.prev_gradient is not None else (hist_current, _)
    #             diff_hist = torch.abs(hist_current - hist_prev)
    #
    #             # 选取变化最大的top_k_bins区间
    #             top_diff_indices = torch.topk(diff_hist, k=5)[1]
    #             avg_bin_loc = torch.mean(top_diff_indices.float()) / self.histogram_bins  # 平均位置作为调整因子的一个参考
    #
    #             # 结合直方图分析与梯度相关性调整量化间隔
    #             grad_corr = F.cosine_similarity(gradient.flatten(), self.prev_gradient.flatten(),
    #                                             dim=0) if self.prev_gradient is not None else 0
    #             adjustment_factor = (avg_bin_loc + torch.clamp(grad_corr, -1, 1) * 0.1) / 2
    #
    #             lower_bound = self.weight - self.sparsity * adjustment_factor * torch.abs(self.weight)
    #             upper_bound = self.weight + self.sparsity * adjustment_factor * torch.abs(self.weight)
    #
    #         # 量化操作
    #         quantized_weight = torch.clamp(self.weight, lower_bound, upper_bound)
    #
    #         # 应用量化后的权重
    #         self.weight.data.copy_(quantized_weight)
    #
    #         # 更新记录
    #         self.prev_weight = self.weight.clone()
    #         self.prev_gradient = gradient.clone()

    def update_weight(self):
        with torch.no_grad():
            # 计算当前权重的梯度（这里简化处理，实际训练中应从优化器获取）
            # 注意：实际应用中，这一步应当在反向传播后，优化器.step()之前执行
            # gradient = torch.randn_like(self.weight)  # 示例，实际应使用真实的梯度

            # 确保在此前已经进行了loss.backward()操作并计算了梯度
            if self.weight.grad is not None:  # 检查梯度是否存在，避免初次未计算的情况
                gradient = self.weight.grad.clone()  # 使用真实的梯度，克隆一份防止外部修改影响
                # 接下来的逻辑保持不变，使用gradient进行量化区间的动态调整...
            else:
                gradient = torch.randn_like(self.weight)  # 示例，实际应使用真实的梯度
                print("Gradient is not available. Please ensure that backward has been called before updating.")

            # 动态调整量化区间
            if self.prev_weight is None or self.prev_gradient is None:
                lower_bound = self.weight - self.sparsity * torch.abs(self.weight)
                upper_bound = self.weight + self.sparsity * torch.abs(self.weight)
            else:
                # 基于梯度的动态调整
                grad_diff = gradient - self.prev_gradient
                weight_diff = self.weight - self.prev_weight
                correlation = F.cosine_similarity(grad_diff.flatten(), weight_diff.flatten(), dim=0)

                # 根据梯度与权重变化的相关性调整量化间隔
                adjustment_factor = torch.clamp(correlation, -1, 1) * 0.1 + 1  # 示例调整因子，可调
                lower_bound = self.weight - self.sparsity * adjustment_factor * torch.abs(self.weight)
                upper_bound = self.weight + self.sparsity * adjustment_factor * torch.abs(self.weight)

            # 量化操作
            quantized_weight = torch.clamp(self.weight, lower_bound, upper_bound)

            # 应用量化后的权重
            self.weight.data.copy_(quantized_weight)

            # 更新记录
            self.prev_weight = self.weight.clone()
            self.prev_gradient = gradient.clone()


    # def forward(self, x):
    #     binary_weight = self.weight.sign()  # Binarize weights to {-1, 1}
    #     return F.conv2d(x, binary_weight, stride=self.stride, padding=self.padding)

    # def quantize_weights(self, weights):
    #     # 量化为0, 1, -1
    #     quantized_weights = torch.where(weights > 0, torch.ones_like(weights),
    #                                     torch.where(weights < 0, -torch.ones_like(weights),
    #                                                 torch.zeros_like(weights)))
    #     return quantized_weights
    #
    # def forward(self, x):
    #     # 在每次前向传播时量化权重
    #     quantized_weights = self.quantize_weights(self.original_weight)
    #     self.weight.data.copy_(quantized_weights)
    #     return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



# 注册钩子函数以在反向传播后量化权重
# def quantize_weights_hook(module, grad_input, grad_output):
#     if isinstance(module, ConvLBP):
#         module.weight.data = module.quantize_weights(module.weight.data)

def save_convlbp_weights_to_txt(epoch, model, file_path):
    with open(file_path, 'a') as f:
        f.write("Epoch:" + str(epoch) + '\n')
        for name, module in model.named_modules():
            if isinstance(module, ConvLBP):
                f.write(f'Layer: {name}\n')
                weights = module.weight.data.cpu().numpy()
                for kernel in weights:
                    f.write(f'{kernel}\n')
                f.write('\n')

class BlockLBP(nn.Module):
    def __init__(self, batch_size, out_dim, kernel_size, sparsity=0.5):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(batch_size)
        self.conv_lbp = ConvLBP(batch_size, out_dim, kernel_size=(kernel_size, 2), sparsity=sparsity)
        self.conv_1x1 = nn.Conv2d(out_dim, batch_size, kernel_size=(kernel_size, 1))

    def forward(self, x):
        x = self.batch_norm(x)

        x = self.conv_lbp(x)
        x = F.relu(x)
        x = self.conv_1x1(x)
        return x

class Lbcnn(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.embedding_dim = hparams.embedding_dim
        self.dropout = nn.Dropout(hparams.dropout)
        self.tagset_size = hparams.tagset_size
        self.activation = nn.Sigmoid()
        self.num_words = hparams.num_words
        self.dense = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.pretrained_model_path = hparams.pretrained_model_path or 'bert_uncased_L-12_H-768_A-12'  # 预训练模型


        # Embedding layer definition 词典长度：self.num_words + 1、向量维度 self.embedding_dim
        self.bert_model = BertModel.from_pretrained(self.pretrained_model_path)  # 加载预训练模型

        # self.embedding = nn.Embedding(self.num_words, self.embedding_dim, padding_idx=0)
        # embeding_vector = load_word2vec(hparams.pretrained_word_vectors, self.embedding_dim, vocab)
        # self.embedding.weight.data.copy_(torch.from_numpy(embeding_vector))  # 表示的是在反向传播的时候, 是否对这些词向量进行求导更新

        self.preprocess_block1 = nn.Sequential(
            nn.Conv2d(1, hparams.train_batch_size, (hparams.kernel_size1, hparams.embedding_dim)),
            nn.BatchNorm2d(hparams.train_batch_size),
            nn.ReLU(inplace=True)
        )

        chain1 = [BlockLBP(hparams.train_batch_size, hparams.out_dim, hparams.kernel_size1, hparams.sparsity) for i in range(hparams.depth)]
        self.chained_blocks1 = nn.Sequential(*chain1)
        self.pool1 = nn.AvgPool2d(kernel_size=(hparams.kernel_size1, 1), stride=1)
        self.fc1 = nn.Linear(17024, self.tagset_size)

    def forward(self, token_ids, token_type_ids, attention_mask, e1_mask, e2_mask):
        sequence_output, pooled_output = self.bert_model(input_ids=token_ids, token_type_ids=token_type_ids,
                                                         attention_mask=attention_mask,
                                                         return_dict=False)  # sequence_output [32 168 768]
        sequence_output = sequence_output.unsqueeze(1)
        x = self.preprocess_block1(sequence_output)

        # 量化所有 ConvLBP 层的权重
        # for module in self.modules():
        #     if isinstance(module, ConvLBP):
        #         module.weight.data = module.quantize_weights(module.weight.data)

        x = self.chained_blocks1(x)
        x1 = self.pool1(x)
        x1 = x1.view(x1.shape[0], -1)
        x = self.fc1(self.dropout(x1))
        out = self.dropout(x)
        out = out.squeeze()
        return out

# class DiscretizeGrad(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         return torch.where(input > 0, torch.ones_like(input), torch.where(input < 0, -torch.ones_like(input), torch.zeros_like(input)))
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output