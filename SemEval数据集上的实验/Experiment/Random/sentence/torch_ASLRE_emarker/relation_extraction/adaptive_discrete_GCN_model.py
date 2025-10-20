# coding: utf-8
# _*_ coding: utf-8 _*_
# @Time : 2024/5/21 14:46 
# @Author : wz.yang 
# @File : model.py
# @desc :
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
from .data_utils import *
from transformers import BertTokenizer, BertModel
from .hparams import hparams

class DynamicAdjacency(nn.Module):
    """自适应邻接矩阵生成层,基于节点特征的余弦相似度动态生成稀疏邻接矩阵"""

    def __init__(self, node_dim, k_neighbors=hparams.k_neighbors):
        super().__init__()
        self.k = k_neighbors
        self.proj = nn.Linear(node_dim, node_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, num_nodes, node_dim)
        batch_size, num_nodes, _ = x.shape

        # 计算节点相似度
        x_norm = F.normalize(x, p=2, dim=-1)
        sim_matrix = torch.bmm(x_norm, x_norm.transpose(1, 2))  # (B,N,N)

        # 保持topk连接
        topk = min(self.k, num_nodes)
        values, indices = torch.topk(sim_matrix, topk, dim=-1)

        # 创建稀疏邻接矩阵
        mask = torch.zeros_like(sim_matrix)
        mask.scatter_(-1, indices, values)

        # 对称化处理
        mask = (mask + mask.transpose(1, 2)) / 2
        return mask


class DiscreteGCNLayer(nn.Module):
    """离散化图卷积层,在每次前向传播时对权重进行三值量化（-sparsity, 0, +sparsity）"""

    def __init__(self, in_dim, out_dim, sparsity=hparams.gcn_sparsity):
        super().__init__()
        self.sparsity = sparsity
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def quantize_weights(self):
        """三值量化策略"""
        with torch.no_grad():
            pos_mask = (self.weight > self.sparsity).float()
            neg_mask = (self.weight < -self.sparsity).float()
            discrete_weight = pos_mask - neg_mask
            return discrete_weight * self.sparsity

    def forward(self, x, adj):
        # x: (B, N, D_in)
        # adj: (B, N, N)

        # 应用离散化权重
        discrete_weight = self.quantize_weights()

        # 图卷积操作
        support = torch.matmul(x, discrete_weight)  # (B,N,D_out)
        output = torch.bmm(adj, support) + self.bias

        return F.relu(output)


class AdaptiveStructureBlock(nn.Module):
    """自适应结构学习模块，结合了图卷积和离散卷积两个分支，通过残差连接融合两种结构学习方式"""

    def __init__(self, node_dim, k_neighbors=3):
        super().__init__()
        self.adj_generator = DynamicAdjacency(node_dim, k_neighbors)
        self.gcn_layer = DiscreteGCNLayer(node_dim, node_dim)
        self.conv_lbp = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=hparams.conv_kernel_size, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=hparams.conv_kernel_size, padding=1)
        )

    def forward(self, x):
        # x: (B, N, D)
        adj = self.adj_generator(x)  # 动态邻接矩阵

        # 图卷积分支
        gcn_out = self.gcn_layer(x, adj)

        # 离散卷积分支
        conv_in = x.unsqueeze(1)  # 增加通道维度
        conv_out = self.conv_lbp(conv_in).squeeze(1)

        # 特征融合
        combined = gcn_out + conv_out
        return combined


class ASLREModel(nn.Module):
    """自适应语义结构学习的关系抽取模型"""

    def __init__(self, hparams, vocab):
        super().__init__()
        self.num_labels = hparams.num_labels
        self.num_words = hparams.num_words
        self.embedding_dim = hparams.embedding_dim

        # self.bert_model = BertModel.from_pretrained(hparams.pretrained_model_path)
        # 替换BERT为普通Embedding层，Embedding layer definition 词典长度：self.num_words + 1、向量维度 self.embedding_dim
        # 仅初始化Embedding结构
        self.embedding = nn.Embedding(
            num_embeddings=self.num_words,
            embedding_dim=self.embedding_dim,
            padding_idx=0
        )
        # 仅在训练时初始化预训练权重
        if hparams.mode == 'train':  # 新增模式判断
            print("Initializing with pretrained word vectors")
            # self.embedding = nn.Embedding(self.num_words, self.embedding_dim, padding_idx=0)
            embeding_vector = load_word2vec(hparams.pretrained_word_vectors, self.embedding_dim, vocab)
            self.embedding.weight.data.copy_(torch.from_numpy(embeding_vector))  # 表示的是在反向传播的时候, 是否对这些词向量进行求导更新

            # 是否冻结词向量
            if hparams.freeze_embedding:
                self.embedding.weight.requires_grad = False
        else:
            print("Skipping pretrained vectors in prediction mode")

        self.entity_proj = nn.Linear(hparams.embedding_dim*2, hparams.entity_dim)

        # 结构学习模块堆叠
        self.structure_blocks = nn.ModuleList([
            AdaptiveStructureBlock(hparams.embedding_dim + hparams.entity_dim)
            for _ in range(hparams.num_structure_layers)
        ])

        # 上下文感知门控，通过可学习的门控机制动态调整各层特征的比例，实现自适应特征融合
        self.gate = nn.Sequential(
            nn.Linear((hparams.embedding_dim + hparams.entity_dim) * 2, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(hparams.embedding_dim + hparams.entity_dim, hparams.gcn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hparams.gcn_hidden_dim, self.num_labels)
        )

    def forward(self, input_ids, e1_mask, e2_mask):
        # 获取BERT编码
        # sequence_output, pooled_output = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
        #                                                  attention_mask=attention_mask,
        #                                                  return_dict=False)  # sequence_output [32 168 768]
        # outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        # sequence_output = outputs.last_hidden_state  # (B, L, D)

        # 获取词向量
        sequence_output = self.embedding(input_ids)  # (B, L, D)

        # 双实体特征提取
        def extract_entity_features(mask):
            # mask: (B, L)
            mask = mask.unsqueeze(-1)  # (B, L, 1)
            sum_features = torch.sum(sequence_output * mask, dim=1)  # (B, D)
            sum_mask = torch.sum(mask, dim=1) + 1e-13  # 防止除零
            return sum_features / sum_mask

        # 提取并融合双实体特征
        e1_features = extract_entity_features(e1_mask)  # (B, D)
        e2_features = extract_entity_features(e2_mask)  # (B, D)
        entity_features = torch.cat([e1_features, e2_features], dim=-1)  # (B, 2D)
        entity_features = self.entity_proj(entity_features)  # (B, E)

        # 构造增强特征
        entity_features = entity_features.unsqueeze(1).expand(-1, sequence_output.size(1), -1)
        enhanced_features = torch.cat([sequence_output, entity_features], dim=-1)

        # 结构学习过程
        structural_features = enhanced_features
        for block in self.structure_blocks:
            residual = structural_features
            structural_features = block(structural_features)

            # 自适应门控残差连接
            gate_input = torch.cat([residual, structural_features], dim=-1)
            gate_value = self.gate(gate_input)
            structural_features = gate_value * residual + (1 - gate_value) * structural_features

        # 特征聚合与分类
        pooled = structural_features.mean(dim=1)
        # pooled = structural_features.max(dim=1)
        logits = self.classifier(pooled)
        return logits





# class ConvLBP(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, sparsity=0.5):
#         super().__init__(in_channels, out_channels, kernel_size, padding=1, bias=False)
#         self.sparsity = sparsity
#         self.initialize_weights()
#
#     def initialize_weights(self):
#         weights = self.weight.data
#         # matrix_proba = torch.FloatTensor(weights.shape).fill_(0.5)
#         # binary_weights = torch.bernoulli(matrix_proba) * 2 - 1
#         # mask_inactive = torch.rand(matrix_proba.shape) > self.sparsity
#         # binary_weights.masked_fill_(mask_inactive, 0)
#         # self.original_weight = nn.Parameter(binary_weights.clone())  # 保存原始二值权重作为参数
#         self.weight.data.copy_(weights)
#         self.weight.requires_grad_(True)
#         # 添加一个历史趋势变量，用于记录权重变化趋势
#         self.prev_mean = None
#         self.prev_std = None
#         # self.weight.data = self.quantize_weights(self.weight.data)
#
#     def update_weight(self):
#         with torch.no_grad():
#             # 计算当前权重的梯度（这里简化处理，实际训练中应从优化器获取）
#             # 注意：实际应用中，这一步应当在反向传播后，优化器.step()之前执行
#             # gradient = torch.randn_like(self.weight)  # 示例，实际应使用真实的梯度
#
#             # 确保在此前已经进行了loss.backward()操作并计算了梯度
#             if self.weight.grad is not None:  # 检查梯度是否存在，避免初次未计算的情况
#                 gradient = self.weight.grad.clone()  # 使用真实的梯度，克隆一份防止外部修改影响
#                 # 接下来的逻辑保持不变，使用gradient进行量化区间的动态调整...
#             else:
#                 gradient = torch.randn_like(self.weight)  # 示例，实际应使用真实的梯度
#                 print("Gradient is not available. Please ensure that backward has been called before updating.")
#
#             # 动态调整量化区间
#             if self.prev_weight is None or self.prev_gradient is None:
#                 lower_bound = self.weight - self.sparsity * torch.abs(self.weight)
#                 upper_bound = self.weight + self.sparsity * torch.abs(self.weight)
#             else:
#                 # 基于梯度的动态调整
#                 grad_diff = gradient - self.prev_gradient
#                 weight_diff = self.weight - self.prev_weight
#                 correlation = F.cosine_similarity(grad_diff.flatten(), weight_diff.flatten(), dim=0)
#
#                 # 根据梯度与权重变化的相关性调整量化间隔
#                 adjustment_factor = torch.clamp(correlation, -1, 1) * 0.1 + 1  # 示例调整因子，可调
#                 lower_bound = self.weight - self.sparsity * adjustment_factor * torch.abs(self.weight)
#                 upper_bound = self.weight + self.sparsity * adjustment_factor * torch.abs(self.weight)
#
#             # 量化操作
#             quantized_weight = torch.clamp(self.weight, lower_bound, upper_bound)
#
#             # 应用量化后的权重
#             self.weight.data.copy_(quantized_weight)
#
#             # 更新记录
#             self.prev_weight = self.weight.clone()
#             self.prev_gradient = gradient.clone()
#
# def save_convlbp_weights_to_txt(epoch, model, file_path):
#     with open(file_path, 'a') as f:
#         f.write("Epoch:" + str(epoch) + '\n')
#         for name, module in model.named_modules():
#             if isinstance(module, ConvLBP):
#                 f.write(f'Layer: {name}\n')
#                 weights = module.weight.data.cpu().numpy()
#                 for kernel in weights:
#                     f.write(f'{kernel}\n')
#                 f.write('\n')
#
# class BlockLBP(nn.Module):
#     def __init__(self, batch_size, out_dim, kernel_size, sparsity=0.5):
#         super().__init__()
#         self.batch_norm = nn.BatchNorm2d(batch_size)
#         self.conv_lbp = ConvLBP(batch_size, out_dim, kernel_size=(kernel_size, 2), sparsity=sparsity)
#         self.conv_1x1 = nn.Conv2d(out_dim, batch_size, kernel_size=(kernel_size, 1))
#
#     def forward(self, x):
#         x = self.batch_norm(x)
#
#         x = self.conv_lbp(x)
#         x = F.relu(x)
#         x = self.conv_1x1(x)
#         return x
#
# class Lbcnn(nn.Module):
#     def __init__(self, hparams):
#         super().__init__()
#         self.embedding_dim = hparams.embedding_dim
#         self.dropout = nn.Dropout(hparams.dropout)
#         self.tagset_size = hparams.tagset_size
#         self.activation = nn.Sigmoid()
#         self.num_words = hparams.num_words
#         self.dense = nn.Linear(self.embedding_dim, self.embedding_dim)
#         self.pretrained_model_path = hparams.pretrained_model_path or 'bert_uncased_L-12_H-768_A-12'  # 预训练模型
#
#
#         # Embedding layer definition 词典长度：self.num_words + 1、向量维度 self.embedding_dim
#         self.bert_model = BertModel.from_pretrained(self.pretrained_model_path)  # 加载预训练模型
#
#         # self.embedding = nn.Embedding(self.num_words, self.embedding_dim, padding_idx=0)
#         # embeding_vector = load_word2vec(hparams.pretrained_word_vectors, self.embedding_dim, vocab)
#         # self.embedding.weight.data.copy_(torch.from_numpy(embeding_vector))  # 表示的是在反向传播的时候, 是否对这些词向量进行求导更新
#
#         self.preprocess_block1 = nn.Sequential(
#             nn.Conv2d(1, hparams.train_batch_size, (hparams.kernel_size1, hparams.embedding_dim)),
#             nn.BatchNorm2d(hparams.train_batch_size),
#             nn.ReLU(inplace=True)
#         )
#
#         chain1 = [BlockLBP(hparams.train_batch_size, hparams.out_dim, hparams.kernel_size1, hparams.sparsity) for i in range(hparams.depth)]
#         self.chained_blocks1 = nn.Sequential(*chain1)
#         self.pool1 = nn.AvgPool2d(kernel_size=(hparams.kernel_size1, 1), stride=1)
#         self.fc1 = nn.Linear(17024, self.tagset_size)
#
#     def forward(self, token_ids, token_type_ids, attention_mask, e1_mask, e2_mask):
#         sequence_output, pooled_output = self.bert_model(input_ids=token_ids, token_type_ids=token_type_ids,
#                                                          attention_mask=attention_mask,
#                                                          return_dict=False)  # sequence_output [32 168 768]
#         sequence_output = sequence_output.unsqueeze(1)
#         x = self.preprocess_block1(sequence_output)
#
#         # 量化所有 ConvLBP 层的权重
#         # for module in self.modules():
#         #     if isinstance(module, ConvLBP):
#         #         module.weight.data = module.quantize_weights(module.weight.data)
#
#         x = self.chained_blocks1(x)
#         x1 = self.pool1(x)
#         x1 = x1.view(x1.shape[0], -1)
#         x = self.fc1(self.dropout(x1))
#         out = self.dropout(x)
#         out = out.squeeze()
#         return out
#
#
#



# 自适应离散图卷积模型
