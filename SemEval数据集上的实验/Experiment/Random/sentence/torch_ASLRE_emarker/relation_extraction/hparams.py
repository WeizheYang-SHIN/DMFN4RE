import os
from torch import torch
import argparse

here = os.path.dirname(os.path.abspath(__file__))
# default_pretrained_model_path = os.path.join(here, '../../../bert_uncased_L-12_H-768_A-12')
default_pretrained_model_path = os.path.join('E:\pycharm\workspace\Phd\预训练语言模型/bert_uncased_L-12_H-768_A-12')
default_pretrained_word_vectors_path = os.path.join('/remote-home/TCCI19/phd/预训练词向量/GoogleNews-vectors-negative300.bin')
# default_pretrained_word_vectors_path = os.path.join('E:\pycharm\workspace\Phd\预训练词向量/GoogleNews-vectors-negative300.bin')
default_train_file = os.path.join(here, '../datasets/SemEval2010_task8_all_data/SemEval2010_task8_training/process_TRAIN_FILE.json')
default_validation_file = os.path.join(here, '../datasets/SemEval2010_task8_all_data/SemEval2010_task8_training/process_TRAIN_FILE.json')
default_test_file = os.path.join(here, '../datasets/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/process_TEST_FILE_FULL.json')
default_tagset_file = os.path.join(here, '../datasets/SemEval2010_task8_all_data/tags.txt')

#default_train_file = os.path.join('E:\pycharm\workspace\Phd\Relation-Extraction/7_Seventh-Ada-GCN-dis\-------SemEval-dataset\SemEval2010_task8_all_data/SemEval2010_task8_training/process_TRAIN_FILE.json')
#default_validation_file = os.path.join('E:\pycharm\workspace\Phd\Relation-Extraction/7_Seventh-Ada-GCN-dis\-------SemEval-dataset\SemEval2010_task8_all_data/SemEval2010_task8_training/process_TRAIN_FILE.json')
#default_test_file = os.path.join('E:\pycharm\workspace\Phd\Relation-Extraction/7_Seventh-Ada-GCN-dis\-------SemEval-dataset\SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/process_TEST_FILE_FULL.json')
#default_tagset_file = os.path.join('E:\pycharm\workspace\Phd\Relation-Extraction/7_Seventh-Ada-GCN-dis\-------SemEval-dataset\SemEval2010_task8_all_data/tags.txt')
default_output_dir = os.path.join(here, '../saved_models')
default_log_dir = os.path.join(default_output_dir, 'runs')
default_model_file = os.path.join(default_output_dir, 'model.bin')
default_checkpoint_file = os.path.join(default_output_dir, 'checkpoint.json')

parser = argparse.ArgumentParser()
# ========== 模型架构参数 ==========
# 图卷积相关
parser.add_argument("--gcn_hidden_dim", type=int, default=256,
                    help="图卷积隐藏层维度")
parser.add_argument("--num_gcn_layers", type=int, default=3,
                    help="GCN层堆叠数量")
parser.add_argument("--k_neighbors", type=int, default=5,
                    help="动态邻接矩阵的topk连接数")
parser.add_argument("--gcn_sparsity", type=float, default=0.7,
                    help="离散GCN层的稀疏率")

# 卷积相关
parser.add_argument("--conv_out_channels", type=int, default=32,
                    help="离散卷积输出通道数")
parser.add_argument("--conv_kernel_size", type=int, default=3,
                    help="卷积核尺寸")

# 通用结构参数
parser.add_argument("--entity_dim", type=int, default=64,
                    help="实体特征投影维度")
parser.add_argument("--fusion_dim", type=int, default=512,
                    help="特征融合层维度")
parser.add_argument("--num_structure_layers", type=int, default=3,
                    help="结构学习模块堆叠层数")

# ========== 训练参数 ==========
parser.add_argument("--seed", type=int, default=12360)
parser.add_argument("--max_len", type=int, default=148,
                    help="最大序列长度")
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--validation_batch_size", type=int, default=32)
parser.add_argument("--test_batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=5000)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0.001)
parser.add_argument("--dropout", type=float, default=0.05,
                    help="全连接层dropout率")
parser.add_argument('--embedding_dim', type=int, default=300, required=False, help='embedding_dim')
parser.add_argument('--num_words', type=int, default=2500)
parser.add_argument("--out_dim", type=int, default=30)
# 在超参数中添加模式标识
parser.add_argument("--mode", type=str, default="train",
                    choices=["train", "predict"])
parser.add_argument("--freeze_embedding", type=bool, default=False)

# ========== 路径参数 ==========
parser.add_argument("--pretrained_model_path", type=str, default=default_pretrained_model_path)
parser.add_argument("--pretrained_word_vectors",type=str,default=default_pretrained_word_vectors_path)
parser.add_argument("--train_file", type=str, default=default_train_file)
parser.add_argument("--validation_file", type=str, default=default_validation_file)
parser.add_argument("--test_file", type=str, default=default_test_file)
parser.add_argument("--output_dir", type=str, default=default_output_dir)
parser.add_argument("--log_dir", type=str, default=default_log_dir)
parser.add_argument("--tagset_file", type=str, default=default_tagset_file)
parser.add_argument("--model_file", type=str, default=default_model_file)
parser.add_argument("--checkpoint_file", type=str, default=default_checkpoint_file)

# ========== 设备参数 ==========
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="训练设备(cuda/cpu)")
# parser.add_argument('--device', type=str, default='cuda')

hparams = parser.parse_args()

# parser.add_argument("--numWeights",type=int,default=32)
# parser.add_argument("--full",type=int,default=36)
# parser.add_argument("--depth",type=int,default=6)
# parser.add_argument("--sparsity",type=int,default=0.5)





