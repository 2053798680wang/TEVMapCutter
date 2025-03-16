import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
# 定义 FullyConnected 模块
class FullyConnected(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FullyConnected, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)  # 使用传入的 in_dim 和 out_dim
        self.relu = nn.ReLU()  # ReLU 激活函数
        self.bn = nn.BatchNorm1d(out_dim)  # 批归一化

    def forward(self, z0, z1):
        # 扩展 z0 和 z1
        z0_expanded = z0.unsqueeze(2)  # [batch_size, seq_len0, 1, in_dim]
        z1_expanded = z1.unsqueeze(1)  # [batch_size, 1, seq_len1, in_dim]

        # 将 z0 和 z1 相加
        combined = z0_expanded + z1_expanded  # [batch_size, seq_len0, seq_len1, in_dim]
        print("combine:", combined.size())

        # 通过线性层
        batch_size, seq_len0, seq_len1, in_dim = combined.size()
        combined = combined.view(-1, in_dim)  # [batch_size * seq_len0 * seq_len1, in_dim]
        output = self.fc(combined)  # [batch_size * seq_len0 * seq_len1, out_dim]

        # 通过批归一化
        output = self.bn(output)  # 批归一化，输入形状为 [batch_size * seq_len0 * seq_len1, out_dim]
        output = output.view(batch_size, seq_len0, seq_len1, -1)  # [batch_size, seq_len0, seq_len1, out_dim]

        # 通过 ReLU 激活函数
        output = self.relu(output)  # [batch_size, seq_len0, seq_len1, out_dim]
        print("out_dim:", output.size())

        return output


# 定义 ContactCNN 模块
class ContactCNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, width=3, activation=nn.Sigmoid()):
        super(ContactCNN, self).__init__()
        self.hidden = FullyConnected(in_dim, hidden_dim)
        self.conv = nn.Conv2d(hidden_dim, 1, width, padding=width // 2)
        self.batchnorm = nn.BatchNorm2d(1)
        self.activation = activation
        self.clip()

        # 用于存储卷积层的输出和梯度
        self.conv_output = None
        self.conv_grad = None

        # 注册hook
        self.conv.register_forward_hook(self.save_output)
        self.conv.register_backward_hook(self.save_grad)

    def clip(self):
        w = self.conv.weight
        print("w:", w.size())
        self.conv.weight.data[:] = 0.5 * (w + w.transpose(2, 3))

    def save_output(self, module, input, output):
        self.conv_output = output

    def save_grad(self, module, grad_input, grad_output):
        self.conv_grad = grad_output[0]

    def forward(self, z0, z1):
        C = self.cmap(z0, z1)
        return self.predict(C)

    def cmap(self, z0, z1):
        C = self.hidden(z0, z1)
        return C

    def predict(self, C):
        # 调整 C 的形状，使其符合 nn.Conv2d 的输入要求
        C = C.permute(0, 3, 1, 2)  # [batch_size, hidden_dim, seq_len0, seq_len1]

        # S is (b, 1, N, M)
        s = self.conv(C)
        print("s:", s.size())
        s = self.batchnorm(s)
        s = self.activation(s)

        # 移除大小为 1 的维度
        s = s.squeeze(1)  # [batch_size, seq_len0, seq_len1]
        print("s:", s.size())
        return s


# 定义 ProteinInteractionModel 模块
class ProteinInteractionModel(nn.Module):
    def __init__(self, input_dim_protein, in_dim, d=64, dropout_rate=0.5):
        super(ProteinInteractionModel, self).__init__()

        # Shared projection layer for both proteins
        self.projection = nn.Sequential(
            nn.Linear(input_dim_protein, d),  # Linear layer for dimensionality reduction
            nn.ReLU(),  # ReLU activation
            nn.Dropout(dropout_rate)  # Dropout for regularization
        )

        # Residue Contact Module
        self.residue_contact = ContactCNN(in_dim, hidden_dim=d, width=5)  # 确保 in_dim 和 hidden_dim 正确传递

        # Interaction Prediction Module
        self.interaction_prediction = nn.Sequential(
            nn.Linear(242 * 10, 64),  # 输入维度为 242 * 10
            nn.ReLU(),  # ReLU 激活函数
            nn.Linear(64, 1),  # 输出维度为 1
            nn.Sigmoid()  # Sigmoid 输出概率
        )

    def forward(self, peptide1, protease2):
        # Apply projection (Linear → ReLU → Dropout) to both inputs
        peptide1 = self.projection(peptide1)
        protease2 = self.projection(protease2)

        # 调用 ContactCNN 的 forward 方法
        contact_map = self.residue_contact(peptide1, protease2)

        # Reshape contact_map to [batch_size, 1, height, width]
        # Flatten the contact_map for the Interaction Prediction Module
        contact_map_flattened = contact_map.view(contact_map.size(0), -1)  # [batch_size, 242 * 10]

        # Interaction Prediction
        interaction_prob = self.interaction_prediction(contact_map_flattened)

        return interaction_prob, contact_map


# 定义数据集
class ProteinInteractionDataset(Dataset):
    def __init__(self, peptidase_tensors, protease_tensors, labels):
        self.peptidase_tensors = peptidase_tensors
        self.protease_tensors = protease_tensors
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.peptidase_tensors[idx], self.protease_tensors[idx], self.labels[idx]


# 定义 Grad-CAM
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def __call__(self, peptide, protease):
        # 前向传播
        output, _ = self.model(peptide, protease)
        output = output.mean()  # 假设我们关注的是输出的平均值

        # 反向传播
        self.model.zero_grad()
        output.backward()

        # 获取卷积层的输出和梯度
        conv_output = self.model.residue_contact.conv_output
        conv_grad = self.model.residue_contact.conv_grad

        # 计算权重
        weights = torch.mean(conv_grad, dim=(2, 3), keepdim=True)

        # 计算Grad-CAM
        grad_cam = torch.sum(weights * conv_output, dim=1, keepdim=True)
        grad_cam = torch.relu(grad_cam)  # ReLU激活

        # 归一化
        grad_cam -= grad_cam.min()
        grad_cam /= grad_cam.max()

        return grad_cam.squeeze().detach().cpu().numpy()

def load_embeddings(directory):
    return {
        file_name.strip('.csv'): pd.read_csv(os.path.join(directory, file_name.strip(' ')), header=None,
                                             index_col=False).to_numpy()
        for file_name in os.listdir(directory)
    }

def pad_embedding(embedding, target_shape):
    """将嵌入向量补零到目标形状"""
    current_shape = embedding.shape
    if current_shape == target_shape:
        return embedding
    padded_embedding = np.zeros(target_shape)
    padded_embedding[:current_shape[0], :current_shape[1]] = embedding
    return padded_embedding

def load_new_data(peptide_embedding_dir, protease_embedding_dir, data_csv_path):
    # 加载肽和蛋白酶的嵌入向量
    peptide_embeddings = load_embeddings(peptide_embedding_dir)
    protease_embeddings = load_embeddings(protease_embedding_dir)

    # 读取新数据的 CSV 文件
    new_data = pd.read_csv(data_csv_path)
    name=new_data["id"]

    # 获取肽和蛋白酶的嵌入向量，并进行补零操作
    peptidase_list = [pad_embedding(peptide_embeddings[id_], (10, 320)) for id_ in new_data['id']]
    protease_list = [pad_embedding(protease_embeddings[pro], (242, 320)) for pro in new_data['Protease']]

    # 将数据转换为 PyTorch 张量
    peptidase_tensors = [torch.tensor(embedding, dtype=torch.float32) for embedding in peptidase_list]
    protease_tensors = [torch.tensor(embedding, dtype=torch.float32) for embedding in protease_list]
    # 对 peptidase_tensors 和 protease_tensors 进行标准化

    # 创建数据集
    labels = [0] * len(peptidase_tensors)  # 标签可以设为 0，因为我们不需要它们
    new_dataset = ProteinInteractionDataset(peptidase_tensors, protease_tensors, labels)

    # 创建 DataLoader
    new_loader = DataLoader(new_dataset, batch_size=512, shuffle=False)
    return peptidase_tensors,protease_tensors,name

# 主程序
if __name__ == "__main__":
    # 生成随机数据
    peptide_embedding_dir = r"D:\desk\PGCN\PDB结构研究\predict\peptidease"
    protease_embedding_dir = r"D:\desk\PGCN\PDB结构研究\predict\protease"
    data_csv_path = r"D:\desk\PGCN\PDB结构研究\predict\predict_seq_test.csv"

    # 加载模型


    # 加载新数据并进行标准化
    pep,pro,name = load_new_data(
        peptide_embedding_dir, protease_embedding_dir, data_csv_path
    )


    # 加载模型
    model = ProteinInteractionModel(input_dim_protein=320, in_dim=64, d=64)
    model.load_state_dict(
        torch.load(r"D:\python-learning\Autodock_CNN_Map\best_model_1232345.pth", map_location=torch.device('cpu'))
    )
    model.eval()

    # 创建 Grad-CAM 对象
    grad_cam = GradCAM(model)
    for i in range(len(pep)):
        print(name[i])
    # 获取一个样本
        peptide_sample = pep[i].unsqueeze(0)  # [1, 10, 320]
        protease_sample = pro[i].unsqueeze(0)  # [1, 242, 320]
        print(protease_sample)

        # 计算 Grad-CAM
        heatmap = grad_cam(peptide_sample, protease_sample)
        print("heatmap:",heatmap)

        # 将 heatmap 转换为 DataFrame
        heatmap_df = pd.DataFrame(heatmap)
        print(heatmap_df)

    # 保存为 CSV 文件
    # heatmap_df.to_csv("D:/desk/PGCN/PDB结构研究/predict/MAP/heat_map/heatmap.csv", index=False, header=False)
    # print("Heatmap saved to heatmap.csv")
    '''

    # 设置科研风格
    sns.set_style("whitegrid")  # 使用白色网格背景
    plt.rcParams['font.size'] = 12  # 设置字体大小
    plt.rcParams['axes.labelsize'] = 14  # 坐标轴标签字体大小
    plt.rcParams['axes.titlesize'] = 16  # 标题字体大小


    # 可视化热力图
    def plot_heatmap(heatmap, title="Grad-CAM Heatmap", cmap="viridis"):
        """
        绘制科研风格的热力图
        :param heatmap: 热力图数据 (2D numpy array)
        :param title: 图的标题
        :param cmap: 颜色方案，默认为 'viridis'
        """
        plt.figure(figsize=(8, 6))  # 设置图的大小
        sns.heatmap(heatmap, cmap=cmap, annot=False, cbar=True, square=True)
        plt.title(title, fontweight="bold")
        plt.xlabel("Protease Residue Index")
        plt.ylabel("Peptide Residue Index")
        plt.tight_layout()  # 自动调整布局
        plt.show()


    # 调用函数绘制热力图
    plot_heatmap(heatmap, title="Grad-CAM Heatmap of Protein Interaction", cmap="RdBu")
    '''