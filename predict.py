import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from Model import ProteinInteractionModel
import random
from sklearn.preprocessing import StandardScaler

# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ProteinInteractionDataset(torch.utils.data.Dataset):
    def __init__(self, peptidase_tensors, protease_tensors, labels):
        self.peptidase_tensors = peptidase_tensors
        self.protease_tensors = protease_tensors
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.peptidase_tensors[idx], self.protease_tensors[idx], self.labels[idx]


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


def load_model(model_path):
    input_dim_protein = 320  # protease_tensors[0].shape[1]  # 假设所有嵌入向量的维度相同
    d = 64
    # hidden_dim = 64
    # output_dim = 1
    # max_len = 242  # 假设最大长度为242
    model = ProteinInteractionModel(input_dim_protein, d)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # 将模型加载到CPU
    model.eval()  # 设置为评估模式
    return model

def load_new_data(peptide_embedding_dir, protease_embedding_dir, data_csv_path):
    # 加载肽和蛋白酶的嵌入向量
    peptide_embeddings = load_embeddings(peptide_embedding_dir)
    protease_embeddings = load_embeddings(protease_embedding_dir)

    # 读取新数据的 CSV 文件
    new_data = pd.read_csv(data_csv_path)

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
    return new_loader, new_data, list(new_data['id']), list(new_data['Protease_seq']), list(new_data['seq'])


def predict(model, data_loader):
    model.eval()
    all_predictions = []
    all_contact_maps = []

    with torch.no_grad():
        for peptidase, protease, _ in data_loader:  # 不需要标签
            peptidase, protease = peptidase, protease
            output = model(peptidase, protease)[0]

            # 获取 contact_map
            contact_map = model(peptidase, protease)[1]  # 假设你在模型中保存了 contact_map
            all_contact_maps.extend(contact_map.cpu().numpy())
            all_predictions.extend(output.cpu().numpy())

    return all_predictions, all_contact_maps


def main():
    # 模型路径
    model_path = r"D:\python-learning\Autodock_CNN_Map\best_model_1232345.pth"

    # 新数据的路径
    peptide_embedding_dir = r"D:\desk\try\embedding\peptidase_embedding"
    protease_embedding_dir = r"D:\desk\PGCN\特异性改变突变\mutated_protein_csv"
    data_csv_path = r"D:\desk\PGCN\特异性改变突变\顺序output.csv"

    # 加载模型
    model = load_model(model_path)

    # 加载新数据并进行标准化
    new_loader, new_data, name_list, protease_list, peptide_list = load_new_data(
        peptide_embedding_dir, protease_embedding_dir, data_csv_path
    )

    # 进行预测
    predictions, contact_maps = predict(model, new_loader)


    # 将预测结果转换为二进制分类结果（0 或 1）
    binary_predictions = [1 if prob > 0.5 else 0 for prob in predictions]
    predictions_list=[prob.item() for prob in predictions]
    print(name_list)
    print(binary_predictions)
    print(predictions_list)
    name_list_list=[]
    a=pd.DataFrame([name_list,binary_predictions,predictions_list]).T
    a.to_csv(r"D:\desk\PGCN\特异性改变突变\顺序output_predict.csv")
    # 将每个二维张量转换为 DataFrame，并存储到列表中
    # dataframes = [pd.DataFrame(contact_maps[i]) for i in range(len(contact_maps))]

    # 打印每个 DataFrame
    # for i, df in enumerate(dataframes):
    #     print(fr"{name_list[i]}: {i + 1}:")
    #     name_list_list.append(name_list[i])
    #     print("\n")


if __name__ == "__main__":
    main()