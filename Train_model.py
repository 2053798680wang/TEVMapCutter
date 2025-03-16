import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score


def load_embeddings(directory):
    return {
        file_name.strip('.csv'): pd.read_csv(os.path.join(directory, file_name.strip(' ')), header=None,
                                             index_col=False).to_numpy()
        for file_name in os.listdir(directory)
    }
# 加载肽和蛋白酶的嵌入向量
peptide_embeddings = load_embeddings(r"/peptidase")
protease_embeddings = load_embeddings(r"/protein")

# 创建数据集和数据加载器
class ProteinInteractionDataset(torch.utils.data.Dataset):
    def __init__(self, peptidase_tensors, protease_tensors, labels):
        self.peptidase_tensors = peptidase_tensors
        self.protease_tensors = protease_tensors
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.peptidase_tensors[idx], self.protease_tensors[idx], self.labels[idx]

def Dataset(path,peptide_embeddings,protease_embeddings):
    # 读取主Excel文件
    main_df = pd.read_csv(rf"{path}")
    # 获取肽和蛋白酶的嵌入向量列表
    peptidase_list = [peptide_embeddings[id_] for id_ in main_df['id']]
    protease_list = [protease_embeddings[pro] for pro in main_df['Protease']]

    # 将标签转换为one-hot编码
    labels = np.array(main_df['label'])

    # 将数据转换为PyTorch张量
    peptidase_tensors = [torch.tensor(embedding, dtype=torch.float32) for embedding in peptidase_list]
    protease_tensors = [torch.tensor(embedding, dtype=torch.float32) for embedding in protease_list]
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return peptidase_tensors,protease_tensors,labels_tensor

Train_tensor=Dataset(r"train.csv",peptide_embeddings,protease_embeddings)
Val_tensor=Dataset(r"val.csv",peptide_embeddings,protease_embeddings)
Test_tensor=Dataset(r"test.csv",peptide_embeddings,protease_embeddings)

protease_tensors=Train_tensor[1]
peptidase_tensors=Train_tensor[0]


Train_dataset=ProteinInteractionDataset(Train_tensor[0], Train_tensor[1], Train_tensor[2])
Val_dataset=ProteinInteractionDataset(Val_tensor[0], Val_tensor[1], Val_tensor[2])
Test_dataset=ProteinInteractionDataset(Test_tensor[0], Test_tensor[1], Test_tensor[2])

train_loader = DataLoader(Train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(Val_dataset, batch_size=10000, shuffle=False)
test_loader = DataLoader(Test_dataset, batch_size=10000, shuffle=False)

# 定义模型
class ContactCNN(nn.Module):
    def __init__(self, in_channels, hidden_dim=32, width=5, activation=nn.ReLU()):
        super(ContactCNN, self).__init__()

        # 1. 将输入通道数从 in_channels 降到 hidden_dim
        self.hidden = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

        # 2. 使用卷积层将 hidden_dim 降到 1，并调整空间维度
        self.conv = nn.Conv2d(hidden_dim, 1, kernel_size=(width, width), padding=(width // 2, width // 2))

        # 3. 使用 BatchNorm 和激活函数
        self.batchnorm = nn.BatchNorm2d(1)
        self.activation = activation

    def forward(self, z):
        # 输入 z 的形状: [batch_size, in_channels, height, width]
        # 1. 将通道数从 in_channels 降到 hidden_dim
        z = self.hidden(z)  # 输出形状: [batch_size, hidden_dim, height, width]

        # 2. 将通道数从 hidden_dim 降到 1
        z = self.conv(z)  # 输出形状: [batch_size, 1, height, width]

        # 3. 应用 BatchNorm 和激活函数
        z = self.batchnorm(z)
        z = self.activation(z)

        # 4. 移除通道维度，得到 [batch_size, height, width]
        z = z.squeeze(1)  # 输出形状: [batch_size, height, width]
        return z

class ProteinInteractionModel(nn.Module):
    def __init__(self, input_dim_protein, input_dim_peptide, d=64, dropout_rate=0.5):
        super(ProteinInteractionModel, self).__init__()

        # Protein 1 Projection (Linear → ReLU → Dropout)
        self.protein1_projection = nn.Sequential(
            nn.Linear(input_dim_protein, d),  # Linear layer for dimensionality reduction
            nn.ReLU(),  # ReLU activation
            nn.Dropout(dropout_rate)  # Dropout for regularization
        )

        # Protein 2 Projection (Linear → ReLU → Dropout)
        self.protein2_projection = nn.Sequential(
            nn.Linear(input_dim_peptide, d),  # Linear layer for dimensionality reduction
            nn.ReLU(),  # ReLU activation
            nn.Dropout(dropout_rate)  # Dropout for regularization
        )

        # Residue Contact Module
        self.residue_contact = ContactCNN(in_channels=2 * d, hidden_dim=d, width=5)

        # Interaction Prediction Module
        self.interaction_prediction = nn.Sequential(
            nn.Linear(242 * 10, 64),  # 输入维度为 242 * 10
            nn.ReLU(),  # ReLU 激活函数
            nn.Linear(64, 1),  # 输出维度为 1
            nn.Sigmoid()  # Sigmoid 输出概率
        )

    def forward(self, peptide1, protease2):
        # Apply projection (Linear → ReLU → Dropout) to both inputs
        peptide1 = self.protein1_projection(peptide1)
        protease2 = self.protein2_projection(protease2)

        # Expand and concatenate
        peptide1_expanded = peptide1.unsqueeze(1).expand(-1, 242, -1, -1)
        protease2_expanded = protease2.unsqueeze(2).expand(-1, -1, 10, -1)
        combined = torch.cat((peptide1_expanded, protease2_expanded), dim=-1)
        combined = combined.permute(0, 3, 1, 2)  # Adjust dimensions for Conv2d

        # Residue Contact Map
        contact_map = self.residue_contact(combined)
        # Reshape contact_map to [batch_size, 1, height, width]
        # Flatten the contact_map for the Interaction Prediction Module
        # contact_map_flattened = contact_map.view(contact_map.size(0), -1)  # [batch_size, 242 * 10]

        # Interaction Prediction
        interaction_prob = self.interaction_prediction(contact_map)

        return interaction_prob

# 计算指标
def calculate_metrics(y_true, y_pred):
    TP = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))
    TN = sum((y_true[i] == 0 and y_pred[i] == 0) for i in range(len(y_true)))
    FP = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))
    FN = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true)))

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    sensitivity = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    F1_score = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-10)
    return accuracy, sensitivity, precision, specificity, F1_score, mcc

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim_protein = protease_tensors[0].shape[1]  # 假设所有嵌入向量的维度相同
input_dim_peptide = peptidase_tensors[0].shape[1]
d = 128  # 投影维度
model = ProteinInteractionModel(input_dim_protein, input_dim_peptide, d).to(device)
criterion = nn.BCELoss()  # 二分类问题使用二元交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# 早停相关参数
best_auc = 0
patience = 20
no_improvement_epochs = 0

# 训练模型
def train_model(model, device, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (peptidase, protease, labels) in enumerate(train_loader):

        peptidase, protease, labels = peptidase.to(device), protease.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(peptidase, protease)
        loss = criterion(output, labels.float().unsqueeze(1))
        loss.backward()
        optimizer.step()

        if batch_idx % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# 验证模型
def validate_model(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    allPredictions = []
    allTargets = []

    with torch.no_grad():
        for peptidase, protease, labels in val_loader:
            peptidase, protease, labels = peptidase.to(device), protease.to(device), labels.to(device)
            output = model(peptidase, protease)
            val_loss += criterion(output, labels.float().unsqueeze(1)).item()
            allPredictions.extend(output.cpu().numpy())
            allTargets.extend(labels.cpu().numpy())

    binaryPredictions = [1 if prob > 0.5 else 0 for prob in allPredictions]
    accuracy, sensitivity, precision, specificity, F1_score, mcc = calculate_metrics(allTargets, binaryPredictions)
    val_loss /= len(val_loader)
    val_auc = roc_auc_score(allTargets, allPredictions)

    return val_loss, val_auc, accuracy, sensitivity, precision, specificity, F1_score, mcc

# 测试模型
def test_model(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    allPredictions = []
    allTargets = []

    with torch.no_grad():
        for peptidase, protease, labels in test_loader:
            peptidase, protease, labels = peptidase.to(device), protease.to(device), labels.to(device)
            output = model(peptidase, protease)
            test_loss += criterion(output, labels.float().unsqueeze(1)).item()
            allPredictions.extend(output.cpu().numpy())
            allTargets.extend(labels.cpu().numpy())

    binaryPredictions = [1 if prob > 0.5 else 0 for prob in allPredictions]
    accuracy, sensitivity, precision, specificity, F1_score, mcc = calculate_metrics(allTargets, binaryPredictions)
    test_loss /= len(test_loader)
    test_auc = roc_auc_score(allTargets, allPredictions)

    return test_loss, test_auc, accuracy, sensitivity, precision, specificity, F1_score, mcc

# 训练和验证
num_epochs = 100
for epoch in range(num_epochs):
    train_model(model, device, train_loader, optimizer, criterion)
    val_loss, val_auc, val_accuracy, val_sensitivity, val_precision, val_specificity, val_F1_score, val_mcc = validate_model(model, device, val_loader, criterion)
    print(f'====================================================\n Validation Loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}, Validation accuracy: {val_accuracy:.4f}, Validation sensitivity: {val_sensitivity:.4f}, Validation precision: {val_precision:.4f}, Validation specificity: {val_specificity:.4f}, Validation F1_score: {val_F1_score:.4f}, Validation mcc: {val_mcc:.4f} \n ====================================================')

    # 更新学习率
    scheduler.step(val_loss)

    # 保存最佳模型
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), 'best_model.pth')
        no_improvement_epochs = 0
    else:
        no_improvement_epochs += 1

    # 早停机制
    if no_improvement_epochs >= patience:
        print(f'Early stopping at epoch {epoch + 1}')
        break

# 测试模型
model.load_state_dict(torch.load('best_model.pth'))
test_loss, test_auc, test_accuracy, test_sensitivity, test_precision, test_specificity, test_F1_score, test_mcc = test_model(model, device, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Test accuracy: {test_accuracy:.4f}, Test sensitivity: {test_sensitivity:.4f}, Test precision: {test_precision:.4f}, Test specificity: {test_specificity:.4f}, Test F1_score: {test_F1_score:.4f}, Test mcc: {test_mcc:.4f}')
