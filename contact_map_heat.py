import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv(r"D:\desk\PGCN\前置分析\Grad-cam\E107D_D127A_S135F_1_Cam.csv", index_col=0)#.iloc[:, :10].T
print(df)

# 设置每个小方格的期望大小（单位：英寸）
square_size = 1.5  # 你可以根据需要调整这个值

# 根据数据框的行数和列数动态计算画布大小
num_rows, num_cols = df.shape
fig_width = num_cols * square_size  # 宽度 = 列数 * 每个方格的宽度
fig_height = num_rows * square_size  # 高度 = 行数 * 每个方格的高度

# 设置画布大小
plt.figure(figsize=(10, 8))  # 宽度 10 英寸，高度 8 英寸

# 绘制热图
heatmap = sns.heatmap(
    df,
    square=True,  # 使每个小方块为正方形
    cbar=True,  # 不显示颜色条
    annot=False,  # 不显示数值
    xticklabels=False,  # 不显示x轴标签
    yticklabels=False,  # 不显示y轴标签
    cmap="Blues"
)

# 隐藏坐标轴
plt.axis('off')

# 保存热图
plt.savefig(r"D:\desk\PGCN\图\194.png",  pad_inches=0,dpi=2000)

# 显示热图
plt.show()