import pandas as pd

# 读取全年12个月的典型日风光发电数据
monthly_generation_data = pd.read_excel('./mnt/data/附件3：12个月各园区典型日风光发电数据.xlsx')

# 查看数据结构
print("Monthly Generation Data:\n", monthly_generation_data.head())

from sklearn.preprocessing import MinMaxScaler

# 提取特征（去掉时间列）
features = monthly_generation_data.iloc[:, 1:]  

# 数据标准化
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# 显示标准化后的数据
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
print("Scaled Features:\n", features_scaled_df.head())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 生成虚拟标签数据，这里假设我们需要预测的储能容量
# 假设使用每个月的总发电量的平均值作为储能需求的虚拟标签
labels = features_scaled_df.mean(axis=1)  

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# 训练随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测和评估模型
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 使用训练好的模型进行全年储能需求预测
storage_predictions = model.predict(features_scaled)
print("Storage Predictions:\n", storage_predictions)

# 经济性分析函数
def economic_analysis(storage_predictions, power_price=800, energy_price=1800, years=5):
    power_cost = storage_predictions * power_price
    energy_cost = storage_predictions * energy_price
    total_cost = power_cost + energy_cost
    annual_savings = storage_predictions * 365 * (1 - 0.4)  # 以节约的电费计算
    return total_cost, annual_savings

# 计算经济性
total_cost, annual_savings = economic_analysis(storage_predictions)

# 输出经济性分析结果
for i, month in enumerate(monthly_generation_data.columns[1:], start=1):
    print(f"{month} - 总投资成本：{total_cost[i-1]:.2f} 元，年度节约成本：{annual_savings[i-1]:.2f} 元")
