import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 读取数据
load_data = pd.read_excel('./mnt/data/附件1：各园区典型日负荷数据.xlsx')
generation_data = pd.read_excel('./mnt/data/附件2：各园区典型日风光发电数据.xlsx')
monthly_generation_data = pd.read_excel('./mnt/data/附件3：12个月各园区典型日风光发电数据.xlsx')

# 数据标准化
scaler = MinMaxScaler()

# 负荷数据标准化
load_data_scaled = scaler.fit_transform(load_data.iloc[:, 1:])  # 假设第一列是时间
load_data_scaled = pd.DataFrame(load_data_scaled, columns=load_data.columns[1:])

# 风光发电数据标准化
generation_data_scaled = scaler.fit_transform(generation_data.iloc[:, 1:])  # 假设第一列是时间
generation_data_scaled = pd.DataFrame(generation_data_scaled, columns=generation_data.columns[1:])

# 显示标准化后的数据
print("Scaled Load Data:\n", load_data_scaled.head())
print("Scaled Generation Data:\n", generation_data_scaled.head())

# 储能系统容量计算
def calculate_storage_capacity(load, generation, max_load_increase=0.5):
    # 负荷增长50%
    increased_load = load * (1 + max_load_increase)
    
    # 计算储能需求
    storage_capacity = max(0, (increased_load.sum() - generation.sum()) * 0.9)  # 考虑SOC允许范围和充/放电效率
    return storage_capacity

# 使用原始数据计算各园区储能需求
storage_capacity_A = calculate_storage_capacity(load_data['园区A负荷(kW)'], generation_data['园区A 光伏出力（p.u.）'] * 750)
storage_capacity_B = calculate_storage_capacity(load_data['园区B负荷(kW)'], generation_data['园区B风电出力（p.u.）'] * 1000)
storage_capacity_C = calculate_storage_capacity(load_data['园区C负荷(kW)'], (generation_data['园区C 光伏出力（p.u.）'] * 600 + generation_data['园区C 风电出力（p.u.）'] * 500))

print(f"储能系统容量（园区A）：{storage_capacity_A} kWh")
print(f"储能系统容量（园区B）：{storage_capacity_B} kWh")
print(f"储能系统容量（园区C）：{storage_capacity_C} kWh")

# 经济性分析
def economic_analysis(storage_capacity, power_price=800, energy_price=1800, years=5):
    power_cost = storage_capacity * power_price
    energy_cost = storage_capacity * energy_price
    total_cost = power_cost + energy_cost
    annual_savings = storage_capacity * 365 * (1 - 0.4)  # 以节约的电费计算
    return total_cost, annual_savings

# 计算经济性
total_cost_A, annual_savings_A = economic_analysis(storage_capacity_A)
total_cost_B, annual_savings_B = economic_analysis(storage_capacity_B)
total_cost_C, annual_savings_C = economic_analysis(storage_capacity_C)

print(f"总投资成本（园区A）：{total_cost_A} 元")
print(f"年度节约成本（园区A）：{annual_savings_A} 元")
print(f"总投资成本（园区B）：{total_cost_B} 元")
print(f"年度节约成本（园区B）：{annual_savings_B} 元")
print(f"总投资成本（园区C）：{total_cost_C} 元")
print(f"年度节约成本（园区C）：{annual_savings_C} 元")
