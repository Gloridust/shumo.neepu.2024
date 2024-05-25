# A 题：园区微电网风光储协调优化配置 

print(
    """
    ##### 问题1：各园区独立运营储能配置方案及其经济性分析 #####
    """
)

import pandas as pd

# 读取附件1：各园区典型日负荷数据
load_data = pd.read_excel("./mnt/data/附件1：各园区典型日负荷数据.xlsx")
print("各园区典型日负荷数据：")
print(load_data.head())

# 读取附件2：各园区典型日风光发电数据
generation_data = pd.read_excel("./mnt/data/附件2：各园区典型日风光发电数据.xlsx")
print("\n各园区典型日风光发电数据：")
print(generation_data)

print(
    """
###############################
# 计算各园区的购电量和弃风弃光电量
###############################
"""
)

import numpy as np

# 参数设置
P_pv_A = 750  # 园区A光伏装机容量 (kW)
P_w_B = 1000  # 园区B风电装机容量 (kW)
P_pv_C = 600  # 园区C光伏装机容量 (kW)
P_w_C = 500  # 园区C风电装机容量 (kW)

# 读取负荷数据
load_data["时间（h）"] = pd.to_datetime(load_data["时间（h）"], format="%H:%M:%S")
load_data.set_index("时间（h）", inplace=True)

# 读取发电数据
generation_data["时间（h）"] = pd.to_datetime(
    generation_data["时间（h）"], format="%H:%M:%S"
)
generation_data.set_index("时间（h）", inplace=True)

# 计算每个园区的光伏和风电实际出力 (kW)
generation_data["园区A 光伏出力(kW)"] = (
    generation_data["园区A 光伏出力（p.u.）"] * P_pv_A
)
generation_data["园区B风电出力(kW)"] = generation_data["园区B风电出力（p.u.）"] * P_w_B
generation_data["园区C 光伏出力(kW)"] = (
    generation_data["园区C 光伏出力（p.u.）"] * P_pv_C
)
generation_data["园区C 风电出力(kW)"] = (
    generation_data["园区C 风电出力（p.u.）"] * P_w_C
)

# 计算每个园区的总出力
generation_data["园区A 总出力(kW)"] = generation_data["园区A 光伏出力(kW)"]
generation_data["园区B 总出力(kW)"] = generation_data["园区B风电出力(kW)"]
generation_data["园区C 总出力(kW)"] = (
    generation_data["园区C 光伏出力(kW)"] + generation_data["园区C 风电出力(kW)"]
)

# 合并负荷和发电数据
data = load_data.join(
    generation_data[["园区A 总出力(kW)", "园区B 总出力(kW)", "园区C 总出力(kW)"]]
)

# 计算购电量和弃风弃光电量
data["园区A 购电量(kW)"] = np.maximum(
    data["园区A负荷(kW)"] - data["园区A 总出力(kW)"], 0
)
data["园区B 购电量(kW)"] = np.maximum(
    data["园区B负荷(kW)"] - data["园区B 总出力(kW)"], 0
)
data["园区C 购电量(kW)"] = np.maximum(
    data["园区C负荷(kW)"] - data["园区C 总出力(kW)"], 0
)

data["园区A 弃光电量(kW)"] = np.maximum(
    data["园区A 总出力(kW)"] - data["园区A负荷(kW)"], 0
)
data["园区B 弃风电量(kW)"] = np.maximum(
    data["园区B 总出力(kW)"] - data["园区B负荷(kW)"], 0
)
data["园区C 弃光弃风电量(kW)"] = np.maximum(
    data["园区C 总出力(kW)"] - data["园区C负荷(kW)"], 0
)

# 计算总购电量和弃风弃光电量
total_purchase_A = data["园区A 购电量(kW)"].sum()
total_purchase_B = data["园区B 购电量(kW)"].sum()
total_purchase_C = data["园区C 购电量(kW)"].sum()

total_waste_A = data["园区A 弃光电量(kW)"].sum()
total_waste_B = data["园区B 弃风电量(kW)"].sum()
total_waste_C = data["园区C 弃光弃风电量(kW)"].sum()

# 计算总供电成本和单位电量平均供电成本
purchase_cost_A = total_purchase_A * 1  # 园区A的购电成本
purchase_cost_B = total_purchase_B * 1  # 园区B的购电成本
purchase_cost_C = total_purchase_C * 1  # 园区C的购电成本

average_cost_A = purchase_cost_A / data["园区A负荷(kW)"].sum()
average_cost_B = purchase_cost_B / data["园区B负荷(kW)"].sum()
average_cost_C = purchase_cost_C / data["园区C负荷(kW)"].sum()

# 输出结果
result = {
    "园区A": {
        "总购电量(kWh)": total_purchase_A,
        "弃光电量(kWh)": total_waste_A,
        "总供电成本(元)": purchase_cost_A,
        "单位电量平均供电成本(元/kWh)": average_cost_A,
    },
    "园区B": {
        "总购电量(kWh)": total_purchase_B,
        "弃风电量(kWh)": total_waste_B,
        "总供电成本(元)": purchase_cost_B,
        "单位电量平均供电成本(元/kWh)": average_cost_B,
    },
    "园区C": {
        "总购电量(kWh)": total_purchase_C,
        "弃光弃风电量(kWh)": total_waste_C,
        "总供电成本(元)": purchase_cost_C,
        "单位电量平均供电成本(元/kWh)": average_cost_C,
    },
}

for key, value in result.items():
    print(f"\n{key} 结果：")
    for sub_key, sub_value in value.items():
        print(f"{sub_key}: {sub_value:.2f}")

print(
    """
############################
# 配置50kW/100kWh储能后的分析
############################
"""
)

# 储能参数设置
storage_power = 50  # 储能功率 (kW)
storage_capacity = 100  # 储能容量 (kWh)
charge_efficiency = 0.95
discharge_efficiency = 0.95
SOC_min = 0.10
SOC_max = 0.90
SOC_initial = 0.50 * storage_capacity

# 初始化储能SOC
data["SOC_A"] = SOC_initial
data["SOC_B"] = SOC_initial
data["SOC_C"] = SOC_initial

# 储能充放电策略
for index, row in data.iterrows():
    # 园区A储能策略
    if row["园区A负荷(kW)"] > row["园区A 总出力(kW)"]:  # 负荷大于发电，放电
        discharge = min(
            (row["园区A负荷(kW)"] - row["园区A 总出力(kW)"]) / discharge_efficiency,
            storage_power,
        )
        actual_discharge = min(discharge, (row["SOC_A"] - storage_capacity * SOC_min))
        data.at[index, "SOC_A"] -= actual_discharge
        data.at[index, "园区A 购电量(kW)"] = max(
            row["园区A负荷(kW)"]
            - row["园区A 总出力(kW)"]
            - actual_discharge * discharge_efficiency,
            0,
        )
    else:  # 负荷小于发电，充电
        charge = min(
            (row["园区A 总出力(kW)"] - row["园区A负荷(kW)"]) * charge_efficiency,
            storage_power,
        )
        actual_charge = min(charge, (storage_capacity * SOC_max - row["SOC_A"]))
        data.at[index, "SOC_A"] += actual_charge
        data.at[index, "园区A 弃光电量(kW)"] = max(
            row["园区A 总出力(kW)"]
            - row["园区A负荷(kW)"]
            - actual_charge / charge_efficiency,
            0,
        )

    # 园区B储能策略
    if row["园区B负荷(kW)"] > row["园区B 总出力(kW)"]:  # 负荷大于发电，放电
        discharge = min(
            (row["园区B负荷(kW)"] - row["园区B 总出力(kW)"]) / discharge_efficiency,
            storage_power,
        )
        actual_discharge = min(discharge, (row["SOC_B"] - storage_capacity * SOC_min))
        data.at[index, "SOC_B"] -= actual_discharge
        data.at[index, "园区B 购电量(kW)"] = max(
            row["园区B负荷(kW)"]
            - row["园区B 总出力(kW)"]
            - actual_discharge * discharge_efficiency,
            0,
        )
    else:  # 负荷小于发电，充电
        charge = min(
            (row["园区B 总出力(kW)"] - row["园区B负荷(kW)"]) * charge_efficiency,
            storage_power,
        )
        actual_charge = min(charge, (storage_capacity * SOC_max - row["SOC_B"]))
        data.at[index, "SOC_B"] += actual_charge
        data.at[index, "园区B 弃风电量(kW)"] = max(
            row["园区B 总出力(kW)"]
            - row["园区B负荷(kW)"]
            - actual_charge / charge_efficiency,
            0,
        )

    # 园区C储能策略
    if row["园区C负荷(kW)"] > row["园区C 总出力(kW)"]:  # 负荷大于发电，放电
        discharge = min(
            (row["园区C负荷(kW)"] - row["园区C 总出力(kW)"]) / discharge_efficiency,
            storage_power,
        )
        actual_discharge = min(discharge, (row["SOC_C"] - storage_capacity * SOC_min))
        data.at[index, "SOC_C"] -= actual_discharge
        data.at[index, "园区C 购电量(kW)"] = max(
            row["园区C负荷(kW)"]
            - row["园区C 总出力(kW)"]
            - actual_discharge * discharge_efficiency,
            0,
        )
    else:  # 负荷小于发电，充电
        charge = min(
            (row["园区C 总出力(kW)"] - row["园区C负荷(kW)"]) * charge_efficiency,
            storage_power,
        )
        actual_charge = min(charge, (storage_capacity * SOC_max - row["SOC_C"]))
        data.at[index, "SOC_C"] += actual_charge
        data.at[index, "园区C 弃光弃风电量(kW)"] = max(
            row["园区C 总出力(kW)"]
            - row["园区C负荷(kW)"]
            - actual_charge / charge_efficiency,
            0,
        )

# 计算配置储能后的购电量和弃电量
total_purchase_A_storage = data["园区A 购电量(kW)"].sum()
total_purchase_B_storage = data["园区B 购电量(kW)"].sum()
total_purchase_C_storage = data["园区C 购电量(kW)"].sum()

total_waste_A_storage = data["园区A 弃光电量(kW)"].sum()
total_waste_B_storage = data["园区B 弃风电量(kW)"].sum()
total_waste_C_storage = data["园区C 弃光弃风电量(kW)"].sum()

# 计算配置储能后的总供电成本和单位电量平均供电成本
purchase_cost_A_storage = total_purchase_A_storage * 1  # 园区A的购电成本
purchase_cost_B_storage = total_purchase_B_storage * 1  # 园区B的购电成本
purchase_cost_C_storage = total_purchase_C_storage * 1  # 园区C的购电成本

average_cost_A_storage = purchase_cost_A_storage / data["园区A负荷(kW)"].sum()
average_cost_B_storage = purchase_cost_B_storage / data["园区B负荷(kW)"].sum()
average_cost_C_storage = purchase_cost_C_storage / data["园区C负荷(kW)"].sum()

# 输出结果
result_storage = {
    "园区A": {
        "总购电量(kWh)": total_purchase_A_storage,
        "弃光电量(kWh)": total_waste_A_storage,
        "总供电成本(元)": purchase_cost_A_storage,
        "单位电量平均供电成本(元/kWh)": average_cost_A_storage,
    },
    "园区B": {
        "总购电量(kWh)": total_purchase_B_storage,
        "弃风电量(kWh)": total_waste_B_storage,
        "总供电成本(元)": purchase_cost_B_storage,
        "单位电量平均供电成本(元/kWh)": average_cost_B_storage,
    },
    "园区C": {
        "总购电量(kWh)": total_purchase_C_storage,
        "弃光弃风电量(kWh)": total_waste_C_storage,
        "总供电成本(元)": purchase_cost_C_storage,
        "单位电量平均供电成本(元/kWh)": average_cost_C_storage,
    },
}

for key, value in result_storage.items():
    print(f"\n{key} 结果（配置储能后）：")
    for sub_key, sub_value in value.items():
        print(f"{sub_key}: {sub_value:.2f}")
print()

print(
    """
#######################################
# 判断50kW/100kWh的方案是否最优
# 并制定各园区最优的储能功率和容量配置方案
#######################################
"""
)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# 准备数据
# 我们需要生成一些特征和目标数据，用于机器学习模型的训练
# 特征包括：负荷、发电、储能参数等
# 目标包括：总供电成本、弃电量等

# 生成特征数据
features = data[
    [
        "园区A负荷(kW)",
        "园区A 总出力(kW)",
        "园区B负荷(kW)",
        "园区B 总出力(kW)",
        "园区C负荷(kW)",
        "园区C 总出力(kW)",
    ]
]

# 生成目标数据
# 这里我们假设目标数据是购电量、弃电量和总供电成本的加权和
data["总供电成本"] = (
    data["园区A 购电量(kW)"] * 1
    + data["园区B 购电量(kW)"] * 1
    + data["园区C 购电量(kW)"] * 1
)
data["弃电量"] = (
    data["园区A 弃光电量(kW)"]
    + data["园区B 弃风电量(kW)"]
    + data["园区C 弃光弃风电量(kW)"]
)
data["目标"] = data["总供电成本"] + data["弃电量"] * 0.1

# 拆分训练集和测试集
X = features.values
y = data["目标"].values

# 创建随机森林回归模型
rf = RandomForestRegressor()

# 定义参数网格用于网格搜索
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
}

# 网格搜索
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
)
grid_search.fit(X, y)

# 最优参数
best_params = grid_search.best_params_
best_rf = grid_search.best_estimator_

# 输出最优参数
print("最优参数：", best_params)
print()

# 使用最优模型进行预测
y_pred = best_rf.predict(X)

# 计算最优储能配置方案
optimal_storage_power_A = np.mean(y_pred[: len(data["园区A负荷(kW)"])]) * storage_power
optimal_storage_capacity_A = (
    np.mean(y_pred[: len(data["园区A负荷(kW)"])]) * storage_capacity
)
optimal_storage_power_B = np.mean(y_pred[: len(data["园区B负荷(kW)"])]) * storage_power
optimal_storage_capacity_B = (
    np.mean(y_pred[: len(data["园区B负荷(kW)"])]) * storage_capacity
)
optimal_storage_power_C = np.mean(y_pred[: len(data["园区C负荷(kW)"])]) * storage_power
optimal_storage_capacity_C = (
    np.mean(y_pred[: len(data["园区C负荷(kW)"])]) * storage_capacity
)

# 输出结果
print(f"最优储能配置方案：")
print(f"园区A储能功率: {optimal_storage_power_A:.2f} kW")
print(f"园区A储能容量: {optimal_storage_capacity_A:.2f} kWh")
print(f"园区B储能功率: {optimal_storage_power_B:.2f} kW")
print(f"园区B储能容量: {optimal_storage_capacity_B:.2f} kWh")
print(f"园区C储能功率: {optimal_storage_power_C:.2f} kW")
print(f"园区C储能容量: {optimal_storage_capacity_C:.2f} kWh")
print()

print(
    """
    ##### 问题2：联合园区储能配置方案及其经济性分析 #####
    """
)

print(
    """
#####################################################################
# 若未配置储能，分析联合园区运行经济性
# 包括：联合园区的总购电量、总弃风弃光电量、总供电成本和单位电量平均供电成本
#####################################################################
"""
)

import pandas as pd

# 读取负荷数据
load_data = pd.read_excel("./mnt/data/附件1：各园区典型日负荷数据.xlsx")
generation_data = pd.read_excel("./mnt/data/附件2：各园区典型日风光发电数据.xlsx")

# 将时间转换为datetime类型
load_data["时间（h）"] = pd.to_datetime(load_data["时间（h）"], format="%H:%M:%S")
generation_data["时间（h）"] = pd.to_datetime(
    generation_data["时间（h）"], format="%H:%M:%S"
)

# 设置时间为索引
load_data.set_index("时间（h）", inplace=True)
generation_data.set_index("时间（h）", inplace=True)

# 计算各园区的总负荷和总发电
load_data["总负荷(kW)"] = (
    load_data["园区A负荷(kW)"] + load_data["园区B负荷(kW)"] + load_data["园区C负荷(kW)"]
)
generation_data["总光伏出力(kW)"] = (
    generation_data["园区A 光伏出力（p.u.）"] * 750
    + generation_data["园区C 光伏出力（p.u.）"] * 600
)
generation_data["总风电出力(kW)"] = (
    generation_data["园区B风电出力（p.u.）"] * 1000
    + generation_data["园区C 风电出力（p.u.）"] * 500
)
generation_data["总发电(kW)"] = (
    generation_data["总光伏出力(kW)"] + generation_data["总风电出力(kW)"]
)

# 合并负荷和发电数据
combined_data = pd.concat([load_data, generation_data], axis=1)

# 计算购电量和弃电量
combined_data["购电量(kW)"] = np.maximum(
    combined_data["总负荷(kW)"] - combined_data["总发电(kW)"], 0
)
combined_data["弃电量(kW)"] = np.maximum(
    combined_data["总发电(kW)"] - combined_data["总负荷(kW)"], 0
)

# 计算总购电量、总弃电量、总供电成本和单位电量平均供电成本
total_purchase = combined_data["购电量(kW)"].sum()
total_waste = combined_data["弃电量(kW)"].sum()
total_cost = total_purchase * 1  # 购电成本为1元/kWh
average_cost = total_cost / combined_data["总负荷(kW)"].sum()

# 输出结果
print(f"联合园区未配置储能时的经济性分析：")
print(f"总购电量(kWh): {total_purchase:.2f}")
print(f"总弃电量(kWh): {total_waste:.2f}")
print(f"总供电成本(元): {total_cost:.2f}")
print(f"单位电量平均供电成本(元/kWh): {average_cost:.2f}")
print()

print(
    """
#####################################
# 假设风光荷功率波动特性保持上述条件不变
# 制定联合园区的总储能最优配置方案
#####################################
"""
)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

# 加载数据
load_data = pd.read_excel('./mnt/data/附件1：各园区典型日负荷数据.xlsx')
generation_data = pd.read_excel('./mnt/data/附件2：各园区典型日风光发电数据.xlsx')

# 处理数据
load_data['时间（h）'] = pd.to_datetime(load_data['时间（h）'], format='%H:%M:%S')
generation_data['时间（h）'] = pd.to_datetime(generation_data['时间（h）'], format='%H:%M:%S')
load_data.set_index('时间（h）', inplace=True)
generation_data.set_index('时间（h）', inplace=True)

# 合并负荷数据
load_data['总负荷(kW)'] = load_data['园区A负荷(kW)'] + load_data['园区B负荷(kW)'] + load_data['园区C负荷(kW)']

# 合并发电数据
generation_data['总光伏出力(kW)'] = generation_data['园区A 光伏出力（p.u.）'] * 750 + generation_data['园区C 光伏出力（p.u.）'] * 600
generation_data['总风电出力(kW)'] = generation_data['园区B风电出力（p.u.）'] * 1000 + generation_data['园区C 风电出力（p.u.）'] * 500
generation_data['总发电(kW)'] = generation_data['总光伏出力(kW)'] + generation_data['总风电出力(kW)']

# 合并数据
combined_data = pd.concat([load_data, generation_data], axis=1)

# 计算购电量和弃电量
combined_data['购电量(kW)'] = np.maximum(combined_data['总负荷(kW)'] - combined_data['总发电(kW)'], 0)
combined_data['弃电量(kW)'] = np.maximum(combined_data['总发电(kW)'] - combined_data['总负荷(kW)'], 0)

# 储能参数设置
charge_efficiency = 0.95
discharge_efficiency = 0.95
SOC_min = 0.10
SOC_max = 0.90

# 生成模拟数据
def simulate_storage_operation(storage_power, storage_capacity):
    combined_data['SOC'] = 0.50 * storage_capacity  # 初始SOC为50%的storage_capacity
    combined_data['购电量_储能后(kW)'] = combined_data['购电量(kW)']
    combined_data['弃电量_储能后(kW)'] = combined_data['弃电量(kW)']

    for index, row in combined_data.iterrows():
        # 如果负荷大于发电，放电
        if row['总负荷(kW)'] > row['总发电(kW)']:
            discharge = min((row['总负荷(kW)'] - row['总发电(kW)']) / discharge_efficiency, storage_power)
            actual_discharge = min(discharge, (row['SOC'] - storage_capacity * SOC_min))
            combined_data.at[index, 'SOC'] -= actual_discharge
            combined_data.at[index, '购电量_储能后(kW)'] = max(row['总负荷(kW)'] - row['总发电(kW)'] - actual_discharge * discharge_efficiency, 0)
        # 如果负荷小于发电，充电
        else:
            charge = min((row['总发电(kW)'] - row['总负荷(kW)']) * charge_efficiency, storage_power)
            actual_charge = min(charge, (storage_capacity * SOC_max - row['SOC']))
            combined_data.at[index, 'SOC'] += actual_charge
            combined_data.at[index, '弃电量_储能后(kW)'] = max(row['总发电(kW)'] - row['总负荷(kW)'] - actual_charge / charge_efficiency, 0)

    total_purchase_storage = combined_data['购电量_储能后(kW)'].sum()
    total_waste_storage = combined_data['弃电量_储能后(kW)'].sum()
    total_cost_storage = total_purchase_storage * 1  # 购电成本为1元/kWh
    average_cost_storage = total_cost_storage / combined_data['总负荷(kW)'].sum()
    
    return total_purchase_storage, total_waste_storage, total_cost_storage, average_cost_storage

# 生成特征数据和目标数据
storage_powers = np.linspace(0, 200, 20)  # 储能功率从0到200kW
storage_capacities = np.linspace(0, 500, 20)  # 储能容量从0到500kWh
X = []
y = []

for storage_power in storage_powers:
    for storage_capacity in storage_capacities:
        total_purchase_storage, total_waste_storage, total_cost_storage, average_cost_storage = simulate_storage_operation(storage_power, storage_capacity)
        X.append([storage_power, storage_capacity])
        y.append(total_cost_storage)

X = np.array(X)
y = np.array(y)

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林回归模型
rf = RandomForestRegressor()

# 定义参数网格用于网格搜索
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# 网格搜索
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 最优参数
best_params = grid_search.best_params_
best_rf = grid_search.best_estimator_

# 使用最优模型进行预测
y_pred = best_rf.predict(X_test)

# 寻找最优储能配置
min_cost_index = np.argmin(y_pred)
optimal_storage_power = X_test[min_cost_index, 0]
optimal_storage_capacity = X_test[min_cost_index, 1]

# 输出结果
print("最优参数：", best_params)
print(f"最优储能配置方案：")
print(f"储能功率: {optimal_storage_power:.2f} kW")
print(f"储能容量: {optimal_storage_capacity:.2f} kWh")

# # 线性规划

# from scipy.optimize import linprog

# # 储能参数设置
# storage_cost_power = 800  # 储能功率单价 (元/kW)
# storage_cost_capacity = 1800  # 储能容量单价 (元/kWh)
# years = 10  # 运行寿命 (年)

# # 目标函数系数 (总成本)
# c = [
#     storage_cost_power * years,  # 联合园区储能功率成本
#     storage_cost_capacity * years,  # 联合园区储能容量成本
# ]

# # 约束条件 (购电量减少)
# # 计算配置储能后的购电量和弃电量
# # 我们假设储能效率为95%，SOC范围为10%-90%
# charge_efficiency = 0.95
# discharge_efficiency = 0.95
# SOC_min = 0.10
# SOC_max = 0.90

# # 初始化数据
# combined_data["SOC"] = 0.50 * 100  # 假设初始SOC为50%的100kWh
# combined_data["购电量_储能后(kW)"] = combined_data["购电量(kW)"]
# combined_data["弃电量_储能后(kW)"] = combined_data["弃电量(kW)"]

# # 储能充放电策略模拟
# for index, row in combined_data.iterrows():
#     # 如果负荷大于发电，放电
#     if row["总负荷(kW)"] > row["总发电(kW)"]:
#         discharge = min(
#             (row["总负荷(kW)"] - row["总发电(kW)"]) / discharge_efficiency, 50
#         )
#         actual_discharge = min(discharge, (row["SOC"] - 100 * SOC_min))
#         combined_data.at[index, "SOC"] -= actual_discharge
#         combined_data.at[index, "购电量_储能后(kW)"] = max(
#             row["总负荷(kW)"]
#             - row["总发电(kW)"]
#             - actual_discharge * discharge_efficiency,
#             0,
#         )
#     # 如果负荷小于发电，充电
#     else:
#         charge = min((row["总发电(kW)"] - row["总负荷(kW)"]) * charge_efficiency, 50)
#         actual_charge = min(charge, (100 * SOC_max - row["SOC"]))
#         combined_data.at[index, "SOC"] += actual_charge
#         combined_data.at[index, "弃电量_储能后(kW)"] = max(
#             row["总发电(kW)"] - row["总负荷(kW)"] - actual_charge / charge_efficiency, 0
#         )

# # 计算配置储能后的购电量和弃电量
# total_purchase_storage = combined_data["购电量_储能后(kW)"].sum()
# total_waste_storage = combined_data["弃电量_储能后(kW)"].sum()
# total_cost_storage = total_purchase_storage * 1  # 购电成本为1元/kWh
# average_cost_storage = total_cost_storage / combined_data["总负荷(kW)"].sum()

# # 输出结果
# print(f"联合园区配置储能后的经济性分析：")
# print(f"总购电量(kWh): {total_purchase_storage:.2f}")
# print(f"总弃电量(kWh): {total_waste_storage:.2f}")
# print(f"总供电成本(元): {total_cost_storage:.2f}")
# print(f"单位电量平均供电成本(元/kWh): {average_cost_storage:.2f}")
# print()

print(
    """
######################
# 与各园区独立运营相比
# 联合运营的经济收益分析
######################
"""
)

# 各园区独立运营总量
total_purchase_independent = 4874.12 + 2432.30 + 2699.39
total_waste_independent = 951.20 + 897.50 + 1128.02
total_cost_independent = 4874.12 + 2432.30 + 2699.39
average_cost_independent = total_cost_independent / (
    data["园区A负荷(kW)"].sum()
    + data["园区B负荷(kW)"].sum()
    + data["园区C负荷(kW)"].sum()
)

# 联合运营总量
total_purchase_joint = total_purchase_storage
total_waste_joint = total_waste_storage
total_cost_joint = total_cost_storage
average_cost_joint = average_cost_storage

# 计算经济收益变化
savings_purchase = total_purchase_independent - total_purchase_joint
savings_waste = total_waste_independent - total_waste_joint
savings_cost = total_cost_independent - total_cost_joint
savings_average_cost = average_cost_independent - average_cost_joint

# 输出结果
print(f"各园区独立运营 vs 联合运营的经济性对比：")
print(f"总购电量节省(kWh): {savings_purchase:.2f}")
print(f"总弃电量减少(kWh): {savings_waste:.2f}")
print(f"总供电成本节省(元): {savings_cost:.2f}")
print(f"单位电量平均供电成本减少(元/kWh): {savings_average_cost:.2f}")
print()

print(
    """
    ##### 问题3：园区风、光、储能的协调配置方案及其经济性分析 #####
    """
)

print(
    """
#############################
# 分别按各园区独立运营、联合运营
# 制定风光储协调配置方案
#############################
"""
)
from scipy.optimize import linprog

# 独立运营配置方案

# 风光电源配置成本
cost_wind = 3000  # 元/kW
cost_pv = 2500  # 元/kW

# 储能配置成本
cost_storage_power = 800  # 元/kW
cost_storage_capacity = 1800  # 元/kWh
years = 5  # 投资回报期

# 最大负荷增长50%
max_load_A = load_data['园区A负荷(kW)'].max() * 1.5
max_load_B = load_data['园区B负荷(kW)'].max() * 1.5
max_load_C = load_data['园区C负荷(kW)'].max() * 1.5

# 定义目标函数 (总成本)
c = [
    cost_wind,  # 园区A风电成本
    cost_pv,  # 园区A光伏成本
    cost_storage_power * years + cost_storage_capacity * years,  # 园区A储能成本
    cost_wind,  # 园区B风电成本
    cost_pv,  # 园区B光伏成本
    cost_storage_power * years + cost_storage_capacity * years,  # 园区B储能成本
    cost_wind,  # 园区C风电成本
    cost_pv,  # 园区C光伏成本
    cost_storage_power * years + cost_storage_capacity * years   # 园区C储能成本
]

# 定义约束条件 (满足负荷需求)
A_eq = [
    [1, 1, 1, 0, 0, 0, 0, 0, 0],  # 园区A风电+光伏+储能>=负荷
    [0, 0, 0, 1, 1, 1, 0, 0, 0],  # 园区B风电+光伏+储能>=负荷
    [0, 0, 0, 0, 0, 0, 1, 1, 1]   # 园区C风电+光伏+储能>=负荷
]
b_eq = [
    max_load_A,  # 园区A负荷
    max_load_B,  # 园区B负荷
    max_load_C   # 园区C负荷
]

# 变量界限 (风电、光伏、储能)
x_bounds = [(0, None)] * 9

# 使用线性规划求解最优配置
result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=x_bounds, method='highs')

# 提取结果
optimal_wind_A, optimal_pv_A, optimal_storage_A, optimal_wind_B, optimal_pv_B, optimal_storage_B, optimal_wind_C, optimal_pv_C, optimal_storage_C = result.x

# 输出结果
print(f"园区A独立运营配置方案：")
print(f"所需风电配置(kW): {optimal_wind_A:.2f}")
print(f"所需光伏配置(kW): {optimal_pv_A:.2f}")
print(f"所需储能配置(kW): {optimal_storage_A:.2f}")
print(f"总配置成本(元): {optimal_wind_A * cost_wind + optimal_pv_A * cost_pv + optimal_storage_A * (cost_storage_power * years + cost_storage_capacity * years):.2f}")

print(f"\n园区B独立运营配置方案：")
print(f"所需风电配置(kW): {optimal_wind_B:.2f}")
print(f"所需光伏配置(kW): {optimal_pv_B:.2f}")
print(f"所需储能配置(kW): {optimal_storage_B:.2f}")
print(f"总配置成本(元): {optimal_wind_B * cost_wind + optimal_pv_B * cost_pv + optimal_storage_B * (cost_storage_power * years + cost_storage_capacity * years):.2f}")

print(f"\n园区C独立运营配置方案：")
print(f"所需风电配置(kW): {optimal_wind_C:.2f}")
print(f"所需光伏配置(kW): {optimal_pv_C:.2f}")
print(f"所需储能配置(kW): {optimal_storage_C:.2f}")
print(f"总配置成本(元): {optimal_wind_C * cost_wind + optimal_pv_C * cost_pv + optimal_storage_C * (cost_storage_power * years + cost_storage_capacity * years):.2f}")

# 联合运营配置方案

# 总负荷增长50%
total_max_load = max_load_A + max_load_B + max_load_C

# 定义目标函数 (总成本)
c_joint = [
    cost_wind,  # 联合园区风电成本
    cost_pv,  # 联合园区光伏成本
    cost_storage_power * years + cost_storage_capacity * years  # 联合园区储能成本
]

# 定义约束条件 (满足负荷需求)
A_eq_joint = [
    [1, 1, 1]  # 联合园区风电+光伏+储能>=负荷
]
b_eq_joint = [
    total_max_load  # 联合园区负荷
]

# 变量界限 (风电、光伏、储能)
x_bounds_joint = [(0, None)] * 3

# 使用线性规划求解最优配置
result_joint = linprog(c_joint, A_eq=A_eq_joint, b_eq=b_eq_joint, bounds=x_bounds_joint, method='highs')

# 提取结果
optimal_wind_joint, optimal_pv_joint, optimal_storage_joint = result_joint.x

# 输出结果
print(f"联合运营配置方案：")
print(f"所需风电配置(kW): {optimal_wind_joint:.2f}")
print(f"所需光伏配置(kW): {optimal_pv_joint:.2f}")
print(f"所需储能配置(kW): {optimal_storage_joint:.2f}")
print(f"总配置成本(元): {optimal_wind_joint * cost_wind + optimal_pv_joint * cost_pv + optimal_storage_joint * (cost_storage_power * years + cost_storage_capacity * years):.2f}")
