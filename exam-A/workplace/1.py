import pandas as pd

# 读取附件1：各园区典型日负荷数据
load_data = pd.read_excel('./mnt/data/附件1：各园区典型日负荷数据.xlsx')
print("各园区典型日负荷数据：")
print(load_data.head())

# 读取附件2：各园区典型日风光发电数据
generation_data = pd.read_excel('./mnt/data/附件2：各园区典型日风光发电数据.xlsx')
print("\n各园区典型日风光发电数据：")
print(generation_data.head())

###############################
# 计算各园区的购电量和弃风弃光电量
###############################

import numpy as np

# 参数设置
P_pv_A = 750  # 园区A光伏装机容量 (kW)
P_w_B = 1000  # 园区B风电装机容量 (kW)
P_pv_C = 600  # 园区C光伏装机容量 (kW)
P_w_C = 500   # 园区C风电装机容量 (kW)

# 读取负荷数据
load_data['时间（h）'] = pd.to_datetime(load_data['时间（h）'], format='%H:%M:%S')
load_data.set_index('时间（h）', inplace=True)

# 读取发电数据
generation_data['时间（h）'] = pd.to_datetime(generation_data['时间（h）'], format='%H:%M:%S')
generation_data.set_index('时间（h）', inplace=True)

# 计算每个园区的光伏和风电实际出力 (kW)
generation_data['园区A 光伏出力(kW)'] = generation_data['园区A 光伏出力（p.u.）'] * P_pv_A
generation_data['园区B风电出力(kW)'] = generation_data['园区B风电出力（p.u.）'] * P_w_B
generation_data['园区C 光伏出力(kW)'] = generation_data['园区C 光伏出力（p.u.）'] * P_pv_C
generation_data['园区C 风电出力(kW)'] = generation_data['园区C 风电出力（p.u.）'] * P_w_C

# 计算每个园区的总出力
generation_data['园区A 总出力(kW)'] = generation_data['园区A 光伏出力(kW)']
generation_data['园区B 总出力(kW)'] = generation_data['园区B风电出力(kW)']
generation_data['园区C 总出力(kW)'] = generation_data['园区C 光伏出力(kW)'] + generation_data['园区C 风电出力(kW)']

# 合并负荷和发电数据
data = load_data.join(generation_data[['园区A 总出力(kW)', '园区B 总出力(kW)', '园区C 总出力(kW)']])

# 计算购电量和弃风弃光电量
data['园区A 购电量(kW)'] = np.maximum(data['园区A负荷(kW)'] - data['园区A 总出力(kW)'], 0)
data['园区B 购电量(kW)'] = np.maximum(data['园区B负荷(kW)'] - data['园区B 总出力(kW)'], 0)
data['园区C 购电量(kW)'] = np.maximum(data['园区C负荷(kW)'] - data['园区C 总出力(kW)'], 0)

data['园区A 弃光电量(kW)'] = np.maximum(data['园区A 总出力(kW)'] - data['园区A负荷(kW)'], 0)
data['园区B 弃风电量(kW)'] = np.maximum(data['园区B 总出力(kW)'] - data['园区B负荷(kW)'], 0)
data['园区C 弃光弃风电量(kW)'] = np.maximum(data['园区C 总出力(kW)'] - data['园区C负荷(kW)'], 0)

# 计算总购电量和弃风弃光电量
total_purchase_A = data['园区A 购电量(kW)'].sum()
total_purchase_B = data['园区B 购电量(kW)'].sum()
total_purchase_C = data['园区C 购电量(kW)'].sum()

total_waste_A = data['园区A 弃光电量(kW)'].sum()
total_waste_B = data['园区B 弃风电量(kW)'].sum()
total_waste_C = data['园区C 弃光弃风电量(kW)'].sum()

# 计算总供电成本和单位电量平均供电成本
purchase_cost_A = total_purchase_A * 1  # 园区A的购电成本
purchase_cost_B = total_purchase_B * 1  # 园区B的购电成本
purchase_cost_C = total_purchase_C * 1  # 园区C的购电成本

average_cost_A = purchase_cost_A / data['园区A负荷(kW)'].sum()
average_cost_B = purchase_cost_B / data['园区B负荷(kW)'].sum()
average_cost_C = purchase_cost_C / data['园区C负荷(kW)'].sum()

# 输出结果
result = {
    "园区A": {
        "总购电量(kWh)": total_purchase_A,
        "弃光电量(kWh)": total_waste_A,
        "总供电成本(元)": purchase_cost_A,
        "单位电量平均供电成本(元/kWh)": average_cost_A
    },
    "园区B": {
        "总购电量(kWh)": total_purchase_B,
        "弃风电量(kWh)": total_waste_B,
        "总供电成本(元)": purchase_cost_B,
        "单位电量平均供电成本(元/kWh)": average_cost_B
    },
    "园区C": {
        "总购电量(kWh)": total_purchase_C,
        "弃光弃风电量(kWh)": total_waste_C,
        "总供电成本(元)": purchase_cost_C,
        "单位电量平均供电成本(元/kWh)": average_cost_C
    }
}

for key, value in result.items():
    print(f"\n{key} 结果：")
    for sub_key, sub_value in value.items():
        print(f"{sub_key}: {sub_value:.2f}")

############################
# 配置50kW/100kWh储能后的分析
############################

# 储能参数设置
storage_power = 50  # 储能功率 (kW)
storage_capacity = 100  # 储能容量 (kWh)
charge_efficiency = 0.95
discharge_efficiency = 0.95
SOC_min = 0.10
SOC_max = 0.90
SOC_initial = 0.50 * storage_capacity

# 初始化储能SOC
data['SOC_A'] = SOC_initial
data['SOC_B'] = SOC_initial
data['SOC_C'] = SOC_initial

# 储能充放电策略
for index, row in data.iterrows():
    # 园区A储能策略
    if row['园区A负荷(kW)'] > row['园区A 总出力(kW)']:  # 负荷大于发电，放电
        discharge = min((row['园区A负荷(kW)'] - row['园区A 总出力(kW)']) / discharge_efficiency, storage_power)
        actual_discharge = min(discharge, (row['SOC_A'] - storage_capacity * SOC_min))
        data.at[index, 'SOC_A'] -= actual_discharge
        data.at[index, '园区A 购电量(kW)'] = max(row['园区A负荷(kW)'] - row['园区A 总出力(kW)'] - actual_discharge * discharge_efficiency, 0)
    else:  # 负荷小于发电，充电
        charge = min((row['园区A 总出力(kW)'] - row['园区A负荷(kW)']) * charge_efficiency, storage_power)
        actual_charge = min(charge, (storage_capacity * SOC_max - row['SOC_A']))
        data.at[index, 'SOC_A'] += actual_charge
        data.at[index, '园区A 弃光电量(kW)'] = max(row['园区A 总出力(kW)'] - row['园区A负荷(kW)'] - actual_charge / charge_efficiency, 0)

    # 园区B储能策略
    if row['园区B负荷(kW)'] > row['园区B 总出力(kW)']:  # 负荷大于发电，放电
        discharge = min((row['园区B负荷(kW)'] - row['园区B 总出力(kW)']) / discharge_efficiency, storage_power)
        actual_discharge = min(discharge, (row['SOC_B'] - storage_capacity * SOC_min))
        data.at[index, 'SOC_B'] -= actual_discharge
        data.at[index, '园区B 购电量(kW)'] = max(row['园区B负荷(kW)'] - row['园区B 总出力(kW)'] - actual_discharge * discharge_efficiency, 0)
    else:  # 负荷小于发电，充电
        charge = min((row['园区B 总出力(kW)'] - row['园区B负荷(kW)']) * charge_efficiency, storage_power)
        actual_charge = min(charge, (storage_capacity * SOC_max - row['SOC_B']))
        data.at[index, 'SOC_B'] += actual_charge
        data.at[index, '园区B 弃风电量(kW)'] = max(row['园区B 总出力(kW)'] - row['园区B负荷(kW)'] - actual_charge / charge_efficiency, 0)

    # 园区C储能策略
    if row['园区C负荷(kW)'] > row['园区C 总出力(kW)']:  # 负荷大于发电，放电
        discharge = min((row['园区C负荷(kW)'] - row['园区C 总出力(kW)']) / discharge_efficiency, storage_power)
        actual_discharge = min(discharge, (row['SOC_C'] - storage_capacity * SOC_min))
        data.at[index, 'SOC_C'] -= actual_discharge
        data.at[index, '园区C 购电量(kW)'] = max(row['园区C负荷(kW)'] - row['园区C 总出力(kW)'] - actual_discharge * discharge_efficiency, 0)
    else:  # 负荷小于发电，充电
        charge = min((row['园区C 总出力(kW)'] - row['园区C负荷(kW)']) * charge_efficiency, storage_power)
        actual_charge = min(charge, (storage_capacity * SOC_max - row['SOC_C']))
        data.at[index, 'SOC_C'] += actual_charge
        data.at[index, '园区C 弃光弃风电量(kW)'] = max(row['园区C 总出力(kW)'] - row['园区C负荷(kW)'] - actual_charge / charge_efficiency, 0)

# 计算配置储能后的购电量和弃电量
total_purchase_A_storage = data['园区A 购电量(kW)'].sum()
total_purchase_B_storage = data['园区B 购电量(kW)'].sum()
total_purchase_C_storage = data['园区C 购电量(kW)'].sum()

total_waste_A_storage = data['园区A 弃光电量(kW)'].sum()
total_waste_B_storage = data['园区B 弃风电量(kW)'].sum()
total_waste_C_storage = data['园区C 弃光弃风电量(kW)'].sum()

# 计算配置储能后的总供电成本和单位电量平均供电成本
purchase_cost_A_storage = total_purchase_A_storage * 1  # 园区A的购电成本
purchase_cost_B_storage = total_purchase_B_storage * 1  # 园区B的购电成本
purchase_cost_C_storage = total_purchase_C_storage * 1  # 园区C的购电成本

average_cost_A_storage = purchase_cost_A_storage / data['园区A负荷(kW)'].sum()
average_cost_B_storage = purchase_cost_B_storage / data['园区B负荷(kW)'].sum()
average_cost_C_storage = purchase_cost_C_storage / data['园区C负荷(kW)'].sum()

# 输出结果
result_storage = {
    "园区A": {
        "总购电量(kWh)": total_purchase_A_storage,
        "弃光电量(kWh)": total_waste_A_storage,
        "总供电成本(元)": purchase_cost_A_storage,
        "单位电量平均供电成本(元/kWh)": average_cost_A_storage
    },
    "园区B": {
        "总购电量(kWh)": total_purchase_B_storage,
        "弃风电量(kWh)": total_waste_B_storage,
        "总供电成本(元)": purchase_cost_B_storage,
        "单位电量平均供电成本(元/kWh)": average_cost_B_storage
    },
    "园区C": {
        "总购电量(kWh)": total_purchase_C_storage,
        "弃光弃风电量(kWh)": total_waste_C_storage,
        "总供电成本(元)": purchase_cost_C_storage,
        "单位电量平均供电成本(元/kWh)": average_cost_C_storage
    }
}

for key, value in result_storage.items():
    print(f"\n{key} 结果（配置储能后）：")
    for sub_key, sub_value in value.items():
        print(f"{sub_key}: {sub_value:.2f}")

#######################################
# 判断50kW/100kWh的方案是否最优
# 并制定各园区最优的储能功率和容量配置方案
#######################################

from scipy.optimize import minimize

# 储能参数
years = 10  # 运行寿命 (年)
cost_per_kWh = 1  # 购电成本 (元/kWh)
storage_cost_power = 800  # 储能功率单价 (元/kW)
storage_cost_capacity = 1800  # 储能容量单价 (元/kWh)
charge_efficiency = 0.95
discharge_efficiency = 0.95
hours_per_day = 24

# 计算储能优化后购电量的函数
def calculate_purchase(storage_power, storage_capacity, load, generation, initial_soc, min_soc, max_soc):
    soc = initial_soc
    purchase = 0
    for i in range(len(load)):
        if load[i] > generation[i]:  # 放电
            discharge = min((load[i] - generation[i]) / discharge_efficiency, storage_power)
            actual_discharge = min(discharge, soc - storage_capacity * min_soc)
            soc -= actual_discharge
            purchase += max(load[i] - generation[i] - actual_discharge * discharge_efficiency, 0)
        else:  # 充电
            charge = min((generation[i] - load[i]) * charge_efficiency, storage_power)
            actual_charge = min(charge, storage_capacity * max_soc - soc)
            soc += actual_charge
    return purchase

# 目标函数：最小化总成本 = 储能成本 + 购电成本
def total_cost(x):
    P_A, E_A, P_B, E_B, P_C, E_C = x
    storage_cost = (P_A * storage_cost_power + E_A * storage_cost_capacity + P_B * storage_cost_power + E_B * storage_cost_capacity + P_C * storage_cost_power + E_C * storage_cost_capacity) * years
    purchase_cost_A = calculate_purchase(P_A, E_A, data['园区A负荷(kW)'], data['园区A 总出力(kW)'], SOC_initial, SOC_min, SOC_max) * cost_per_kWh
    purchase_cost_B = calculate_purchase(P_B, E_B, data['园区B负荷(kW)'], data['园区B 总出力(kW)'], SOC_initial, SOC_min, SOC_max) * cost_per_kWh
    purchase_cost_C = calculate_purchase(P_C, E_C, data['园区C负荷(kW)'], data['园区C 总出力(kW)'], SOC_initial, SOC_min, SOC_max) * cost_per_kWh
    total_purchase_cost = purchase_cost_A + purchase_cost_B + purchase_cost_C
    return storage_cost + total_purchase_cost

# 约束条件
constraints = [
    {'type': 'ineq', 'fun': lambda x: x[0]},  # P_A >= 0
    {'type': 'ineq', 'fun': lambda x: x[1]},  # E_A >= 0
    {'type': 'ineq', 'fun': lambda x: x[2]},  # P_B >= 0
    {'type': 'ineq', 'fun': lambda x: x[3]},  # E_B >= 0
    {'type': 'ineq', 'fun': lambda x: x[4]},  # P_C >= 0
    {'type': 'ineq', 'fun': lambda x: x[5]}   # E_C >= 0
]

# 初始猜测值
x0 = [50, 100, 50, 100, 50, 100]

# 优化求解
result = minimize(total_cost, x0, method='SLSQP', constraints=constraints)

# 提取结果
optimal_storage_power_A = result.x[0]
optimal_storage_capacity_A = result.x[1]
optimal_storage_power_B = result.x[2]
optimal_storage_capacity_B = result.x[3]
optimal_storage_power_C = result.x[4]
optimal_storage_capacity_C = result.x[5]

# 输出结果
print(f"最优储能配置方案：")
print(f"园区A储能功率: {optimal_storage_power_A:.2f} kW")
print(f"园区A储能容量: {optimal_storage_capacity_A:.2f} kWh")
print(f"园区B储能功率: {optimal_storage_power_B:.2f} kW")
print(f"园区B储能容量: {optimal_storage_capacity_B:.2f} kWh")
print(f"园区C储能功率: {optimal_storage_power_C:.2f} kW")
print(f"园区C储能容量: {optimal_storage_capacity_C:.2f} kWh")
