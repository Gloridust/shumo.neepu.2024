import pandas as pd

# 读取各园区典型日负荷数据
load_data = pd.read_excel('./mnt/data/附件1：各园区典型日负荷数据.xlsx')

# 读取各园区典型日风光发电数据
generation_data = pd.read_excel('./mnt/data/附件2：各园区典型日风光发电数据.xlsx')

# 各园区装机容量
Pv_A = 750  # kW
Pv_C = 600  # kW
Pw_B = 1000 # kW
Pw_C = 500  # kW

# 计算各园区的光伏和风电实际出力
generation_data['园区A 光伏出力(kW)'] = generation_data['园区A 光伏出力（p.u.）'] * Pv_A
generation_data['园区C 光伏出力(kW)'] = generation_data['园区C 光伏出力（p.u.）'] * Pv_C
generation_data['园区B 风电出力(kW)'] = generation_data['园区B风电出力（p.u.）'] * Pw_B
generation_data['园区C 风电出力(kW)'] = generation_data['园区C 风电出力（p.u.）'] * Pw_C

# 合并负荷数据和发电数据
merged_data = pd.merge(load_data, generation_data, on='时间（h）')

# 计算每日总负荷和总发电量
daily_load = merged_data[['园区A负荷(kW)', '园区B负荷(kW)', '园区C负荷(kW)']].sum()
daily_generation_A = merged_data['园区A 光伏出力(kW)'].sum()
daily_generation_B = merged_data['园区B 风电出力(kW)'].sum()
daily_generation_C = merged_data[['园区C 光伏出力(kW)', '园区C 风电出力(kW)']].sum().sum()

# 打印每日总负荷和总发电量
print(f"园区A每日总负荷: {daily_load['园区A负荷(kW)']} kWh")
print(f"园区B每日总负荷: {daily_load['园区B负荷(kW)']} kWh")
print(f"园区C每日总负荷: {daily_load['园区C负荷(kW)']} kWh")
print(f"园区A每日总光伏发电量: {daily_generation_A} kWh")
print(f"园区B每日总风电发电量: {daily_generation_B} kWh")
print(f"园区C每日总发电量: {daily_generation_C} kWh")

print("##### 独立运营方案计算 #####")

# 储能系统参数
power_unit_cost = 800  # 元/kW
energy_unit_cost = 1800  # 元/kWh
soc_min = 0.1
soc_max = 0.9
efficiency = 0.95

# 计算各园区储能需求
def calculate_storage(daily_load, daily_generation):
    # 需要储能的能量 = (每日负荷 - 每日发电) * 充/放电效率
    energy_storage_needed = max(0, (daily_load - daily_generation) / efficiency)
    # 储能系统能量配置 = 储能需求 / (SOC最大值 - SOC最小值)
    energy_storage_capacity = energy_storage_needed / (soc_max - soc_min)
    # 储能系统功率配置 = 每日最大负荷
    power_storage_capacity = daily_load / 24  # 假设最大负荷平均分布
    return energy_storage_capacity, power_storage_capacity

# 园区A储能配置
energy_storage_A, power_storage_A = calculate_storage(daily_load['园区A负荷(kW)'], daily_generation_A)
# 园区B储能配置
energy_storage_B, power_storage_B = calculate_storage(daily_load['园区B负荷(kW)'], daily_generation_B)
# 园区C储能配置
energy_storage_C, power_storage_C = calculate_storage(daily_load['园区C负荷(kW)'], daily_generation_C)

# 计算投资成本
def calculate_investment_cost(energy_storage, power_storage):
    return energy_storage * energy_unit_cost + power_storage * power_unit_cost

# 园区A投资成本
investment_cost_A = calculate_investment_cost(energy_storage_A, power_storage_A)
# 园区B投资成本
investment_cost_B = calculate_investment_cost(energy_storage_B, power_storage_B)
# 园区C投资成本
investment_cost_C = calculate_investment_cost(energy_storage_C, power_storage_C)

# 打印结果
print(f"园区A储能系统能量配置: {energy_storage_A:.2f} kWh, 功率配置: {power_storage_A:.2f} kW, 投资成本: {investment_cost_A:.2f} 元")
print(f"园区B储能系统能量配置: {energy_storage_B:.2f} kWh, 功率配置: {power_storage_B:.2f} kW, 投资成本: {investment_cost_B:.2f} 元")
print(f"园区C储能系统能量配置: {energy_storage_C:.2f} kWh, 功率配置: {power_storage_C:.2f} kW, 投资成本: {investment_cost_C:.2f} 元")

print("##### 联合运营方案计算 #####")

# 计算整体储能需求
total_daily_load = daily_load.sum()
total_daily_generation = daily_generation_A + daily_generation_B + daily_generation_C

# 联合运营储能配置
total_energy_storage, total_power_storage = calculate_storage(total_daily_load, total_daily_generation)

# 联合运营投资成本
total_investment_cost = calculate_investment_cost(total_energy_storage, total_power_storage)

# 打印结果
print(f"联合运营储能系统能量配置: {total_energy_storage:.2f} kWh, 功率配置: {total_power_storage:.2f} kW, 投资成本: {total_investment_cost:.2f} 元")

print("##### 购电成本计算 #####")

import numpy as np

# 分时电价
peak_price = 1.0  # 元/kWh
off_peak_price = 0.4  # 元/kWh

# 定义分时电价时段
def get_electricity_price(hour):
    if 7 <= hour < 22:
        return peak_price
    else:
        return off_peak_price

# 计算购电成本
def calculate_electricity_cost(load_data, generation_data, daily_load, daily_generation):
    total_cost = 0
    for hour in range(24):
        load = load_data.iloc[hour, 1:4].sum()  # 每小时总负荷
        generation = generation_data.iloc[hour, 1:5].sum()  # 每小时总发电量
        net_load = load - generation  # 净负荷
        if net_load > 0:
            price = get_electricity_price(hour)
            total_cost += net_load * price
    return total_cost

# 各园区独立运营购电成本
electricity_cost_A = calculate_electricity_cost(load_data[['时间（h）', '园区A负荷(kW)']], generation_data[['时间（h）', '园区A 光伏出力(kW)']], daily_load['园区A负荷(kW)'], daily_generation_A)
electricity_cost_B = calculate_electricity_cost(load_data[['时间（h）', '园区B负荷(kW)']], generation_data[['时间（h）', '园区B 风电出力(kW)']], daily_load['园区B负荷(kW)'], daily_generation_B)
electricity_cost_C = calculate_electricity_cost(load_data[['时间（h）', '园区C负荷(kW)']], generation_data[['时间（h）', '园区C 光伏出力(kW)', '园区C 风电出力(kW)']], daily_load['园区C负荷(kW)'], daily_generation_C)

# 联合运营购电成本
electricity_cost_joint = calculate_electricity_cost(load_data, generation_data, total_daily_load, total_daily_generation)

# 打印购电成本
print(f"园区A每日购电成本: {electricity_cost_A:.2f} 元")
print(f"园区B每日购电成本: {electricity_cost_B:.2f} 元")
print(f"园区C每日购电成本: {electricity_cost_C:.2f} 元")
print(f"联合运营每日购电成本: {electricity_cost_joint:.2f} 元")

# 计算年购电成本
days_per_year = 365
annual_cost_A = electricity_cost_A * days_per_year
annual_cost_B = electricity_cost_B * days_per_year
annual_cost_C = electricity_cost_C * days_per_year
annual_cost_joint = electricity_cost_joint * days_per_year

print("##### 五年总购电成本和总投资成本 #####")
# 计算5年总购电成本
total_cost_5_years_A = annual_cost_A * 5
total_cost_5_years_B = annual_cost_B * 5
total_cost_5_years_C = annual_cost_C * 5
total_cost_5_years_joint = annual_cost_joint * 5

# 计算5年总投资成本（包括储能系统投资和购电成本）
total_investment_5_years_A = investment_cost_A + total_cost_5_years_A
total_investment_5_years_B = investment_cost_B + total_cost_5_years_B
total_investment_5_years_C = investment_cost_C + total_cost_5_years_C
total_investment_5_years_joint = total_investment_cost + total_cost_5_years_joint

# 打印5年总成本
print(f"园区A储能系统5年总成本: {total_investment_5_years_A:.2f} 元")
print(f"园区B储能系统5年总成本: {total_investment_5_years_B:.2f} 元")
print(f"园区C储能系统5年总成本: {total_investment_5_years_C:.2f} 元")
print(f"联合运营储能系统5年总成本: {total_investment_5_years_joint:.2f} 元")

# 比较不同方案的5年总成本
print("\n##### 5年总成本比较 #####")
print(f"园区A储能系统5年总成本: {total_investment_5_years_A:.2f} 元")
print(f"园区B储能系统5年总成本: {total_investment_5_years_B:.2f} 元")
print(f"园区C储能系统5年总成本: {total_investment_5_years_C:.2f} 元")
print(f"联合运营储能系统5年总成本: {total_investment_5_years_joint:.2f} 元")

# 总结最佳方案
if total_investment_5_years_joint < min(total_investment_5_years_A, total_investment_5_years_B, total_investment_5_years_C):
    print("联合运营方案在经济上最优。")
else:
    min_cost = min(total_investment_5_years_A, total_investment_5_years_B, total_investment_5_years_C)
    if min_cost == total_investment_5_years_A:
        print("园区A的独立运营方案在经济上最优。")
    elif min_cost == total_investment_5_years_B:
        print("园区B的独立运营方案在经济上最优。")
    else:
        print("园区C的独立运营方案在经济上最优。")
