import pandas as pd

# 读取负荷数据
load_data = pd.read_excel('./mnt/data/附件1：各园区典型日负荷数据.xlsx')
generation_data = pd.read_excel('./mnt/data/附件2：各园区典型日风光发电数据.xlsx')

# 将时间转换为datetime类型
load_data['时间（h）'] = pd.to_datetime(load_data['时间（h）'], format='%H:%M:%S')
generation_data['时间（h）'] = pd.to_datetime(generation_data['时间（h）'], format='%H:%M:%S')

# 设置时间为索引
load_data.set_index('时间（h）', inplace=True)
generation_data.set_index('时间（h）', inplace=True)

# 计算各园区的总负荷和总发电
load_data['总负荷(kW)'] = load_data['园区A负荷(kW)'] + load_data['园区B负荷(kW)'] + load_data['园区C负荷(kW)']
generation_data['总光伏出力(kW)'] = generation_data['园区A 光伏出力（p.u.）'] * 750 + generation_data['园区C 光伏出力（p.u.）'] * 600
generation_data['总风电出力(kW)'] = generation_data['园区B风电出力（p.u.）'] * 1000 + generation_data['园区C 风电出力（p.u.）'] * 500
generation_data['总发电(kW)'] = generation_data['总光伏出力(kW)'] + generation_data['总风电出力(kW)']

# 合并负荷和发电数据
combined_data = pd.concat([load_data, generation_data], axis=1)

# 计算购电量和弃电量
combined_data['购电量(kW)'] = np.maximum(combined_data['总负荷(kW)'] - combined_data['总发电(kW)'], 0)
combined_data['弃电量(kW)'] = np.maximum(combined_data['总发电(kW)'] - combined_data['总负荷(kW)'], 0)

# 计算总购电量、总弃电量、总供电成本和单位电量平均供电成本
total_purchase = combined_data['购电量(kW)'].sum()
total_waste = combined_data['弃电量(kW)'].sum()
total_cost = total_purchase * 1  # 购电成本为1元/kWh
average_cost = total_cost / combined_data['总负荷(kW)'].sum()

# 输出结果
print(f"联合园区未配置储能时的经济性分析：")
print(f"总购电量(kWh): {total_purchase:.2f}")
print(f"总弃电量(kWh): {total_waste:.2f}")
print(f"总供电成本(元): {total_cost:.2f}")
print(f"单位电量平均供电成本(元/kWh): {average_cost:.2f}")
