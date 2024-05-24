import pandas as pd

# 加载数据
male_student_diet = pd.read_excel('./mnt/data/附件1：1名男大学生的一日食谱.xlsx')
female_student_diet = pd.read_excel('./mnt/data/附件2：1名女大学生的一日食谱.xlsx')

# 提取和整理食谱数据
male_student_diet_cleaned = male_student_diet[['食物名称', '可食部（克/份）', '食用份数', '食用时间']].dropna().reset_index(drop=True)
female_student_diet_cleaned = female_student_diet[['食物名称', '可食部（克/份）', '食用份数', '食用时间']].dropna().reset_index(drop=True)

# 计算每种食物的总可食部量
male_student_diet_cleaned['总可食部量（克）'] = male_student_diet_cleaned['可食部（克/份）'] * male_student_diet_cleaned['食用份数']
female_student_diet_cleaned['总可食部量（克）'] = female_student_diet_cleaned['可食部（克/份）'] * female_student_diet_cleaned['食用份数']

# 模拟的营养成分数据（每100克食物）
nutrient_data = {
    '食物名称': ['小米粥', '大米饭', '砂锅面', '豆浆', '鸡蛋饼'],
    '能量（kcal）': [70, 130, 150, 54, 180],
    '蛋白质（g）': [2.3, 2.5, 5.0, 3.6, 6.0],
    '脂肪（g）': [0.6, 0.3, 2.0, 1.8, 9.0],
    '碳水化合物（g）': [15, 28, 30, 6, 23],
    '钙（mg）': [8, 10, 20, 25, 50],
    '铁（mg）': [0.5, 0.3, 1.0, 1.2, 2.5],
    '锌（mg）': [0.3, 0.4, 0.8, 0.6, 1.1],
    '维生素A（μg）': [0, 0, 0, 0, 100],
    '维生素B1（mg）': [0.05, 0.04, 0.1, 0.15, 0.2],
    '维生素B2（mg）': [0.02, 0.03, 0.05, 0.1, 0.15],
    '维生素C（mg）': [0, 0, 0, 1, 2]
}

# 转换为DataFrame
nutrient_df = pd.DataFrame(nutrient_data)

# 合并营养数据和学生饮食数据
male_diet_merged = male_student_diet_cleaned.merge(nutrient_df, on='食物名称', how='left')
female_diet_merged = female_student_diet_cleaned.merge(nutrient_df, on='食物名称', how='left')

# 计算总营养素摄入量
male_diet_merged['能量总量（kcal）'] = male_diet_merged['总可食部量（克）'] * male_diet_merged['能量（kcal）'] / 100
male_diet_merged['蛋白质总量（g）'] = male_diet_merged['总可食部量（克）'] * male_diet_merged['蛋白质（g）'] / 100
male_diet_merged['脂肪总量（g）'] = male_diet_merged['总可食部量（克）'] * male_diet_merged['脂肪（g）'] / 100
male_diet_merged['碳水化合物总量（g）'] = male_diet_merged['总可食部量（克）'] * male_diet_merged['碳水化合物（g）'] / 100
male_diet_merged['钙总量（mg）'] = male_diet_merged['总可食部量（克）'] * male_diet_merged['钙（mg）'] / 100
male_diet_merged['铁总量（mg）'] = male_diet_merged['总可食部量（克）'] * male_diet_merged['铁（mg）'] / 100
male_diet_merged['锌总量（mg）'] = male_diet_merged['总可食部量（克）'] * male_diet_merged['锌（mg）'] / 100
male_diet_merged['维生素A总量（μg）'] = male_diet_merged['总可食部量（克）'] * male_diet_merged['维生素A（μg）'] / 100
male_diet_merged['维生素B1总量（mg）'] = male_diet_merged['总可食部量（克）'] * male_diet_merged['维生素B1（mg）'] / 100
male_diet_merged['维生素B2总量（mg）'] = male_diet_merged['总可食部量（克）'] * male_diet_merged['维生素B2（mg）'] / 100
male_diet_merged['维生素C总量（mg）'] = male_diet_merged['总可食部量（克）'] * male_diet_merged['维生素C（mg）'] / 100

female_diet_merged['能量总量（kcal）'] = female_diet_merged['总可食部量（克）'] * female_diet_merged['能量（kcal）'] / 100
female_diet_merged['蛋白质总量（g）'] = female_diet_merged['总可食部量（克）'] * female_diet_merged['蛋白质（g）'] / 100
female_diet_merged['脂肪总量（g）'] = female_diet_merged['总可食部量（克）'] * female_diet_merged['脂肪（g）'] / 100
female_diet_merged['碳水化合物总量（g）'] = female_diet_merged['总可食部量（克）'] * female_diet_merged['碳水化合物（g）'] / 100
female_diet_merged['钙总量（mg）'] = female_diet_merged['总可食部量（克）'] * female_diet_merged['钙（mg）'] / 100
female_diet_merged['铁总量（mg）'] = female_diet_merged['总可食部量（克）'] * female_diet_merged['铁（mg）'] / 100
female_diet_merged['锌总量（mg）'] = female_diet_merged['总可食部量（克）'] * female_diet_merged['锌（mg）'] / 100
female_diet_merged['维生素A总量（μg）'] = female_diet_merged['总可食部量（克）'] * female_diet_merged['维生素A（μg）'] / 100
female_diet_merged['维生素B1总量（mg）'] = female_diet_merged['总可食部量（克）'] * female_diet_merged['维生素B1（mg）'] / 100
female_diet_merged['维生素B2总量（mg）'] = female_diet_merged['总可食部量（克）'] * female_diet_merged['维生素B2（mg）'] / 100
female_diet_merged['维生素C总量（mg）'] = female_diet_merged['总可食部量（克）'] * female_diet_merged['维生素C（mg）'] / 100

# 计算每日总摄入量
male_daily_totals = male_diet_merged[['能量总量（kcal）', '蛋白质总量（g）', '脂肪总量（g）', '碳水化合物总量（g）',
                                      '钙总量（mg）', '铁总量（mg）', '锌总量（mg）', '维生素A总量（μg）',
                                      '维生素B1总量（mg）', '维生素B2总量（mg）', '维生素C总量（mg）']].sum()

female_daily_totals = female_diet_merged
