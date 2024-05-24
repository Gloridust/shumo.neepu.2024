# import pandas as pd

# # 加载数据
# male_student_diet = pd.read_excel('./mnt/data/附件1：1名男大学生的一日食谱.xlsx')
# female_student_diet = pd.read_excel('./mnt/data/附件2：1名女大学生的一日食谱.xlsx')

# # 提取和整理食谱数据
# male_student_diet_cleaned = male_student_diet[['食物名称', '可食部（克/份）', '食用份数', '食用时间']].dropna().reset_index(drop=True)
# female_student_diet_cleaned = female_student_diet[['食物名称', '可食部（克/份）', '食用份数', '食用时间']].dropna().reset_index(drop=True)

# # 计算每种食物的总可食部量
# male_student_diet_cleaned['总可食部量（克）'] = male_student_diet_cleaned['可食部（克/份）'] * male_student_diet_cleaned['食用份数']
# female_student_diet_cleaned['总可食部量（克）'] = female_student_diet_cleaned['可食部（克/份）'] * female_student_diet_cleaned['食用份数']

import pandas as pd

# Load the provided files
male_student_diet = pd.read_excel('./mnt/data/附件1：1名男大学生的一日食谱.xlsx')
female_student_diet = pd.read_excel('./mnt/data/附件2：1名女大学生的一日食谱.xlsx')

# Nutrition data for a selection of common foods (per 100g)
nutrition_data = {
    '小米粥': {'蛋白质': 1.5, '脂肪': 0.4, '碳水化合物': 9.2, '钙': 6, '铁': 0.4, '锌': 0.3, '维生素A': 0, '维生素B1': 0.03, '维生素B2': 0.02, '维生素C': 0},
    '油条': {'蛋白质': 6.6, '脂肪': 20.0, '碳水化合物': 47.8, '钙': 20, '铁': 1.8, '锌': 0.5, '维生素A': 0, '维生素B1': 0.02, '维生素B2': 0.08, '维生素C': 0},
    '煎鸡蛋': {'蛋白质': 13.3, '脂肪': 10.3, '碳水化合物': 1.0, '钙': 56, '铁': 2.0, '锌': 1.3, '维生素A': 140, '维生素B1': 0.11, '维生素B2': 0.30, '维生素C': 0},
    '拌海带丝': {'蛋白质': 1.7, '脂肪': 0.5, '碳水化合物': 4.1, '钙': 225, '铁': 2.0, '锌': 0.8, '维生素A': 50, '维生素B1': 0.04, '维生素B2': 0.12, '维生素C': 1},
    '大米饭': {'蛋白质': 2.6, '脂肪': 0.3, '碳水化合物': 25.7, '钙': 3, '铁': 0.2, '锌': 0.4, '维生素A': 0, '维生素B1': 0.02, '维生素B2': 0.01, '维生素C': 0},
    '豆浆': {'蛋白质': 3.6, '脂肪': 1.8, '碳水化合物': 2.9, '钙': 15, '铁': 0.9, '锌': 0.6, '维生素A': 0, '维生素B1': 0.03, '维生素B2': 0.05, '维生素C': 0},
    '鸡排面': {'蛋白质': 12.3, '脂肪': 5.6, '碳水化合物': 42.1, '钙': 30, '铁': 2.5, '锌': 1.0, '维生素A': 0, '维生素B1': 0.05, '维生素B2': 0.06, '维生素C': 0},
    '鸡蛋饼': {'蛋白质': 7.2, '脂肪': 8.6, '碳水化合物': 14.4, '钙': 18, '铁': 1.2, '锌': 0.7, '维生素A': 90, '维生素B1': 0.08, '维生素B2': 0.15, '维生素C': 0},
    '水饺': {'蛋白质': 7.3, '脂肪': 4.5, '碳水化合物': 31.7, '钙': 20, '铁': 1.8, '锌': 0.9, '维生素A': 0, '维生素B1': 0.03, '维生素B2': 0.06, '维生素C': 0},
    '葡萄': {'蛋白质': 0.4, '脂肪': 0.2, '碳水化合物': 10.8, '钙': 8, '铁': 0.3, '锌': 0.1, '维生素A': 0, '维生素B1': 0.02, '维生素B2': 0.02, '维生素C': 1},
    '香菇炒油菜': {'蛋白质': 2.1, '脂肪': 3.4, '碳水化合物': 4.2, '钙': 34, '铁': 1.2, '锌': 0.7, '维生素A': 20, '维生素B1': 0.04, '维生素B2': 0.06, '维生素C': 3},
    '炒肉蒜台': {'蛋白质': 6.8, '脂肪': 9.2, '碳水化合物': 5.1, '钙': 25, '铁': 1.0, '锌': 0.8, '维生素A': 10, '维生素B1': 0.07, '维生素B2': 0.14, '维生素C': 4},
    '茄汁沙丁鱼': {'蛋白质': 16.3, '脂肪': 9.6, '碳水化合物': 3.7, '钙': 50, '铁': 2.2, '锌': 1.2, '维生素A': 60, '维生素B1': 0.10, '维生素B2': 0.22, '维生素C': 2},
    '苹果': {'蛋白质': 0.2, '脂肪': 0.1, '碳水化合物': 13.0, '钙': 3, '铁': 0.1, '锌': 0.1, '维生素A': 0, '维生素B1': 0.02, '维生素B2': 0.02, '维生素C': 5}
}

# Function to calculate nutrients from diet
def calculate_nutrients(diet, nutrition_data):
    nutrients = {'蛋白质': 0, '脂肪': 0, '碳水化合物': 0, '钙': 0, '铁': 0, '锌': 0, '维生素A': 0, '维生素B1': 0, '维生素B2': 0, '维生素C': 0}
    for index, row in diet.iterrows():
        food_name = row['食物名称']
        if food_name in nutrition_data:
            for key in nutrients:
                nutrients[key] += nutrition_data[food_name][key] * row['可食部（克/份）'] / 100 * row['食用份数']
    return nutrients

# Calculate nutrients for male and female diets
male_nutrients = calculate_nutrients(male_student_diet, nutrition_data)
female_nutrients = calculate_nutrients(female_student_diet, nutrition_data)

# Display results
print(male_nutrients, female_nutrients)

