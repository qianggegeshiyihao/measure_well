import pandas as pd

# 读取Excel文件
df = pd.read_excel('sample/demo.xlsx')

# 创建新的DataFrame来保存统计结果
results = []

# 按地层分组
for strata, group in df.groupby('_地层'):  # 假设地层列名为'地层'
    # 获取地层的整体顶深和底深
    top_depth = group['顶深'].min()  # 假设顶深列名为'顶深'
    bottom_depth = group['底深'].max()  # 假设底深列名为'底深'
    strata_thickness = bottom_depth - top_depth
    
    # 初始化岩性厚度统计
    rock_thickness = {
        '泥岩': 0,
        '灰岩': 0,
        '煤': 0,
        '砂岩': 0,
        '砾岩': 0
    }
    
    # 统计各岩性的厚度
    for _, row in group.iterrows():
        rock_type = row['岩性']  # 假设岩性列名为'岩性'
        layer_thickness = row['层厚']  # 假设层厚列名为'层厚'
        
        for rock in rock_thickness:
            if rock in rock_type:
                rock_thickness[rock] += layer_thickness
                break  # 如果找到岩性，不需要检查其他岩性
    
    # 合并砂岩和砾岩的层厚
    sandstone_thickness = rock_thickness['砂岩'] + rock_thickness['砾岩']
    
    # 添加到结果列表中
    results.append({
        '井号': group['井号'].iloc[0],  # 假设井号列名为'井号'
        '地层': strata,
        '顶深': top_depth,
        '底深': bottom_depth,
        '地层厚度': strata_thickness,
        '砂体厚度': sandstone_thickness,
        '灰岩厚度': rock_thickness['灰岩'],
        '煤厚度': rock_thickness['煤'],
        '泥岩厚度': rock_thickness['泥岩']
    })

# 将结果列表转换为DataFrame
results_df = pd.DataFrame(results)

# 按照地层顶深排序
results_df.sort_values('顶深', inplace=True)


# 将结果DataFrame写入新的Excel文件
results_df.to_excel('strata_summary.xlsx', index=False, engine='openpyxl')
