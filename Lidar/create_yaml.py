import yaml

# 创建一个字典来存储外参和内参信息
data = {
    'extrinsic': {
        'R': [[0.0092749, -0.999957, 0.000449772], [0.00118781, -0.000438773, -0.999999], [0.999956, 0.00927542, 0.00118369]],
        'T': [0.00529624, 0.0306859, -0.135507]
    },
    'intrinsic': {
        'fx': 1246.7920,
        'fy': 1243.23027688354,
        'cx': 637.8469,
        'cy': 506.5883
    },
    'distortion': [-0.100813, 0.58183, 0.0031347, 0.00040115, 0]
}

# 将字典写入YAML文件
with open('parameters.yaml', 'w') as file:
    yaml.dump(data, file)

# 从YAML文件中读取字典
with open('parameters.yaml', 'r') as file:
    data_loaded = yaml.safe_load(file)

print(data_loaded)