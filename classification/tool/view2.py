import re
from graphviz import Digraph

# 读取文件内容
def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# 解析架构
def parse_architecture(architecture_str):
    layers = []
    connections = []
    layer_pattern = re.compile(r'\(([^)]+)\): ([A-Za-z0-9_]+)')
    
    for match in layer_pattern.finditer(architecture_str):
        layer_id, layer_type = match.groups()
        layers.append((layer_id, layer_type))
        if len(layers) > 1:
            connections.append((layers[-2][0], layers[-1][0]))
    
    return layers, connections

# 生成图像
def create_graph(layers, connections, output_file):
    dot = Digraph(comment='Model Architecture')
    
    for layer_id, layer_type in layers:
        dot.node(layer_id, f'{layer_id}\n({layer_type})')
    
    for connection in connections:
        dot.edge(*connection)
    
    dot.render(output_file, format='png', view=False)  # 禁用自动查看

# 示例使用
file_path = '/home/kara/DiffuSeq/file/model_architecture.txt'
architecture_str = read_file(file_path)
layers, connections = parse_architecture(architecture_str)
create_graph(layers, connections, '/home/kara/DiffuSeq/file/model_architecture')
