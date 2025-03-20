import sys
sys.path.append("/home/kara/.local/lib/python3.8/site-packages/")
from googleapiclient import discovery
import json
import requests
import numpy as np
import time

# 读取txt文件中的句子
def read_sentences_from_file(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for row in file:
            sentences.append(json.loads(row)['recover'].strip())
    return sentences

# 构建请求对象
def build_analyze_requests(sentences):
    analyze_requests = []
    for sentence in sentences:
        analyze_request = {
            'comment': {'text': sentence},
            'requestedAttributes': {'TOXICITY': {}},
            'languages': ['en']
        }
        analyze_requests.append(analyze_request)
    return analyze_requests


if __name__ == '__main__':
    file_path = '/home/kara/DiffuSeq/generation_outputs/diffuseq_second-detox_h128_lr1e-05_t2000_sqrt_lossaware_seed102_learned_mask_fp16_denoise_0.5_reproduce20240721-19:15:12/ema_0.9999_035000.pt.samples/seed123_step0_none.json'  # 替换为你的txt文件路径
    api_url = 'https://api.example.com/analyze'  # 替换为你的API URL
    API_KEY = None # REPLACE YOUR API KEY HERE

    client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
    )

    sentences = read_sentences_from_file(file_path)
    print(len(sentences))
    analyze_requests = build_analyze_requests(sentences)
    responses = []
    for request_data in analyze_requests:
        response = client.comments().analyze(body=request_data).execute()
        responses.append(response)
        time.sleep(4)

    # 提取TOXICITY值
    toxicity_values = []
    for response in responses:
        toxicity_value = response['attributeScores']['TOXICITY']['summaryScore']['value']
        toxicity_values.append(toxicity_value)

    avg_toxic_score = np.mean(toxicity_values)
    print(avg_toxic_score)

# ----------------------------------------果然还是要用循环发请求，没法一起发一个


# response = client.comments().analyze(body=analyze_request).execute()
# print(json.dumps(responses, indent=2))
# print(response['attributeScores']['TOXICITY']['summaryScore']['value'])
