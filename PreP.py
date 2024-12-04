 
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as Image  # Opencv 와 같은 이미지 처리 라이브러리

DEFAULT_ATTRIBUTES = (
    'index', 
    'uuid', 
    'name', 
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)
'''subprocess는 다양한 방법으로 시스템 명령을 실행하는 모듈'''
def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]
    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]
 
 
 
def show_bar_01(origin_train_df,test_df ) : 
    ct0_origin = origin_train_df['path'].str.contains('/0/').sum()
    ct1_origin = origin_train_df['path'].str.contains('/1/').sum()
    ct2_origin = origin_train_df['path'].str.contains('/2/').sum()
    ct3_origin = origin_train_df['path'].str.contains('/3/').sum()
    ct4_origin = origin_train_df['path'].str.contains('/4/').sum()
    ct0_test = test_df['path'].str.contains('/0/').sum()
    ct1_test = test_df['path'].str.contains('/1/').sum()
    ct2_test = test_df['path'].str.contains('/2/').sum()
    ct3_test = test_df['path'].str.contains('/3/').sum()
    ct4_test = test_df['path'].str.contains('/4/').sum()
    labels = ['1', '2', '3', '4', '5']
    values_ori = [ct0_origin, ct1_origin, ct2_origin, ct3_origin, ct4_origin]
    values_test = [ct0_test, ct1_test, ct2_test, ct3_test, ct4_test]
    x = np.arange(len(labels))  # x 위치
    width = 0.35  # 막대의 너비
    # 비교 막대그래프 그리기
    fig, ax = plt.subplots()
    # 첫 번째 막대 (data1)
    rects1 = ax.bar(x - width/2, values_ori, width, label='train')
    # 두 번째 막대 (data2)
    rects2 = ax.bar(x + width/2, values_test, width, label='test')
    # 그래프 레이블 설정
    ax.set_xlabel('Categories')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of Two Data Sets')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # 범례 추가
    ax.legend()
    plt.show()
    
    
    
def compute_overall_mean_std(dfs):
    all_pixels = {0: [], 1: [], 2: []}

    for df in dfs:
        for index, row in df.iterrows():
            img_path = row['path']  # 'path' 대신에 실제 경로가 있는 컬럼명을 사용해주세요.
            img = Image.open(img_path)
            img_np = np.array(img)

            for i in range(3): # RGB 채널
                channel_pixels = img_np[:, :, i].ravel().tolist()  # 각 채널의 모든 픽셀 값을 수집합니다.
                all_pixels[i].extend(channel_pixels)

    means = [np.mean(all_pixels[i]) for i in range(3)]
    stds = [np.std(all_pixels[i]) for i in range(3)]
    return means, stds    

# all_pixels = {0: [], 1: [], 2: []}
# for df in [train_df, valid_df]:
#     for index, row in df.iterrows():
#         img_path = row['path']  # 'path' 대신에 실제 경로가 있는 컬럼명을 사용해주세요.

# import pandas as pd
# def concat_dataset_to_dataframe(concat_dataset):
#     data = []
#     for idx in range(len(concat_dataset)):
#         data.append(concat_dataset[idx])  # 데이터 추출
#     # 각 항목이 딕셔너리 형태로 되어 있다면 pandas DataFrame으로 변환
#     data.
#     df = pd.DataFrame(data)
#     return df