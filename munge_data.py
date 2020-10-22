import os
import pandas as pd
import numpy as np
import ast 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil


DATA_PATH = 'ori_data'
OUTPUT_PATH = 'wheat_data'

def process_data(data,data_type = 'train'):
    for _,row in tqdm(data.iterrows(),total=len(data)):
        image_name = row['image_id']
        bounding_boxes = row['bbox']
        yolo_data = []
        for bbox in bounding_boxes:
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]
            x_center = x+w/2
            y_center = y+h/2
            # 这里需要将图像数据归一化处理（yolo需要的输入为归一化后的数据,且为浮点数）
            x_center /= 1024.0
            y_center /= 1024.0
            w /= 1024.0
            h /= 1024.0
            yolo_data.append([0,x_center,y_center,w,h])
        yolo_data = np.array(yolo_data)
        # 保存bbox的图片信息
        np.savetxt(
            os.path.join(OUTPUT_PATH,f'labels/{data_type}/{image_name}.txt'),
            yolo_data,
            fmt=['%d','%f','%f','%f','%f']
        )
        # 将目标图片文件保存到指定文件中
        shutil.copyfile(
            os.path.join(DATA_PATH,f'train/{image_name}.jpg'),
            os.path.join(OUTPUT_PATH,f'images/{data_type}/{image_name}.jpg'),
        )



if __name__ == "__main__":
    
    df = pd.read_csv(os.path.join(DATA_PATH,'train.csv'))
    # 将string of list 转成list数据
    df.bbox = df.bbox.apply(ast.literal_eval)
    # 利用groupby 将同一个image_id的数据进行聚合，方式为list进行，并且用reset_index直接转变成dataframe
    df = df.groupby(['image_id'])['bbox'].apply(list).reset_index(name = 'bbox')

    # 划分数据集
    df_train,df_val = train_test_split(df,test_size=0.2,random_state=42,shuffle=True)
    # 重设 index （这里数据被打乱，index改变混乱）
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    process_data(df_train,data_type='train')
    process_data(df_val,data_type='val')
