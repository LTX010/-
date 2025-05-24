import os
import random
import json
import hashlib
from shutil import copy2

# 配置文件参数
src_path = r'E:\python_professional\torchLearn\花卉识别\flower7595\flowers'
dst_path = r'E:\python_professional\torchLearn\花卉识别\flower7595\split_data'
split_ratio = [0.8, 0.2]
valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}

# 生成类别映射
classes = sorted([d for d in os.listdir(src_path)
                  if os.path.isdir(os.path.join(src_path, d))])
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
with open('class_to_idx.json', 'w') as f:
    json.dump(class_to_idx, f)

# 创建目标目录
for split in ['train', 'test']:
    for cls in os.listdir(src_path):
        os.makedirs(os.path.join(dst_path, split, cls), exist_ok=True)

# 改进后的数据划分（添加MD5校验）
md5_dict = {}
for cls in classes:
    cls_path = os.path.join(src_path, cls)
    images = [f for f in os.listdir(cls_path)
              if os.path.splitext(f)[1].lower() in valid_ext]
    random.shuffle(images)

    # 划分与复制（略）
    # 在copy2之后添加：
    md5_dict[img] = get_md5(dst)


# 改进后的标签保存
def save_labels(data_root, output_txt):
    with open(output_txt, 'w') as f:
        for cls in os.listdir(data_root):
            cls_dir = os.path.join(data_root, cls)
            if not os.path.isdir(cls_dir):
                continue

            for img in os.listdir(cls_dir):
                if os.path.splitext(img)[1] not in valid_ext:
                    continue

                abs_path = os.path.join(cls_dir, img)
                rel_path = os.path.relpath(abs_path,
                                           start=os.path.dirname(output_txt))
                f.write(f"{rel_path} {class_to_idx[cls]}\n")


save_labels(os.path.join(dst_path, 'train'), "train_labels.txt")
save_labels(os.path.join(dst_path, 'test'), "test_labels.txt")