from tqdm import tqdm
import os

def convert_and_rename(path):
    print(path)
    for file_name in tqdm(os.listdir(path)):
        if file_name.endswith('x4.png'):
            old_path = os.path.join(path, file_name)
            new_path = os.path.join(path, file_name.replace('x4', ''))
            os.rename(old_path, new_path)
convert_and_rename('/root/share/DIV2K_valid_LR_bicubic_X4/DIV2K_valid_LR_bicubic/X4') #DF2K 3450zhang