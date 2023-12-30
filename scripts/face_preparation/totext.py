import os

def save_file_names_to_txt(hr_img_dir, txt_file):
   with open(txt_file, 'w') as f:
       for file in os.listdir(hr_img_dir):
           if file.endswith('.jpg'):
                f.write(file+" 2" + '\n')

if __name__ == "__main__":
   hr_img_dir = "E:\\PyProject\\data\\classical_SR_datasets\\CelebA-HQ_ParsingMap\\inference\\GTmod128"
   txt_file = "val12.30_celeba.txt"
   save_file_names_to_txt(hr_img_dir, txt_file)
