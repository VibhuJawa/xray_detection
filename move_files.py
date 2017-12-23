import pandas as pd
import os
import PIL
from PIL import Image

import  random

# train_CSV_FILE='test.csv'
# train_folder_loc='data/train/'


# test_CSV_FILE='test.csv'
# test_folder_loc='data/test/'

val_CSV_FILE='validation.csv'
val_folder_loc='data/val/'


root="data/all_data/all_images"
df=pd.read_csv(val_CSV_FILE,index_col = False)
print(df)

for index, row in df.iterrows():
     name = row['Image Index']
     loc = os.path.join(root, name)
     img = Image.open(loc)
     img = img.resize((256,256), PIL.Image.ANTIALIAS)
     if index%100==0:
          print("current index ",index)
     img.save(val_folder_loc+name)

