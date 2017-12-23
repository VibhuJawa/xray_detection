import pandas as pd
import os
import  cv2


train_CSV_FILE='train.csv'
train_folder_loc='data/train/'
train_3_channel_loc='data/train_3_channel/'

test_CSV_FILE='test.csv'
test_folder_loc='data/test/'
test_3_channel_loc='data/test_3_channel/'

val_CSV_FILE='validation.csv'
val_folder_loc='data/val/'
val_3_channel_loc='data/val_3_channel/'

root=val_folder_loc
save_3_channel_loc=val_3_channel_loc
df=pd.read_csv(val_CSV_FILE,index_col = False)

for index, row in df.iterrows():
     name = row['Image Index']
     loc = os.path.join(root, name)
     loc2= os.path.join(save_3_channel_loc,name)
     img = cv2.imread(loc)
     cv2.imwrite(loc2, img)

     if index%100==0:
          print("current index ",index)