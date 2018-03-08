import glob
import os
import shutil
import PIL
import pandas as pd
import  random

CSV_FILE='Data_Entry_2017.csv'

df=pd.read_csv(CSV_FILE,index_col = False)
patient_frame=pd.Series.unique(df["Patient ID"])
train_df= pd.DataFrame( columns=df.columns)
test_df= pd.DataFrame(columns=df.columns)
validation_df= pd.DataFrame(columns=df.columns)

data_distribution={0:train_df,1:test_df,2:validation_df}

curr_patient=-1
for index, row in df.iterrows():
    if curr_patient !=row["Patient ID"]:
        curr_patient=row["Patient ID"]
        pr=random.random()
        if pr<=0.7:
            ds_set=0
        elif pr<=0.9:
            ds_set=1
        else:
            ds_set=2
    if index%500==0:
        print("current index",index)

    data_distribution[ds_set]= data_distribution[ds_set].append(row)


data_distribution[0].to_csv('train.csv',index=False)
data_distribution[1].to_csv('test.csv',index=False)
data_distribution[2].to_csv('validation.csv',index=False)




