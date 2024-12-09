import pandas as pd
from utils import *

# identify the constant variables
classes = {"居住用地":0, 
           "商业服务业设施用地":1, 
           "公共管理与公共服务用地":2, 
           "公用设施用地":3, 
           "工业用地":4}
csv_path = "./poi_data/hankou_POI_train.csv"
save_path = "./poi_data/data/"
csv_col = ["TARGET_FID", "type_", "name","土地利用"]

# read .csv file, and choose the helpful colomns, and then alter the labels to 
# numbers.
data_poi = pd.read_csv(csv_path,usecols=csv_col)
data_poi["concat"]=data_poi["type_"]+";"+data_poi["name"]
data_poi["label"]= data_poi["土地利用"].map(classes)
data_poi = data_poi[["TARGET_FID","concat","label"]]

### clean the error labeled data and check the results.
data_poi = clean_error_label(data_poi)

### group the grid point and concatenate the words then discard the duplicated one.
new_data = organize_cols(data_poi,
                         group_col=["TARGET_FID","label"],
                         cat_col="concat",
                         reserve_col=["concat","label"])

### split the datasets into train, val, test datasets, with the proportion of 
### 80:10:10, and then save to .txt files.
X, y = new_data["concat"], new_data["label"]
# split datasets
X_train, y_train, X_val, y_val, X_test, y_test = data_split(X, y, ratio=(0.8, 0.1, 0.1))

### Data augmentations
X_train, y_train = augment_data(X_train, y_train, dup_times=20)

# save datasets
train_path = save_path + "train.txt"
val_path = save_path + "dev.txt"
test_path = save_path + "test.txt"

print("Saving Data ...")
save_txt(X_train, y_train, train_path)
save_txt(X_val, y_val, val_path)
save_txt(X_test, y_test, test_path)

print("Data preprocessing Done!")