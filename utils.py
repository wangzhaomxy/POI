# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

PAD, CLS = '[PAD]', '[CLS]'  # padding stop words

def load_dataset(config, path):
    pad_size = config.pad_size
    contents = []
    with open(path, 'r', encoding='utf_8_sig') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content, label = lin.split('\t')
            token = config.tokenizer.tokenize(content)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)

            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append((token_ids, int(label), seq_len, mask))
    return contents
    
def build_dataset(config):    
    train = load_dataset(config, config.train_path)
    dev = load_dataset(config, config.dev_path)
    test = load_dataset(config, config.test_path)
    return train, dev, test

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, with_label=True):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # Ristrict the same batch size.
        if self.n_batches == 0:
            self.residue = True
        elif len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device
        self.with_label = with_label

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        if self.with_label:
            y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        else:
            y = None
        # culculate the padding size
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
    
        return (x, seq_len, mask), y


    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config, with_label=True):
    iter = DatasetIterater(dataset, config.batch_size, config.device, with_label=with_label)
    return iter


def get_time_dif(start_time):
    """Get running time"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def get_error_keys(data_frame, grid_id="TARGET_FID",label="label"):
    """
    This function is to get the error keys. Some grids have different labels.
    The grid ID is "TARGET_FID", and the label is "label"

    Args:
        data_frame (pd.dataframe): the pandas dataframe

    Returns:
        [list]: a list of error grid ID.
    """
    error_data = data_frame.groupby(grid_id)[label].nunique()
    error_tar_fid = []
    for key in error_data.keys():
        if error_data[key]>1:
            error_tar_fid.append(key)
    return error_tar_fid

def clean_err_label(data_frame, grid_id="TARGET_FID",label="label"):
    """
    A single grid_id should have unique label, while some grid_id have multiple
    labels. This function is to clean the error label, unify the label value as
    the most frequency value under the same grid_id.
    
    Args:
        data_frame (pd.DataFrame): The pandas dataframe.
        grid_id (str, optional): The grid ID column name. Defaults to "TARGET_FID".
        label (str, optional): The label column name. Defaults to "label".

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    error_tar_fid = get_error_keys(data_frame)
    for err_id in error_tar_fid:
        data_frame.loc[data_frame[grid_id]==err_id, label] = data_frame[
            data_frame[grid_id]==err_id][label].value_counts().keys()[0]
    return data_frame

def check_error_label(data_frame):
    """
    A single grid_id should have unique label, while some grid_id have multiple
    labels. This function is to count the anormaly label number and print out, 
    and there is no return.
    Args:
        data_frame (pd.DataFrame): The dataset in pd.DataFrame format.
    """
    error_tar_fid = get_error_keys(data_frame)
    print("The error key number: ", len(error_tar_fid))

def clean_error_label(df, grid_id="TARGET_FID",label="label"):
    """
    A single grid_id should have unique label, while some grid_id have multiple
    labels. This function is to clean the error label, unify the label value as
    the most frequency value under the same grid_id. Before and after the data
    cleaning process, print the error number of the dataset.
    
    Args:
        data_frame (pd.DataFrame): The pandas dataframe.

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """        
    print("Before data clean ...")
    # print the number of error data rows.
    check_error_label(df)
    # clean data
    df = clean_err_label(df, grid_id="TARGET_FID",label="label")
    print("After data clean ... ")
    check_error_label(df)
    return df

def organize_cols(df,group_col, cat_col, reserve_col):
    """
    This function is to groupby the df dataset by group_col, and then process 
    the cat_col to merge the words and split them with space. Finally return
    the reserved columns.

    Args:
        df (pd.DataFrame): The object dataset dataframe.
        group_col (list): A list of group by col names, e.g., ["col1", "col2"]
        cat_col (str): the prospected processed concatenated columns.
        reserve_col (list): A list of reserved col names, e.g., ["col1", "col2"]

    Returns:
        (pd.DataFrame): The dataframe with the reserved columns.
    """
    new_data = df.groupby(group_col)
    new_data = new_data[cat_col].apply(lambda x: " ".join(list(set(x.str.cat(sep=';').split(';'))))).reset_index()
    new_data = new_data[reserve_col]
    return new_data.drop_duplicates()

def augment_data(X, y, dup_times=10):
    """
    This function is to augment dataset. Repeat the dataframe "dup_times" times,
    then shuffle the content in the "concat" column. 

    Args:
        X (pd.DataFrame): The dataframe.
        y (pd.DataFrame): The dataframe.
        dup_times (int, optional): The duplicate times. Defaults to 10.

    Returns:
        pd.DataFrame: The augmented dataframe.
    """
    # duplicate data rows for dup_times times
    print("Augmenting data ...")
    df = pd.concat([X, y], axis=1)
    new_df = pd.DataFrame(np.repeat(df.values, dup_times, axis=0))
    new_df.columns = df.columns
    # random shift the concat data.
    new_df["concat"] = new_df["concat"].apply(lambda x: " ".join(random.sample(x.split(" "),len(x.split(" ")))))
    new_df = new_df.drop_duplicates()
    new_df = new_df.sample(frac=1)
    return new_df[X.name],new_df[y.name]

def data_split(X, y, ratio=(0.8, 0.1, 0.1)):
    """
    Split the X, y data into the ratio of ratio.

    Args:
        X (array.like): The input dataset.
        y (array.like): The label.
        ratio (tuple, optional): The split ratios. Defaults to (0.8, 0.1, 0.1).

    Returns:
        The splited datasets: X_train, y_train, X_val, y_val, X_test, y_test
    """
    print("Splitting data ...")
    X_train, X_mid, y_train, y_mid = train_test_split(
        X, y, test_size=1-ratio[0], stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_mid, y_mid, test_size=(ratio[2]/(ratio[1]+ratio[2])), stratify=y_mid)
    return X_train, y_train, X_val, y_val, X_test, y_test

def save_txt(X, y, path):
    """
    Save the X and y pd.DataFrame to .txt file in the path.

    Args:
        X (pd.DataFrame): The input dataset
        y (pd.DataFrame): The label dataset
        path (str): The path like string that the data to be saved to. 
                    E.g., xx/xx.txt
    """
    data = pd.concat([X, y], axis=1)
    data.to_csv(path,sep="\t",index=False,header=False, encoding='utf_8_sig')

def process_csv_data(csv_path, link_txt):
    """
    Preprocess the predicted raw .csv dataset. The new dataset include the grid
    center ID - "TARGET_FID", and concatenated "name" and "type" sentence column
    - "concat", and "label" column. Save "concat" and "label" to link txt files.

    Args:
        csv_path (str): A path-like strings. e.g. xxx/xxx/xx.csv
        link_txt (str): A path-like strings. e.g. xxx/xxx/xx.txt

    Returns:
        pd.DataFrame: raw_data in pd.DataFrame format.
    """
    raw_data = pd.read_csv(csv_path)
    data_poi = raw_data.copy()
    data_poi["concat"]=data_poi["type_"]+";"+data_poi["name"]
    data_poi = data_poi[["TARGET_FID","concat"]]

    new_data = data_poi.groupby(["TARGET_FID"])
    new_data = new_data["concat"].apply(lambda x: " ".join(list(set(x.str.cat(sep=';').split(';'))))).reset_index()
    new_data = new_data[["TARGET_FID","concat"]]
    new_data["label"] = "100"
    save_txt(new_data["concat"], new_data["label"], link_txt)
    return raw_data, new_data

def post_porcess(raw_data, new_data, pred, classes, save_raw, save_new):
    predicts = pd.DataFrame({"label":pred})
    predicts["label"] = predicts["label"].apply(lambda x: classes[x])    
    new_data["label"] = predicts
    final_df = pd.merge(raw_data, new_data, how="left", on="TARGET_FID")
    final_df["土地利用"] = final_df["label"]
    final_df = final_df.drop(["concat", "label"], axis=1)
    final_df.to_csv(save_raw, encoding='utf_8_sig')
    new_data.to_csv(save_new, encoding='utf_8_sig')
