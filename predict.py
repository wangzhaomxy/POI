import argparse
from importlib import import_module
from utils import *
from train_eval import infer

classes_reverse = {0:"居住用地", 1:"商业服务业设施用地", 2:"公共管理与公共服务用地", 3:"公用设施用地", 4:"工业用地"}
root = "./poi_data"
csv_path = "./poi_data/hankou_POI_predict.csv"
link_txt = "./poi_data/link.txt"
model_path = "./poi_data/saved_dict/bert.ckpt"
predict_save_path = "./poi_data/pred.csv"
predict_save_path_new = "./poi_data/pred_id_label.csv"

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert')
args = parser.parse_args()

if __name__ == '__main__':
    dataset_path = csv_path  # the dataset folder name under the same document with this python script.
    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(root) # See and change the config details in bert.py under the models folder.

    raw_data, new_data = process_csv_data(dataset_path, link_txt)
    infer_data = load_dataset(config, link_txt)
    pred_iter = build_iterator(infer_data, config, with_label=False)    

    model = x.Model(config).to(config.device)
    
    predicts = infer(model=model, data_iter=pred_iter, ckpt=model_path)

    print("Saving files")
    post_porcess(raw_data=raw_data,
                 new_data=new_data,
                 pred=predicts,
                 classes=classes_reverse,
                 save_raw=predict_save_path,
                 save_new=predict_save_path_new)
    print("Inference process DONE!")
    
    
    

    
    