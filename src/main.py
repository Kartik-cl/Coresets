
from coreset import generate_coreset
import json
import os
import numpy as np
import sys
from train_model import create_model

def main():

    config_path = "./config/config.json"
    cfg = json.load(open(config_path,'r'))

    data_dir = cfg['data_dir']
    if not cfg['generate_coreset']:
        print("--Not generating coreset proceeding with training")

    else:

        if not os.path.exists(cfg['coreset_params']['embeddings_path']) or not os.path.exists(cfg['coreset_params']['img_name_list_path']) or not os.path.exists(cfg['coreset_params']['label_map_path']):
            
            if not os.path.exists(os.path.dirname(cfg['coreset_params']['embeddings_path'])):
                os.makedirs(os.path.dirname(cfg['coreset_params']['embeddings_path']), mode = 0o777)

            coreset_obj = generate_coreset()
            coreset_obj.generate_embeddings(cfg['data_dir']['train'],cfg['coreset_params']['embeddings_path'],cfg['coreset_params']['img_name_list_path'],cfg['coreset_params']['label_map_path'])
            #exit(1)
        else:
            
            print("--Reading saved embeddings")
            img_array = np.load(cfg['coreset_params']['embeddings_path'])
            img_list = list(np.load(cfg['coreset_params']['img_name_list_path']))
            label_map = list(np.load(cfg['coreset_params']['label_map_path']))

            coreset_obj = generate_coreset(img_array, img_list,label_map)

        if cfg["coreset_params"]["method"] == "cosine":
            coreset_image_indexes = coreset_obj.cosine_coreset(cfg["coreset_params"]["cosine_sim_threshold"])

        elif cfg["coreset_params"]["method"] == "lsh":
            coreset_image_indexes = coreset_obj.lsh_coreset(cfg["coreset_params"]["lsh_bucket_density"])
        
        else:
            print("-- Invalid coreset generation method in config. Exiting")
            sys.exit()

        coreset_images_dir = coreset_obj.save_coreset(cfg['coreset_params'])

        data_dir =  cfg['data_dir']
        data_dir["train"] = coreset_images_dir

        print(data_dir)
    #==== Model Training code structure

    # model_obj = create_model([cfg['experiment_name'], cfg['model_arch'], cfg['num_classes'],data_dir , cfg['train_params']])
    # model_obj.make_model()

if __name__ == "__main__":
    print("*"*70)
    main()
