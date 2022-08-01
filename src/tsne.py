import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pylab as plt
from coreset import generate_coreset
import json
import os


def tsne(img_array, labels):
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(img_array)
    df = pd.DataFrame()
    df["labels"] = labels
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    df['IntegerLabels'] = pd.factorize(df['labels'])[0] + 1
    return df



def plot(df, cfg):
    plt.scatter(x=df['comp-1'], y=df['comp-2'], c=df['IntegerLabels'], cmap='viridis')
    cbar = plt.colorbar()
    cbar.set_label('Heritage')
    ax = plt.gca()
    ax.set_xlabel('comp-1')
    ax.set_ylabel('comp-2')
    plt.savefig(cfg['tsne_params']['output_path'])
    

    

def main():
    config_path = "../config/config.json"
    cfg = json.load(open(config_path,'r'))
    if not os.path.exists(cfg['coreset_params']['embeddings_path']) or not os.path.exists(cfg['coreset_params']['img_name_list_path']) or not os.path.exists(cfg['coreset_params']['label_map_path']):
        coreset_obj = generate_coreset()
        coreset_obj.generate_embeddings(cfg['data_dir']['train'],cfg['coreset_params']['embeddings_path'],cfg['coreset_params']['img_name_list_path'],cfg['coreset_params']['label_map_path'])
    else:
        print("--Reading saved embeddings")
        img_array = np.load(cfg['coreset_params']['embeddings_path'])
        img_list = list(np.load(cfg['coreset_params']['img_name_list_path']))
        labels = list(np.load(cfg['coreset_params']['label_map_path']))
        coreset_obj = generate_coreset(img_array, img_list,labels)

    tsne_df = tsne(img_array, labels)
    plot(tsne_df, cfg)



if __name__ == "__main__":
    main()




