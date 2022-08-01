from operator import matmul
import os
import glob
#from tkinter import image_names
from img2vec_pytorch import Img2Vec
from PIL import Image
import numpy as np
import time
import sys
import shutil
import pandas as pd
import math

class generate_coreset:

    def __init__(self, img_embeddings=None, img_list=None,label_map=None):
        self.img_embeddings = img_embeddings if img_embeddings is not None else []
        self.img_list = img_list if img_list is not None else []
        self.label_map = label_map if label_map is not None else []
        self.coreset_sample_indexes = []

    def generate_embeddings(self,data_path,embeddings_path,img_name_list_path,label_map_path):
        '''
        '''
        print("--Generating embeddings")
        img2vec = Img2Vec()
        files  = glob.glob(data_path+'/**/*')

        for file in files:
            self.img_list.append(file)
            self.label_map.append(os.path.basename(os.path.dirname(file)))
            img = Image.open(file).convert('RGB')
            vec = img2vec.get_vec(img)
            self.img_embeddings.append(vec)
        
        np.save(embeddings_path,np.array(self.img_embeddings))
        np.save(img_name_list_path,np.array(self.img_list))
        np.save(label_map_path,np.array(self.label_map))
        
        self.img_embeddings = np.array(self.img_embeddings)

        return

    def cosine_coreset(self,threshold):
        print("\n--Computing cosine similarities ")
        start_time = time.time()
        img_embeddings_transpose = np.transpose(self.img_embeddings)
        dot_prod = matmul(self.img_embeddings, img_embeddings_transpose)
        norm_img_embeddings = np.apply_along_axis(np.linalg.norm, 1, self.img_embeddings)
        norm_img_embeddings = np.expand_dims(norm_img_embeddings, axis = 1)
        norm_img_embeddings_transpose = np.transpose(norm_img_embeddings)
        dot_prod = np.divide(dot_prod, norm_img_embeddings)
        dot_prod = np.divide(dot_prod, norm_img_embeddings_transpose)
        print("Time taken to compute similarities: {} seconds".format(time.time() - start_time))
        
        print("\n--Computing coreset samples ")
        start_time_2 = time.time()
        upper_only = np.triu(np.ones(dot_prod.shape) - np.identity(dot_prod.shape[0]))
        dot_prod = dot_prod * upper_only
        above_threshold = dot_prod > threshold

        to_discard_set = set()
        incorrect_similar_counter = [0,0]
        for i in range(above_threshold.shape[0]):
            similar_list = list(np.where(above_threshold[i]==True)[0])
            if len(similar_list)>0:
                representative_label = self.label_map[i]
                similar_label_list = list(map(self.label_map.__getitem__, similar_list))
                
                if len(set(similar_label_list))>1 or representative_label not in set(similar_label_list):
                    incorrect_similar_counter[0] += 1
                    incorrect_similar_counter[1] += len(similar_label_list)
                to_discard_set.update(similar_list)
                above_threshold[:,similar_list] = False
                above_threshold[similar_list,:] = False

        self.coreset_sample_indexes = set(np.arange(above_threshold.shape[0])) - to_discard_set
        print("Time taken to compute representatives (coreset samples): {} seconds".format(time.time() - start_time_2))
        print("\nNumber of training samples before coreseting: ", len(self.img_list))
        print("Number of training samples after coreseting: ", len(self.coreset_sample_indexes))
        print("Percentage Reduction: ",np.around((1-len(self.coreset_sample_indexes)/len(self.img_list))*100,2))


        return self.coreset_sample_indexes

    def lsh_coreset(self,lsh_bucket_density):
        start_time = time.time()
        hash_mat = []
        num_unit_vector = math.floor(math.log(len(self.img_list)))
        print("\n--Computing coreset samples ")
        for i in range(num_unit_vector):
            v = np.random.rand(self.img_embeddings.shape[1])
            v_hat = v / np.linalg.norm(v)
            hash = np.floor(matmul(self.img_embeddings, v_hat)/lsh_bucket_density)
            # hash_val = hash[0] * 10+ hash[1]
            hash_mat.append(list(hash))

        hash_mat = np.transpose(np.array(hash_mat))
        hash_mat_df = pd.DataFrame(hash_mat)

        duplicated_df = hash_mat_df[hash_mat_df.duplicated(keep=False)]
        duplicated_df = duplicated_df.groupby(list(duplicated_df)).apply(lambda x: tuple(x.index)).tolist()
        duplicates = list(duplicated_df)
        duplicates_flatten = [item for sublist in duplicates for item in sublist]
        all_indices = list(range(len(hash_mat)))
        unique_indices = list(set(all_indices) - set(duplicates_flatten))
        relevant_indices = [dup[0] for dup in duplicates]

        for index in unique_indices:
            self.coreset_sample_indexes.append(index)

        for index in relevant_indices:
            self.coreset_sample_indexes.append(index)
        
        print("Time taken to compute representatives (coreset samples) using LSH: {} seconds".format(time.time() - start_time))
        print("\nNumber of training samples before coreseting: ", len(self.img_list))
        print("Number of training samples after coreseting: ", len(self.coreset_sample_indexes))
        print("Percentage Reduction: ",np.around((1-len(self.coreset_sample_indexes)/len(self.img_list))*100,2))

        return self.coreset_sample_indexes

    def save_coreset(self,coreset_params):
        labels = coreset_params["labels"]

        if coreset_params["method"] == "cosine":
            dest_folder =  os.path.join(coreset_params['dest_folder'],"train_"+coreset_params["method"]+"_"+str(coreset_params["cosine_sim_threshold"]))
        else: 
            dest_folder =  os.path.join(coreset_params['dest_folder'],"train_"+coreset_params["method"]+"_"+str(coreset_params["lsh_bucket_density"]))

        if os.path.exists(dest_folder) and os.path.isdir(dest_folder):
            shutil.rmtree(dest_folder)
        os.makedirs(os.path.join(dest_folder), mode = 0o777)

        if len(labels) != 0:
            for label in labels:
                os.makedirs(os.path.join(dest_folder,label), mode = 0o777)

        for index in self.coreset_sample_indexes:
            image_file = self.img_list[index]

            if len(labels) == 0:
                shutil.copy(image_file,image_file.replace(os.path.dirname(image_file),
                                                        dest_folder)) 
            else:
                shutil.copy(image_file,image_file.replace(os.path.dirname(os.path.dirname(image_file)),
                                                        dest_folder)) 
            # sys.exit()
        print("\nImages dumped at ",dest_folder)
        print("\n")                                       
        return dest_folder