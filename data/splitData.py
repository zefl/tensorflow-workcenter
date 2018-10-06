# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 13:40:20 2018

This function takes a Zip folder and creats folders train,test, validation for training

@author: Florian
"""
import os

def unzip_data(dir_path=""):
    import zipfile
    #dirname gets direction name of full file path of current script
    #realpath replaces / with \
    zip_files = []
    lable_names = []
    if not dir_path:
        dir_path = os.path.dirname(os.path.realpath(__file__))
    for fname in os.listdir(dir_path):
        if fname.endswith('.zip'):
            zip_files.append(os.path.join(dir_path,fname))
            lable_names.append(fname.split('.zip')[0])
                
    #throw error if no file is found
    _raw_data_folder = []
    if not zip_files[0]:
        raise ValueError("There is no zip file in the current folder")
    else:
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                current_folder = os.path.join(dir_path,lable_names[zip_files.index(zip_file)]+"_raw_data")
                _raw_data_folder.append(current_folder)
                zip_ref.extractall(current_folder)
            
    return _raw_data_folder

def split_data(raw_data_folder,train_percentage=0.8,test_percentage=0.2,validation_percentage=0.0):
    import random
    import shutil
    if not os.path.exists(raw_data_folder):
        raise ValueError("raw_Data folder does not exist")
    if validation_percentage > 0.0:
        if train_percentage + test_percentage + validation_percentage != 1:
            raise ValueError("Percentage of data deviation is not 100 %")
    else:
        if train_percentage + test_percentage != 1:
            raise ValueError("Percentage of data deviation is not 100 %")
            
    files = []
    other_files = []
    for file in os.listdir(raw_data_folder):
        if file.endswith(('.png', '.jpg', '.jpeg','.JPG')):
            files.append(file)
        else:
            other_files.append(file)
    number_of_files = len(files)
    if number_of_files == 0:
         raise ValueError("No Images in folder")
    
    #data_numbers[train,test,validation]
    data_numbers=[0,0,0]
    data_numbers[1] = int(number_of_files * test_percentage)
    data_numbers[2] = int(number_of_files * validation_percentage)
    data_numbers[0] = number_of_files -  data_numbers[1] -  data_numbers[2]
    #data[train,test,validation]
    data ={}
    #get label from raw data folder
    label_folder = os.path.basename(raw_data_folder)
    label = label_folder.split('_raw_data',1)[0]
    if not label:
        raise ValueError('Raw data folder has no label')
    #cretae key for folder to split images to   
    keys = ['train','test','validation']
        
    for i in range(len(data_numbers)):
        data[keys[i]]= random.sample(files,data_numbers[i])
        for data_name in data[keys[i]]:
            files.remove(data_name)
            
    for key in keys:
        if len(data[key]):
            folder = os.path.join(os.path.dirname(raw_data_folder),key)
            if not os.path.exists(folder):
                #remove folder if it exists
                os.mkdir(folder)
            folder = os.path.join(folder,label)
            if os.path.exists(folder):
                #remove folder if it exists
                shutil.rmtree(folder)
            os.mkdir(folder)
            for file in data[key]:
                file_source = os.path.join(raw_data_folder,file)
                file_dest = os.path.join(folder,file)
                shutil.copyfile(file_source,file_dest)
                for other_file in other_files:
                    if file.split('.')[0]==other_file.split('.')[0]:
                        file_source = os.path.join(raw_data_folder,other_file)
                        file_dest = os.path.join(folder,other_file)
                        shutil.copyfile(file_source,file_dest) 

    
if __name__ == "__main__":
   raw_datas = unzip_data()
   for raw_data in raw_datas:
       if not os.path.exists(raw_data):
           raise ValueError("Erro in unzipping file")
       else:
           split_data(raw_data,0.85,0.15)