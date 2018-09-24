# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 20:56:40 2018

This skript imports all the sys path varaibeles to the current conda environement

@author: Florian
"""
#put api to PYTHONPATH
import sys
import os
tensorflow_models_folder = "C:/Users/Florian/Documents/Tensorflow/tensorflow_api/models/"

add_folders = ["","research","research/slim","research/object_detection"]

if os.path.exists(tensorflow_models_folder):
    for add_folder in add_folders:
        already_added = False
        for item in sys.path:
            new_path = os.path.join(tensorflow_models_folder,add_folder)
            if item == new_path:
                already_added = True
            #end if
        #end for
        if not already_added:
            sys.path.append(new_path)
            print("Add path " + new_path)
        #end if
    #end for
    print(sys.path)  
else:
    raise ValueError("C:/Users/Florian/Documents/Tensorflow/project_folder/tensorflow_api/models/ does not exist")
