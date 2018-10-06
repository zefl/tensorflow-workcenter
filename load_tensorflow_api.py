# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 20:56:40 2018

This skript imports all the sys path varaibeles to the current conda environement

@author: Florian
"""
#put api to PYTHONPATH
import sys
import os
import subprocess
import git

def load_API(tensorflow_models_folder):
    #Add path to system
    print('Loading and installing tensorflow models')
    
    if not os.path.exists(tensorflow_models_folder):
        os.makedirs(tensorflow_models_folder)
        git.Repo.clone_from("https://github.com/tensorflow/models.git",tensorflow_models_folder)
        
    if os.path.exists(tensorflow_models_folder):
        print("Check Protos")
        relative_protobuf_folder = "research/object_detection/protos"
        protobuf_folder = os.path.join(tensorflow_models_folder,relative_protobuf_folder)
        if os.path.exists(protobuf_folder):
            protobuf_not_combiled = True
            #check if protobuf is already combiled on sytem
            for fname in os.listdir(protobuf_folder):
                if fname.endswith('.py'):
                    protobuf_not_combiled = False
    		
            #combile Protobuf
            if protobuf_not_combiled:
                run_folder = os.path.join(tensorflow_models_folder,'research')
                #test = runcmd(['cd /' + tensorflow_models_folder;'dir'])
                for fname in os.listdir(protobuf_folder):
                    if not fname.endswith('.py'):
                        fname = os.path.join('object_detection/protos',fname)
                        #compile Protos one by one 
                        proc  = subprocess.Popen(['"C:/Users/Florian/Documents/Tensorflow/tensorflow_api/protoc-3.5.1-win32/bin/protoc.exe"',fname,'--python_out=.'], cwd=run_folder,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                outs, errs = proc.communicate()
                if errs:
                    raise ValueError(errs)
            else:
                print("Protobuf already compiled")
            print("Compile Protobuf complete")
        else:
            raise ValueError("No Protobuf folder")
    else:
        raise ValueError(tensorflow_models_folder + "does not exist possible error in checkput")
	
    if os.path.exists(tensorflow_models_folder):
        add_folders = ["","research","research/slim","research/object_detection"]
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
        raise ValueError(tensorflow_models_folder + "does not exist")


if __name__ == "__main__":
    _tensorflow_models_folder = "C:/Users/Florian/Documents/Tensorflow/tensorflow_api/models/"
    load_API(_tensorflow_models_folder)