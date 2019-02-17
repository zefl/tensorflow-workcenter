# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 20:56:40 2018

This skript imports all the sys path varaibeles to the current conda environement

@author: Florian
"""
#########################################################################	
#    Description: 
#						set up tensorflow object detection API 
#                       if protobuf is installed on system you can use the cmd
#                       if not download exe from link
#                       skript can also download tensorflow API form git if not present
#    Class Variables: 
#						none
#    Input Args:
#						tensorflow_models_folder --> Contains tensorflow Object detection API from
#                                                    https://github.com/tensorflow/models        
#          				protoc_exe --> location of protoc exe 
#                                      https://github.com/protocolbuffers/protobuf/releases
#    Output Args: 
#						none
#########################################################################
import sys
import os
import subprocess
import git

def prepare_object_detection_API(tf_models_folder,protoc_exe=""):
    #Add path to system
    print('Loading and installing tensorflow models')
    
    if not os.path.exists(tf_models_folder):
        print('Checkout TF Repo from git')
        os.makedirs(tf_models_folder)
        git.Repo.clone_from("https://github.com/tensorflow/models.git",tf_models_folder)
        
    if os.path.exists(tf_models_folder):
        print("Check Protos")
        relative_protobuf_folder = "research/object_detection/protos"
        protobuf_folder = os.path.join(tf_models_folder,relative_protobuf_folder)
        if os.path.exists(protobuf_folder):
            protobuf_not_combiled = True
            #check if protobuf is already combiled on sytem
            for fname in os.listdir(protobuf_folder):
                if fname.endswith('.py') and fname != "__init__.py":
                    protobuf_not_combiled = False
                #endif
            #endfor
    		
            #combile Protobuf
            if protobuf_not_combiled:
                run_folder = os.path.join(tf_models_folder,'research')
                #test = runcmd(['cd /' + tensorflow_models_folder;'dir'])
                for fname in os.listdir(protobuf_folder):
                    if not fname.endswith('.py'):
                        fname = os.path.join('object_detection/protos/',fname)
                        print(run_folder)
                        #switch between cmd and exe
                        if protoc_exe:
                            if os.path.isfile(protoc_exe) and os.access(protoc_exe, os.X_OK):
                                cmd  = protoc_exe + " "+ str(fname) + ' --python_out=.'
                            else:
                               raise ValueError("No protoc exe found with give path: " + protoc_exe) 
                        else:
                            cmd = 'protoc ' + str(fname) + ' --python_out=.'
                        #endif
                        print("Run command: " + cmd)
                        #compile Protos one by one
                        proc  = subprocess.Popen(cmd,cwd=str(run_folder), shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                        outs, errs = proc.communicate()
                        if errs:
                            raise ValueError(errs)
                        #endif
                    #endif
                #endfor
            else:
                print("Protobuf already compiled")
            #endif
            print("Compile Protobuf complete")
        else:
            raise ValueError("No Protobuf folder")
        #endif
    else:
        raise ValueError(tf_models_folder + "does not exist possible error in checkput")
    #endif

#########################################################################	
#    Description: 
#						load API into current python kernel
#    Class Variables: 
#						none
#    Input Args:
#						tensorflow_models_folder --> Contains tensorflow Object detection API
#    Output Args: 
#						none
#########################################################################

def load_TF_object_decetion_API(tf_models_folder):
    sys.path.append(os.getcwd())	
    if os.path.exists(tf_models_folder):
        #add these folders to python kernel
        add_folders = ["","research","research/slim","research/object_detection"]
        for add_folder in add_folders:
            already_added = False
            for item in sys.path:
                new_path = os.path.join(tf_models_folder,add_folder)
                #check if already added
                if item == new_path:
                    already_added = True
                #end if
            #end for
            if not already_added:
                #add to kernel
                sys.path.append(new_path)
                print("Add path " + new_path)
            #end if
        #end for
        #print(sys.path)  
    else:
        raise ValueError(tf_models_folder + "does not exist")


if __name__ == "__main__":
    tensorflow_models_folder = "C:/Users/lofl1011/Documents/workcenter/tensorflow_api/models/"
    protoc_exe = "C:/Users/lofl1011/Documents/workcenter/protoc/bin/protoc.exe"
    prepare_object_detection_API(tensorflow_models_folder,protoc_exe)
    load_TF_object_decetion_API(tensorflow_models_folder)