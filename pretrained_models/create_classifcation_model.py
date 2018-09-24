# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 22:41:59 2018

@author: Florian
"""

import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import Extract_Model_Info
import slim.export_inference_graph


def create_model(_model_file,_save_model_folder):
#    	"""This function checks out the CNN model to a folder
#    
#    	Args:
#    		_model_file: 	Name of the Klassifiaction model see Tensoflow models\research\slim\nets\nets_factory.py
#    						nameofmodle.pb
#    		_save_model_folder: Folder where .pb should be saved too
#    
#    	Returns:
#    		none
#    
#    	"""
    #-------------------------------------
    # create folder for model
    #-------------------------------------
    if not os.path.exists(_save_model_folder):
        os.makedirs(_save_model_folder)
    	
    	#-------------------------------------
    # create flags to input for export_inference_graph
    #-------------------------------------
    flags = tf.app.flags.FLAGS
    flags.output_file = os.path.join(_save_model_folder,_model_file)
    flags.model_name = _model_file.split(".",1)[0]
    flags.dataset_dir = _save_model_folder
    
    #-------------------------------------
    # models\research\slim\export_inference_graph.py
    #-------------------------------------    
    slim.export_inference_graph.main(None)
    
    #-------------------------------------
    # Create Model info files
    #-------------------------------------
    Extract_Model_Info.main(_save_model_folder,_model_file)
    

def freeze_model(_model_file,_checkpoint_file,_save_model_folder):
#    """This function freeze a checkpoint file to a model .pb file
#    
#    	Args:
#    		_model_file: Name of the Klassifiaction model see Tensoflow models\research\slim\nets\nets_factory.py
#    		_checkpoint_file: 	location of checkpoint file 
#    							checkpoint_file.ckpt
#    		_save_model_folder:	folder where cnn model and model infos are located
#    
#    	Returns:
#    		none
#    
#    	"""
	
    #-------------------------------------
    # Define variablen for freeze_graph
    #-------------------------------------
    input_graph_filename=os.path.join(_save_model_folder,_model_file)
    input_saver_def_path=""                  
    input_binary=True 
    checkpoint_path=_checkpoint_file
    output_node_names="none"
    with open(os.path.join(_save_model_folder,_model_file.split(".",1)[0])+"_Layers.txt",'r') as txt:
        layer = txt.readlines()
        output_node_names  =layer[-1].strip('\n')                 
    restore_op_name = None
    filename_tensor_name= None                   
    output_graph_filename=os.path.join(_save_model_folder,_model_file.split(".",1)[0] + "_frozen."+ _model_file.split(".",1)[1])
    clear_devices=False
	
    #-------------------------------------
    # Use freeze graph to freeze checkpoint to model
    # function is defined in tensorflow tensorflow\python\tools\freeze_graph.py
    #-------------------------------------
    freeze_graph.freeze_graph(input_graph_filename, 
                              input_saver_def_path,
                              input_binary, 
                              checkpoint_path, 
                              output_node_names,
                              restore_op_name, 
                              filename_tensor_name,
                              output_graph_filename, 
                              clear_devices,
                              "")


if __name__ == "__main__":
    #-------------------------------------
    # Define Inputs
    #-------------------------------------
    #Name of model
    #Model List see .\tensorflow_api\models\research\slim\nets\nets_factory.py
    input_modelfile = "inception_v3.pb"
    #Save models
    input_models_folder = "C:/Users/Florian/Documents/Tensorflow/workcenter/model/classification/"
    #Checkpoint folder
    input_checkpoint_folder = "C:/Users/Florian/Documents/Tensorflow/models_download/inception_v3/"
    #Checkpoint file
    input_checkpoint_file = "inception_v3.ckpt"
    #-------------------------------------
    checkpoint = input_checkpoint_folder+input_checkpoint_file
    #Model Folder to save model
    model_folder = os.path.join(input_models_folder,input_modelfile.split(".",1)[0])
    #full checkpoint path
    checkpoint_path = os.path.join(input_checkpoint_folder,input_checkpoint_file) 
    create_model(input_modelfile,model_folder)
    if (checkpoint.find(input_modelfile.split(".",1)[0])>-1):
        #print("Nothing to do")
        freeze_model(input_modelfile,checkpoint_path,model_folder)
    else:
        raise ValueError("Model does not correspond to Checkpoint")