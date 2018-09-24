# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 22:41:59 2018

@author: Florian
"""

import os
import tensorflow as tf


def create_model(model_download_folder,save_model_folder):
	from tensorflow.python.tools import freeze_graph
	import Extract_Model_Info
	import object_detection.export_inference_graph

#    	"""This function checks out the CNN model to a folder
#    
#    	Args:
#    		model_download_folder: folder download from tensorflow model zoo
#    		save_model_folder: Folder where .pb should be saved too
#    
#    	Returns:
#    		none
#    
#    	Source:
#    		https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md     
#
#    	"""

    #-------------------------------------
    # check for files
    #-------------------------------------
    files = {}
    for file in os.listdir(model_download_folder):
        if file.endswith('.config'):
            files['config'] = os.path.join(model_download_folder,file)
        if file.find('ckpt') != -1:
            if not 'checkpoint' in files:
                files['checkpoint'] =os.path.join(model_download_folder,file.split(".")[0]+'.'+file.split(".")[1])
    # checkpoint files needs to be checked
    if not os.path.isfile(files['checkpoint']+'.data-00000-of-00001'):
        raise ValueError('.data-00000-of-00001 checkpointfile is missing')
    if not os.path.isfile(files['checkpoint']+'.index'):
        raise ValueError('.index checkpointfile is missing')
    if not os.path.isfile(files['checkpoint']+'.meta'):
        raise ValueError('.meta checkpointfile is missing')

    #-------------------------------------
    # create flags to input for export_inference_graph
    #-------------------------------------
    flags = tf.app.flags.FLAGS
    flags.input_type = "image_tensor"
    flags.pipeline_config_path = files['config']
    flags.trained_checkpoint_prefix = files['checkpoint']
    flags.output_directory = save_model_folder
    
    #-------------------------------------
    # models\research\object_detection\export_inference_graph.py
    #-------------------------------------    
    object_detection.export_inference_graph.main(None)
    
    #-------------------------------------
    # Create Model info files
    #-------------------------------------
    Extract_Model_Info.main(save_model_folder,"frozen_inference_graph.pb")


if __name__ == "__main__":
    #-------------------------------------
    # Define Inputs
    #-------------------------------------
    #Download folder form model zoo
    #https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
    folder = "C:/Users/Florian/Documents/Tensorflow/models_download/"
    model = "faster_rcnn_inception_v2_coco_2018_01_28"
    model_download_folder = os.path.join(folder,model)
    #name under witch model should be saved
    model_name = "faster_rcnn_inception_v2_coco"
    #Save models
    #model_output_dir = os.path.join(os.getcwd(),"object_detection/")
    model_output_dir="C:/Users/Florian/Documents/Tensorflow/workcenter/mouse_detection_model/"
    if os.path.exists(model_output_dir):
        model_output_folder= os.path.join(model_output_dir,model_name)
        if not os.path.exists(model_output_folder):
            os.makedirs(model_output_folder)
    else:
        raise ValueError("Directory to save model folder not present")
    create_model(model_download_folder,model_output_folder)
#    if (checkpoint.find(input_modelfile.split(".",1)[0])>-1):
#        #print("Nothing to do")
#        freeze_model(input_modelfile,checkpoint_path,model_folder)
#    else:
#        raise ValueError("Model does not correspond to Checkpoint")