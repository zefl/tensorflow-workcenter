# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 17:57:13 2018

@author: Florian
"""
import tensorflow as tf
import os

from tensorflow.core.framework import graph_pb2
from google.protobuf import text_format

#---------------
#Define Graph Name
#---------------
GraphType="classification"
GraphName="inception_v3_imagnet"
GrapheFile="classify_image_graph_def.pb"
#---------------
#Code
#---------------
Folder = os.path.join("C:/Users/Florian/Documents/Tensorflow/workcenter/model/",GraphType,GraphName)


def main(_Folder=Folder,_GrapheFile=GrapheFile):
    Model_Path = os.path.join(_Folder,_GrapheFile)
    _graphName=_GrapheFile.split('.',1)
    
    if _graphName[1]=="pb":
        #define graph 
        graph = tf.Graph()
        with graph.as_default():
            with tf.gfile.FastGFile(Model_Path,'rb') as model:
                #empty graph object
                graph_def = tf.GraphDef()
                #read graph model which is a binary file 
                #see https://www.tensorflow.org/extend/tool_developers/ 
                graph_def.ParseFromString(model.read())
                #------------------------------
                #   write Model data to file
                #------------------------------
                with open(_Folder + "/" + _GrapheFile.split(".",1)[0]+'.txt','w') as txt:
                    txt.write(str(graph_def))
                f = open(_Folder + "/" + _GrapheFile.split(".",1)[0]+'_Layers.txt','w')
                for node in graph_def.node:
                    f.write(str(node.name) + '\n')
                #end for
                f.close()
                _ = tf.import_graph_def(graph_def, name='')
            #end with
        #end with 
    #end if
#end main
    
if __name__ == "__main__":
    main()
    