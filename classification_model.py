# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 17:42:43 2018

@author: Florian
"""

import tensorflow as tf
import cv2
import numpy as np
import os

class classification_model:
            
    def __init__(self,modelfolder):
        #get model file and label file
        if os.path.exists(modelfolder):
            modelfile = ""
            label_file = ""
            for file in os.listdir(modelfolder):
                if file.endswith(".pb"):
                    modelfile = os.path.join(modelfolder,file)
                elif file == "labels.txt":
                    label_file = os.path.join(modelfolder,file)
            if not modelfile:
                raise ValueError("No Modelfile in Folder")
            if not label_file:
                raise ValueError("No Labelfile in folder")
        else:
            raise ValueError("Model Folder does not exist")
        
        #get lables
        self.labels = []
        if not os.path.isfile(label_file):
            raise ValueError("Label file does not exist")
        
        for currentLine in tf.gfile.GFile(label_file):
            label = currentLine.rstrip()
            self.labels.append(label)
        
        #get names from layers.txt
        input_tensor_name = ""
        output_tensor_name = ""
        bottelneck_tensor_name = ""
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            with tf.gfile.FastGFile(modelfile,'rb') as model:
                # The graph-def is a saved copy of a TensorFlow graph.
                graph_def = tf.GraphDef()
                #read in graph model
                graph_def.ParseFromString(model.read())
                i=0
                for node in graph_def.node:
                    if i == 0:
                        input_tensor_name = node.name
                    if i == len(graph_def.node)-1:
                        output_tensor_name = node.name
                    i=i+1
                #end for
                # Finally we import the graph-def to the default TensorFlow graph.
                tf.import_graph_def(graph_def, name='')
        #get tensor --> need to add :0 this is the tensor 
        self.input_tensor = self.graph.get_tensor_by_name(input_tensor_name + ":0")
        #get size of input image
        self.input_image_size = self.input_tensor.get_shape()[1:4]
        self.output_tensor = self.graph.get_tensor_by_name(output_tensor_name + ":0")
        self.bottelneck_tensor = self.graph.get_tensor_by_name(bottelneck_tensor_name +":0")
    

    #from Tensorflow API tensorflow/tensorflow/examples/label_image    
    def read_tensor_from_image_file(self,
                                file_name,
                                input_mean=0,
                                input_std=255):
      input_height=self.input_image_size[0]
      input_width=self.input_image_size[1]
      input_name = "file_reader"
      file_reader = tf.read_file(file_name, input_name)
      if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
      elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
      elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
      else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
      float_caster = tf.cast(image_reader, tf.float32)
      dims_expander = tf.expand_dims(float_caster, 0)
      resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
      normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
      sess = tf.Session()
      normalized_image = sess.run(normalized)
      result = normalized_image
      return result        	
  
  
    def read_cv2_from_image_file( self,
                                image_path,
                                mean=0,
                                std=0,
                                padded=False):
        import cv2
        import numpy as np
        input_height=self.input_image_size[0]
        input_width=self.input_image_size[1]
        image_reader = cv2.imread(image_path)
        # see https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
        #image_reader_rgb = cv2.cvtColor(image_reader_brg, cv2.COLOR_BGR2RGB)
        #https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
        #pad image to scare
        if padded:
            height, width, channels = image_reader.shape 
            delta_border = height-width
            top,bottom,left,right = 0 , 0 , 0 , 0
            #Picture higher than wider 
            if delta_border > 0:
                left = round(delta_border/2)
                right = left
                #image_buffer = (image_size[0],image_size[1]+delta_border,image_size[2])
            #Picture wider than higher
            elif delta_border < 0:
                bottom = round(abs(delta_border)/2)
                top = bottom
            #pad with white color
            color = [0, 0, 0]
            # see https://docs.opencv.org/3.1.0/d3/df2/tutorial_py_basic_ops.html
            image_reader = cv2.copyMakeBorder(image_reader,top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
        #see https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html
        image_res = cv2.resize(image_reader,(input_height,input_width),interpolation=cv2.INTER_LINEAR)
        image_model = np.expand_dims(image_res,axis=0)
        #is this right or do i need to do this before adding 4 dimension
        if not mean:
            mean = np.mean(image_model, axis=(1,2))
            mean = np.mean(mean, axis=1)
        if not std:
            std = np.std(image_model, axis=(1,2))
            std = np.mean(std, axis=1)
        image_normalized = np.divide(np.subtract(image_model,mean),std)
        return image_normalized
    
    def classify(self,image_tensor,number=1):
        predictions = [];
        with tf.Session(graph=self.graph) as sess:
            predictions = sess.run(self.output_tensor,{self.input_tensor:image_tensor})
        # Get a sorted index for the pred-array.
        #idx = pred.argsort()
        # The index is sorted lowest-to-highest values. Take the last k.
        #top_k = idx[-k:]        			
        sorted_predictions = predictions[0].argsort()[-len(predictions[0]):][::-1]
        label = []
        accuracy = []
        for i in range(0,number):
            #spli between number and label and return only label
            label.append(self.labels[sorted_predictions[i]].split(":")[1])
            accuracy.append(round(predictions[0][sorted_predictions[i]]*100,2))
        return label, accuracy

def get_random_image(imagefolder):
    import os
    import random
    if not os.path.exists(imagefolder):
        return None
    files = []
    for file in os.listdir(imagefolder):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            files.append(file)
    number_of_files = len(files)
    
    if number_of_files == 0:
         raise ValueError("No Images in folder")
    rand_img = random.randrange(0,number_of_files)
    return os.path.join(imagefolder,files[rand_img])


if __name__ == "__main__":
    #folder needs to contain .pb of model and labels.txt
    model = classification_model("./model/classification/inception_v4/")
    #read image 
    image_path = get_random_image("C:/Users/Florian/Documents/Tensorflow/data/Training_Data\Maus")
    
    #image_tensor = model.read_tensor_from_image_file(image_path,128,128)
    image_tensor = model.read_cv2_from_image_file(image_path)
    #image_tensor = model.read_cv2_from_image_file(image_path,[128],[128])
    label , perdictions =  model.classify(image_tensor)
            
    #remove 4 axis to show image     
    image=np.squeeze(image_tensor,axis=0)
    # Write some Text

    font                   = cv2.LINE_AA 
    bottomLeftCornerOfText = (10,20)
    fontScale              = 0.5
    #Colors are B,G,R
    fontColor              = (0,0,255)
    lineType               = 2
    
    cv2.putText(image,label[0], 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    lineType)
    
    bottomLeftCornerOfText = (10,40)
    
    cv2.putText(image," with " + str(perdictions[0]) + " %", 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    lineType)
    
    cv2.imshow('image_tensor',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()