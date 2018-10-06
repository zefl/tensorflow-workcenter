# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 19:32:12 2018

@author: Florian
"""

import tensorflow as tf
import cv2
import numpy as np
import os


class LocationModel:

    def __init__(self,model_folder):
        self.model_folder = os.path.join(os.getcwd(),model_folder);
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        else:
            pass
            #to do check for model in folder because it is already there
#########################################################################	
#    Description: 
#						Init object detection model
#    Class Variables: 
#						graph --> contains the graph for a tensorflow session
#                   	image_tensor --> input tensor of graph
#                  		detection_boxes --> output tensor of graph
#                    	detection_scores --> output tensor of graph
#                    	detection_classes --> output tensor of graph
#                    	num_detections --> output tensor of graph
#                    	category_index --> contains labels
#    Input Args:
#						modelfile --> contains the inference graph as a.pb file
#          				label_map --> contains the label map for the model as a .pbtxt file
#    Output Args: 
#						none
#########################################################################
    def init_model(self,modelfile,label_map):
        from utils import label_map_util
        #get names from layers.txt
        input_tensor_name = "image_tensor"
        detection_boxes_name = "detection_boxes"
        detection_scores_name  = "detection_scores"
        detection_classes_name = "detection_classes"
        num_detections_name = "num_detections"
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            with tf.gfile.FastGFile(modelfile,'rb') as model:
                # The graph-def is a saved copy of a TensorFlow graph.
                graph_def = tf.GraphDef()
                #read in graph model
                graph_def.ParseFromString(model.read())
                # Finally we import the graph-def to the default TensorFlow graph.
                tf.import_graph_def(graph_def, name='')
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.graph.get_tensor_by_name(input_tensor_name + ":0")
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.graph.get_tensor_by_name(detection_boxes_name + ":0")
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.graph.get_tensor_by_name(detection_scores_name + ":0")
        self.detection_classes = self.graph.get_tensor_by_name(detection_classes_name + ":0")
        self.num_detections = self.graph.get_tensor_by_name(num_detections_name + ":0")
        #get labels
        #TENSORFLOW_OBJECT_DETECTION_API = "C:/Users/Florian/Documents/Tensorflow/tensorflow_api/models/research/object_detection/"
        #LABEL_MAP_LOC = os.path.join(TENSORFLOW_OBJECT_DETECTION_API,"data/mscoco_label_map.pbtxt")
        label_map = label_map_util.load_labelmap(label_map)
        NUM_CLASSES = 90
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
    
    #############################################################
    #Function Name: localize
    #Description: this is to use the model and localize returns the 10 highest 
    #     predictions
    #Class Variables: none
    #Input: _image --> image numpy array
    #Output: boxes --> location of predicted box
    #        scores --> number of score
    #        classes --> class label
    #        num -->
    ##############################################################
    def localize(self,_image):
        image_expanded = np.expand_dims(_image, axis=0)
        with tf.Session(graph=self.graph) as sess:
            boxes, scores, classes, num = sess.run(
                            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                            feed_dict={self.image_tensor: image_expanded})
        
        boxes = np.squeeze(boxes)
        boxes = boxes[0:10]
        scores = np.squeeze(scores)
        scores = scores[0:10]
        classes = np.squeeze(classes).astype(np.int32)
        classes = classes[0:10]
        return (boxes, scores, classes, num)
    
    #############################################################
    #Function Name: perpare_training
    #Description: this is to prepeare the training data
    #             it creates tf record files for training
    #Class Variables: none
    #Input: training_data --> folder with training data
    #       test_data --> folder with test data
    #Output: none
    ##############################################################
    def perpare_training(self,train_data,test_data):   
        import shutil          
        #######################################################################
        #data preperation
        #create csv from xml files for each folder
        #######################################################################
        for folder in os.scandir(train_data):
            if folder.is_dir():
                self._xml_to_csv(folder.path,folder.name)
        for folder in os.scandir(test_data):
            if folder.is_dir():
                self._xml_to_csv(folder.path,folder.name)
                
        #######################################################################
        #get labels
        #label are generated based on foldernames 
        #######################################################################
        self.labels = []
        for root, dirnames, filenames in os.walk(train_data):
            for name in dirnames:
                for label in name.split('_'):
                    self.labels.append(label)
        self.labels.sort()
        labelmap = self._create_lablemap(self.labels,train_data,test_data)       
                
        #######################################################################
        #create tf record files
        #######################################################################
        train_record = self._csv_to_tfRecord(train_data)
        test_record = self._csv_to_tfRecord(test_data)
        
        shutil.copy2(labelmap, os.path.join(self.model_folder,os.path.basename(labelmap)))
        shutil.copy2(train_record, os.path.join(self.model_folder,os.path.basename(train_record)))
        shutil.copy2(test_record, os.path.join(self.model_folder,os.path.basename(test_record)))
        
    def train(self):
        from object_detection.legacy import train
        import datetime
        import re
        
        #dir_path = os.path.dirname(os.path.realpath(__file__))
        self.config_file = os.path.join(self.model_folder,"pipeline.config")
        
        #get time of training start
        now = datetime.datetime.now()
        now_str= '_'.join([str(now.year),str(now.month),str(now.day),str(now.hour),str(now.minute)])
        
        with open(self.config_file, 'r+') as config:
            txt = config.read()
            model = re.findall(r'type\:\s\"\w+\"',txt)
            model = model[0].split()[1]
            model = re.sub('\W+','', model )
            
        txt = re.sub('\.\/', self.model_folder.replace(os.path.sep, '/')+'/', txt)
        
        with open(self.config_file, 'w') as config:
            config.write(txt)
            
        
        folder_name = model + '_ownData_' + now_str
        
        save_folder =  os.path.join(self.model_folder,folder_name)
        save_folder = save_folder.replace(os.path.sep, '/')
        
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        
        save_folder_checkpoints = os.path.join(save_folder,'checkpoints')
        save_folder_checkpoints = save_folder_checkpoints.replace(os.path.sep, '/')
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        flags = tf.app.flags.FLAGS
        flags.train_dir = save_folder_checkpoints
        flags.pipeline_config_path = self.config_file
        train.main(None)
        
    def export_model(self,active_folder):
        from google.protobuf import text_format
        from object_detection import exporter
        from object_detection.protos import pipeline_pb2
        
        model_folder = os.path.join(os.getcwd(),self.model_folder+active_folder)
        
        
        checkpoint_folder = os.path.join(model_folder,'checkpoints')
        import re
        checkpoint_nr = 0
        for fname in os.listdir(checkpoint_folder):
            if fname.endswith('.config'):
                pipieline_config = os.path.join(checkpoint_folder,fname)
                #see object_detection/export_inference_graph.py
                trainEvalPipelineConfig = pipeline_pb2.TrainEvalPipelineConfig()
                with tf.gfile.GFile(pipieline_config, 'r') as f:
                    text_format.Merge(f.read(), trainEvalPipelineConfig)
            if fname.endswith('.meta'):
                _checkpoint_nr = int(re.findall(r'\d+',fname)[0])
                if _checkpoint_nr > checkpoint_nr:
                    checkpoint_nr = _checkpoint_nr
            if fname.endswith('checkpoint'):
                #fix windows \ and replace with /
                with open(os.path.join(checkpoint_folder,'checkpoint'),'r') as f:
                    txt = f.read()
                txt=txt.replace("\\\\","/");
                with open(os.path.join(checkpoint_folder,'checkpoint'),'w') as f:
                    f.write(txt)
                    
        checkpoint = os.path.join(checkpoint_folder,"model.ckpt-"+str(checkpoint_nr))
        
        output_folder =  os.path.join(model_folder,"model")
        
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
            
        input_type = "image_tensor"
        exporter.export_inference_graph(input_type, trainEvalPipelineConfig, checkpoint, output_folder)
        
    def eval_model(self,train_folder):
        import object_detection.legacy.eval as eval_model
        import logging
        
        logging.basicConfig(level=logging.DEBUG)
        
        checkpoint_dir = os.path.join(self.model_folder,train_folder,'checkpoints').replace(os.path.sep, '/')
        
        eval_dir = os.path.join(self.model_folder,train_folder,"eval").replace(os.path.sep, '/')
        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)
        
        pipeline = os.path.join(self.model_folder,train_folder,'checkpoints','pipeline.config').replace(os.path.sep, '/')
        
        flags = tf.app.flags.FLAGS
        flags.checkpoint_dir = checkpoint_dir
        flags.eval_dir = eval_dir
        flags.pipeline_config_path = pipeline
        eval_model.main(None) 
        
    ###########################################################################
    #from 
    def _xml_to_csv(self,path,csv_name):
        import glob
        import pandas as pd
        #from https://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/ 
        try:
            import xml.etree.cElementTree as ET
        except ImportError:
            import xml.etree.ElementTree as ET
        xml_list = []
        #glob searches all files with extion of .xml (* stands for 0 to x characters)
        for xml_file in glob.glob(path + '/*.xml'):
            tree = ET.parse(xml_file)
            #root is description
            root = tree.getroot()
            #object contains all bounding boxes annotations
            for member in root.findall('object'):
                value = (root.find('filename').text, int(root.find('size')[0].text), int(root.find('size')[1].text), member[0].text,
                         int(member[4][0].text), int(member[4][1].text), int(member[4][2].text), int(member[4][3].text))
                xml_list.append(value)
            # end for
    
        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        xml_df.to_csv(os.path.join(os.path.dirname(path),csv_name+'.csv'), index=None)
    # end function
    ###########################################################################

    ###########################################################################
    def _csv_to_tfRecord(self,path):
        from collections import namedtuple
        import pandas as pd
        import glob
        # the purpose of this function is to translate the data from one CSV file in pandas.DataFrame format
        # into a list of the named tuple below, which then can be fed into TensorFlow
        # establish the named tuple data format
        dataFormat = namedtuple('data', ['filename', 'object'])
        # declare, populate, and return the list of named tuples of CSV data
        csvFileDataList = []
        #read all csv to one array
        for csv_file in glob.glob(path + '/*.csv'):
            csvFileDataFrame = pd.read_csv(csv_file)
            folder = os.path.basename(csv_file.split('.csv')[0])
            #  pandas.DataFrame.groupby() returns type pandas.core.groupby.DataFrameGroupBy
            csvFileDataFrameGroupBy = csvFileDataFrame.groupby('filename')
            for filename, x in zip(csvFileDataFrameGroupBy.groups.keys(), csvFileDataFrameGroupBy.groups):
                filelocation = os.path.join(folder,filename)
                csvFileDataList.append(dataFormat(filelocation, csvFileDataFrameGroupBy.get_group(x)))
        
            # instantiate a TFRecordWriter for the file data
        filename = ""
        if path.find('train') > -1 :
            filename = 'train'
        elif path.find('test') > -1 :
            filename = 'test'
        record_file = os.path.join(path,filename+'.tfrecord')
        tfRecordWriter = tf.python_io.TFRecordWriter(record_file)
        
        # for each file (not each line) in the CSV file data . . .
        # (each image/.xml file pair can have more than one box, and therefore more than one line for that file in the CSV file)
        for singleFileData in csvFileDataList:
            try:
                tfExample = self._createTfExample(singleFileData, path)
            except ValueError as error:
                if error.args[0].find('is not in list'):
                    raise ValueError('Labels of folder name and xml files do not match')
                else:
                    raise ValueError(error)
            tfRecordWriter.write(tfExample.SerializeToString())
        # end for
        tfRecordWriter.close()
        return record_file
    # end function
    ###########################################################################

    ###########################################################################
    def _createTfExample(self,singleFileData, path):
        import io
        from object_detection.utils import dataset_util
        from PIL import Image
        # use TensorFlow's GFile function to open the .jpg image matching the current box data
        with tf.gfile.GFile(os.path.join(path, '{}'.format(singleFileData.filename)), 'rb') as tensorFlowImageFile:
            tensorFlowImage = tensorFlowImageFile.read()
        # end with
    
        # get the image width and height via converting from a TensorFlow image to an io library BytesIO image,
        # then to a PIL Image, then breaking out the width and height
        bytesIoImage = io.BytesIO(tensorFlowImage)
        pilImage = Image.open(bytesIoImage)
        width, height = pilImage.size
    
        # get the file name from the file data passed in, and set the image format to .jpg
        fileName = singleFileData.filename.encode('utf8')
        imageFormat = b'jpg'
    
        # declare empty lists for the box x, y, mins and maxes, and the class as text and as an integer
        xMins = []
        xMaxs = []
        yMins = []
        yMaxs = []
        classesAsText = []
        classesAsInts = []
    
        # for each row in the current .xml file's data . . . (each row in the .xml file corresponds to one box)
        for index, row in singleFileData.object.iterrows():
            xMins.append(row['xmin'] / width)
            xMaxs.append(row['xmax'] / width)
            yMins.append(row['ymin'] / height)
            yMaxs.append(row['ymax'] / height)
            classesAsText.append(row['class'].encode('utf8'))
            classesAsInts.append(self.labels.index(row['class'])+1)
        # end for
    
        # finally we can calculate and return the TensorFlow Example
        tfExample = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(fileName),
            'image/source_id': dataset_util.bytes_feature(fileName),
            'image/encoded': dataset_util.bytes_feature(tensorFlowImage),
            'image/format': dataset_util.bytes_feature(imageFormat),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xMins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xMaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(yMins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(yMaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classesAsText),
            'image/object/class/label': dataset_util.int64_list_feature(classesAsInts)}))
    
        return tfExample
    # end function
    
    ###########################################################################



    ###########################################################################
    def _create_lablemap(self,lables,train_location,test_location):
        label_text = ""
        i=1
        for label in lables:
            label_text = label_text + "item {\n" + '\tid: ' + str(i) + "\n\tname:'" + label + "'\n}\n"
            i = i + 1;
        filename_train = os.path.join(train_location,'label_map.pbtxt')
        filename_test =os.path.join(test_location,'label_map.pbtxt')
        with open(filename_train, 'w') as f:
            f.write(label_text)
        with open(filename_test, 'w') as f:
            f.write(label_text)
        return filename_train

    # end function
    
    ###########################################################################

if __name__ == "__main__":
    cnn_model = LocationModel('mouse_detection_model')
    cnn_model.perpare_training('./data/train','./data/test')
    #cnn_model.train()
    #current_model = "faster_rcnn_inception_v2_ownData_2018_9_18_17_0"
    #cnn_model.export_model(current_model)
    #cnn_model.eval_model(current_model)

    
    
    
#    model_own = os.path.join(os.getcwd(),"data/train/trained_model_faster_rcnn_inception_v2_2018_9_14_13_42/model/frozen_inference_graph.pb")
#    labels_own = os.path.join(os.getcwd(),"data/train/label_map.pbtxt")
#    
#    #model = os.path.join(os.getcwd(),'model/object_detection/faster_rcnn_inception_v2_coco/frozen_inference_graph.pb')
#    #TENSORFLOW_OBJECT_DETECTION_API = "C:/Users/Florian/Documents/Tensorflow/tensorflow_api/models/research/object_detection/"
#    #labels = os.path.join(TENSORFLOW_OBJECT_DETECTION_API,"data/mscoco_label_map.pbtxt")
#
#    cnn_model.init(model_own,labels_own)
#    #read image 
#    image_path= "./Cat_93.jpg"
#    image = cv2.imread(image_path)
#    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#    (boxes, scores, classes, num) = cnn_model.localize(image)
#    
#    from utils import visualization_utils as vis_util
#    # Visualization of the results of a detection.
#    vis_util.visualize_boxes_and_labels_on_image_array(image,
#                                                       boxes,
#                                                       classes,
#                                                       scores,
#                                                       cnn_model.category_index,
#                                                       use_normalized_coordinates=True,
#                                                       line_thickness=8)
#    #reizse image to fit screen
#    height, width, channels = image.shape 
#    if height > 500 and width > 500:
#        re_height = 500/height
#        re_width = 500/width
#        image = cv2.resize(image,(0,0), fx=re_height, fy=re_width,interpolation=cv2.INTER_LINEAR)
#    cv2.imshow("result", image)
#    cv2.waitKey()
#    cv2.destroyAllWindows()