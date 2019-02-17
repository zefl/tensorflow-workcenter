# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 19:32:12 2018

@author: Florian
"""

import tensorflow as tf
import numpy as np
import os


def temprature_hook():
    import wmi
    import time
    w = wmi.WMI(namespace="root\OpenHardwareMonitor")
    #init with 50 degree to start while
    temp_dict = {}
    temp_to_high = True
    while temp_to_high :
        temp_to_high = False
        temperature_infos = w.Sensor()
        for sensor in temperature_infos:
            if sensor.SensorType==u'Temperature':
                temp_dict[sensor.Name]=sensor.Value
                print("Temp: " + sensor.Name + " " + str(sensor.Value))
        for temp_key in temp_dict:
            if temp_dict[temp_key]>80:
                time.sleep(15)
                temp_to_high=True


class LocationModel:

    def __init__(self,model_application):
        self.model_application = os.path.join(os.getcwd(),model_application);
        if not os.path.exists(self.model_application):
            os.makedirs(self.model_application)
        else:
            pass
            #to do check for model in folder because it is already there
			
    ###########################################################################	
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
    #						active_model --> contains the inference graph as a.pb file
    #          				lap_pbtxt_file --> contains the label map for the model as a .pbtxt file
    #    Output Args: 
    #						none
    ###########################################################################
    def init_model(self,active_model,lap_pbtxt_file = ""):
        from utils import label_map_util
        #get names from layers.txt
        input_tensor_name = "image_tensor"
        detection_boxes_name = "detection_boxes"
        detection_scores_name  = "detection_scores"
        detection_classes_name = "detection_classes"
        num_detections_name = "num_detections"
        self.graph = tf.Graph()
        model_folder_location = os.path.join(self.model_application,active_model,"model")
        modelfile = os.path.join(model_folder_location,"frozen_inference_graph.pb")
        if not os.path.isfile(modelfile):
            model_folder_location = os.path.join(self.model_application,active_model)            
            modelfile = os.path.join(model_folder_location,"frozen_inference_graph.pb")                
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
        #TODO use old models and new models with labemap only search for pbtxt and show error in export model also copy label map
        for file in os.listdir(model_folder_location):
            if file.endswith('.pbtxt'):
                lap_pbtxt_file = os.path.join(model_folder_location,file)
        
        if not lap_pbtxt_file:
            raise ValueError("No label file with .pbtxt ending is found in the folder or given location")
                
                
        label_map = label_map_util.load_labelmap(lap_pbtxt_file)
        NUM_CLASSES = label_map_util.get_max_label_map_index(label_map)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
    ###########################################################################
    
    ###########################################################################
    #Function Name: localize
    #Description: this is to use the model and localize returns the 10 highest 
    #     predictions
    #Class Variables: none
    #Input: _image --> image numpy array
    #Output: boxes --> location of predicted box
    #        scores --> number of score
    #        classes --> class label
    #        num -->
    ###########################################################################
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
    ###########################################################################
    
    ###########################################################################
    #Function Name: perpare_training
    #Description: this is to prepeare the training data
    #             it creates tf record files for training
    #Class Variables: none
    #Input: training_data --> folder with training data
    #       test_data --> folder with test data
    #Output: none
    ###########################################################################
    def perpare_training(self,train_data,test_data):   
        import shutil
        self.labels = []          
        #######################################################################
        #data preperation
        #create csv and lables from xml files for each folder
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
        if len(self.labels) == 0:
            for root, dirnames, filenames in os.walk(train_data):
                for folder in dirnames:
                    for label in folder.split('_'):
                        self.labels.append(label)
        self.labels.sort()
        labelmap = self._create_lablemap(self.labels,train_data,test_data)       
                
        #######################################################################
        #create tf record files
        #######################################################################
        train_record = self._csv_to_tfRecord(train_data)
        test_record = self._csv_to_tfRecord(test_data)
        
        shutil.copy2(labelmap, os.path.join(self.model_application,os.path.basename(labelmap)))
        shutil.copy2(train_record, os.path.join(self.model_application,os.path.basename(train_record)))
        shutil.copy2(test_record, os.path.join(self.model_application,os.path.basename(test_record)))
    ###########################################################################
        
    ###########################################################################
    #Function Name: training_flow
    #Description: trains the neuronal network
    #Class Variables: none
    #Input: training_flow --> choose between legacy and new
    #LogSettings: Hanlde logging chosse [tensorflow,system,all] to log to file
    #Output: none
    ###########################################################################    
    def train(self,training_flow,log_settings):
        import datetime
        import re
        
        #dir_path = os.path.dirname(os.path.realpath(__file__))
        config_file = os.path.join(self.model_application,"pipeline.config")
        
        #get time of training start
        now = datetime.datetime.now()
        now_str= '_'.join([str(now.year),str(now.month),str(now.day),str(now.hour),str(now.minute)])
        
        with open(config_file, 'r+') as config:
            txt = config.read()
            model = re.findall(r'type\:\s\"\w+\"',txt)
            if not model:
                model = re.findall(r'type\:\s\'\w+\'',txt)
            model = model[0].split()[1]
            model = re.sub('\W+','', model )
            
        txt = re.sub('\.\/', self.model_application.replace(os.path.sep, '/')+'/', txt)
        
        with open(config_file, 'w') as config:
            config.write(txt)
            
        
        folder_name = model + '_ownData_' + now_str
        
        save_folder =  os.path.join(self.model_application,folder_name)
        save_folder = save_folder.replace(os.path.sep, '/')
              
        save_folder_checkpoints = os.path.join(save_folder,'checkpoints')
        save_folder_checkpoints = save_folder_checkpoints.replace(os.path.sep, '/')
        #from https://github.com/datitran/raccoon_dataset/issues/41
        #prevent more then 260 characters in path
        #new training uses longer names
        if save_folder_checkpoints.count('')>260-150 and training_flow!="old":
            save_folder =  os.path.join(self.model_application,now_str)
            save_folder_checkpoints = os.path.join(save_folder,'checkpoints')
            save_folder_checkpoints = save_folder_checkpoints.replace(os.path.sep, '/')
                
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)    
            
        if not os.path.exists(save_folder_checkpoints):
            os.mkdir(save_folder_checkpoints)
		
        if log_settings == 'all' or log_settings=='tensorflow':        
            #from https://stackoverflow.com/questions/40559667/how-to-redirect-tensorflow-logging-to-a-file
            import logging
            # get TF logger
            log = logging.getLogger('tensorflow')
            log.setLevel(logging.DEBUG)
            # create formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            # create file handler which logs even debug messages
            logfile = os.path.join(save_folder_checkpoints,'tensorflow.log')
            print(logfile)
            fh = logging.FileHandler(logfile)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            log.addHandler(fh)
	
        if log_settings == 'all' or log_settings=='system':
            	#from https://www.thecodingforums.com/threads/how-to-change-redirect-stderr.355342/#post-1868822
            import sys
            logfile = os.path.join(save_folder_checkpoints,'logfile.log')
            sys.stderr.flush()
            err = open(logfile, 'a+', 1)
            #get file descriptor https://www.tutorialspoint.com/python/file_fileno.htm
            fid_err = err.fileno()
            fid_stderr = sys.stderr.fileno()
            #copies files with file descriptors https://www.tutorialspoint.com/python/os_dup2.htm
            os.dup2(fid_err, fid_stderr)
    		
        if training_flow == 'old':
            from object_detection.legacy import train
            flags = tf.app.flags
            FLAGS = flags.FLAGS
            FLAGS.train_dir = save_folder_checkpoints
            FLAGS.pipeline_config_path = config_file
            train.main(None)
        elif training_flow == 'new':
            from object_detection import model_main
            import sys
            FLAGS = tf.app.flags.FLAGS
            FLAGS.model_dir = save_folder_checkpoints
            FLAGS.pipeline_config_path = config_file
            #from https://github.com/google/python-gflags/issues/37
            FLAGS(sys.argv)
            model_main.main(None)
        else:
            raise ValueError('''Wrong input choose between 'old' and 'new' training''')
    #end train
    ###########################################################################

    ###########################################################################
    #Function Name: export_model
    #Description: freeze checkpoint to model
    #Class Variables: none
    #Input: active_model --> choose a trained model
    #Output: none    
    ###########################################################################
    def export_model(self,active_model):
        from shutil import copyfile
        from google.protobuf import text_format
        from object_detection import exporter
        from object_detection.protos import pipeline_pb2
        
        model_folder = os.path.join(os.getcwd(),self.model_application,active_model)
        
        
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

        copyfile(os.path.join(self.model_application,"label_map.pbtxt"),os.path.join(output_folder,"label_map.pbtxt"))
    
    
    ###########################################################################
    #Function Name: eval_model
    #Description: runs the evaluation for a model
    #Class Variables: none
    #Input: active_model --> choose a trained model
    #       log --> enable or disable log
    #Output: none    
    ###########################################################################  
    def eval_model(self,active_model,log=True):
        import object_detection.legacy.eval as eval_model
        import logging
        
        logging.basicConfig(level=logging.DEBUG)
        
        checkpoint_dir = os.path.join(self.model_application,active_model,'checkpoints').replace(os.path.sep, '/')
        
        eval_dir = os.path.join(self.model_application,active_model,"eval").replace(os.path.sep, '/')
        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)
        
        if log:        
            #from https://stackoverflow.com/questions/40559667/how-to-redirect-tensorflow-logging-to-a-file
            import logging
            # get TF logger
            log = logging.getLogger('tensorflow')
            log.setLevel(logging.DEBUG)
            # create formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            # create file handler which logs even debug messages
            logfile = os.path.join(eval_dir,'tensorflow.log')
            print(logfile)
            fh = logging.FileHandler(logfile)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            log.addHandler(fh)
        
        
        flags = tf.app.flags.FLAGS
        flags.checkpoint_dir = checkpoint_dir
        flags.eval_dir = eval_dir
        
        pipeline = os.path.join(self.model_application,active_model,'checkpoints','pipeline.config').replace(os.path.sep, '/')
        flags.pipeline_config_path = pipeline
        eval_model.main(None)
    ###########################################################################
        
    ###########################################################################
    #from https://github.com/MicrocontrollersAndMore/TensorFlow_Tut_3_Object_Detection_Walk-through
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
                if not member[0].text in self.labels:
                    self.labels.append(member[0].text)
            # end for
    
        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        xml_df.to_csv(os.path.join(os.path.dirname(path),csv_name+'.csv'), index=None)
    # end function
    ###########################################################################

    ###########################################################################
    #from https://github.com/MicrocontrollersAndMore/TensorFlow_Tut_3_Object_Detection_Walk-through
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
    #from https://github.com/MicrocontrollersAndMore/TensorFlow_Tut_3_Object_Detection_Walk-through
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
    #from https://github.com/MicrocontrollersAndMore/TensorFlow_Tut_3_Object_Detection_Walk-through
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
    cnn_model = LocationModel('mouse_cat_detection_model')
    #cnn_model.perpare_training('./data/train','./data/test')
    cnn_model.train('new','tensorflow')
#    current_model = "ssd_mobilenet_v2_ownData_2019_1_28_18_46"
#    cnn_model.export_model(current_model)
#    cnn_model.eval_model(current_model)
    
###############################################################################
# Test Function to test all pictures in eval folder
###############################################################################
#
#    cnn_model.init_model(current_model)
#    #read image
#    for file in os.listdir(os.path.join(os.getcwd(),cnn_model.model_folder,"eval")):
#        if file.endswith(('.png', '.jpg', '.jpeg','.JPG')):
#            image_path= os.path.join(os.getcwd(),cnn_model.model_folder,"eval",file)
#            import cv2
#            image = cv2.imread(image_path)
#            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#            (boxes, scores, classes, num) = cnn_model.localize(image)
#            
#            from utils import visualization_utils as vis_util
#            # Visualization of the results of a detection.
#            vis_util.visualize_boxes_and_labels_on_image_array(image,
#                                                               boxes,
#                                                               classes,
#                                                               scores,
#                                                               cnn_model.category_index,
#                                                               use_normalized_coordinates=True,
#                                                               line_thickness=2,
#                                                               min_score_thresh=.80)
#            #reizse image to fit screen
#            height, width, channels = image.shape 
#            if height > 500 and width > 500:
#                re_height = 500/height
#                re_width = 500/width
#                image = cv2.resize(image,(0,0), fx=re_height, fy=re_width,interpolation=cv2.INTER_LINEAR)
#            cv2.imshow("result", image)
#            #show 3 sec = 3000ms
#            cv2.waitKey(3000)
#            cv2.destroyAllWindows()
#            eval_dir = os.path.join(cnn_model.model_folder,current_model,"eval").replace(os.path.sep, '/')
#            cv2.imwrite(os.path.join(eval_dir,file),image)