# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 11:19:18 2019

@author: Florian
"""

#from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

import cv2
import os
import object_detection_model
from datetime import datetime

model = 'mouse_cat_detection_model'
model_checkpoints = "faster_rcnn_inception_v2_ownData_2019_2_14_0_48"
cnn_model = object_detection_model.LocationModel(model)
current_model = model_checkpoints
cnn_model.init_model(current_model)


#eval_video = "Own_KatzeFrisstMaus_ohneTon.mp4"
eval_video = "Katze1_Short.mp4"

min_sorce= 0.90

cap = cv2.VideoCapture(os.path.join("./",model,"eval",eval_video))

# Get current width of frame
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
# Get current height of frame
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"MP4V")

eval_folder = os.path.join("./",model,model_checkpoints,"eval")
if not os.path.exists(eval_folder):
    os.mkdir(eval_folder)

video_out = eval_video.split('.')
video_out = video_out[0]+'_'+str(min_sorce)+'.'+video_out[1]
out = cv2.VideoWriter(os.path.join(eval_folder,video_out), fourcc, int(fps), (int(width), int(height)))

past_frames= 0;
max_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
text = ''
while(cap.isOpened()):
    start_time =  datetime.now()
    ret, frame = cap.read()

    if ret:
        past_frames = past_frames + 1
       # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #run object detection
        (boxes, scores, classes, num) = cnn_model.localize(frame)
        from utils import visualization_utils as vis_util
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(frame,
                                                           boxes,
                                                           classes,
                                                           scores,
                                                           cnn_model.category_index,
                                                           use_normalized_coordinates=True,
                                                           line_thickness=2,
                                                           min_score_thresh=min_sorce)
        out.write(frame)
        percent = past_frames/max_frames*100.0
        percent = float("{0:.2f}".format(percent))

        # Write some Text

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (0,25)
        fontScale              = 0.5
        fontColor              = (255,255,255)
        lineType               = 2
        
        cv2.putText(frame,
            text, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.imshow('video',frame)
        stop_time = datetime.now()
        run_time = stop_time-start_time
        balance_time=run_time*max_frames*(1-(percent/100))
        text = 'Percentag analysed: ' + str(percent) + " Time to finish: " + str(balance_time)
        print(text)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
