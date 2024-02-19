# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:55:21 2023

@author: asonm
"""
#%% 1.Bölüm
import cv2
import numpy as np

img = cv2.imread("C:/Users/asonm/Desktop/YOLO/Yolo Pretrained image/images/pexels-tima-miroshnichenko-5686103.jpg")

#print(img)

img_width = img.shape[1]
img_height = img.shape[0]


#%% 2. Bölüm

img_blob = cv2.dnn.blobFromImage(img,1/255,(416,416),swapRB=True,crop=False)

labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
          "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
          "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
          "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
          "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
          "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
          "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
          "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
          "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
          "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]


colors = ["0,255,255","0,0,255","255,255,0","0,255,0","255,0,0","255,0,255"]
colors = [np.array(color.split(",")).astype("int") for color in colors]
colors = np.array(colors)
colors = np.tile(colors,(18,1))

#%% 3.Bölüm

model = cv2.dnn.readNetFromDarknet("C:/Users/asonm/Desktop/YOLO/Pretrained model/yolov3.cfg","C:/Users/asonm/Desktop/YOLO/Pretrained model/yolov3.weights")

layers = model.getLayerNames()
output_layer = [layers[layer -1] for layer in model.getUnconnectedOutLayers()]

model.setInput(img_blob)

detection_layers = model.forward(output_layer)



#%% 4.Bölüm

for detection_layer in detection_layers:
    for object_detection in detection_layer:
        
        scores = object_detection[5:]
        predicted_id = np.argmax(scores)
        confidence = scores[predicted_id]

        if confidence > 0.35:
            
            label = labels[predicted_id]
            bounding_box = object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
            (box_center_x,box_center_y,box_width,box_height) = bounding_box.astype("int")

            start_x = int(box_center_x - (box_width/2))
            start_y = int(box_center_y - (box_height/2))

            end_x = start_x + box_width
            end_y = start_y + box_height

            box_color = colors[predicted_id]
            box_color = [int(each) for each in box_color]
            
            
            label = "{}: {:.2f}%".format(label,confidence * 100)
            print("predincted object {}".format(label))
            

            cv2.rectangle(img, (start_x,start_y), (end_x,end_y), box_color,1)
            cv2.putText(img, label, (start_x,start_y -10), cv2.FONT_HERSHEY_COMPLEX, 0.5, box_color,1)


cv2.imshow("neden göstermiyon amk", img)
cv2.waitKey(0)
cv2.destroyAllWindows()







