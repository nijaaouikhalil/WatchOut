from __future__ import division
import time
import torch, sys
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random 
import argparse
import pickle as pkl
from threading import Thread
import pyautogui, time, os
import win32com.client as wincl
import six.moves.urllib as urllib
import tarfile
from matplotlib import pyplot as plt
from PIL import Image
#from utils import label_map_util
#from utils import visualization_utils as vis_util
import tensorflow as tf
from os import path
import runmodel
from tkinter import *
from PIL import ImageTk
import PIL.Image
from datetime import datetime
from scipy.misc import imsave

label_detected = []
counting,counter = 0,0

check = False
tempimg=''
global_number = 0


X_car_arr =  [0,0]
X_light_arr =  [0,0]

def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    #img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_



def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def buttonClick():
    global fenetre, global_number,fenetre2,canvas2
    #string1 = global_number
    string2 = "cible"
    Thread(target=main_main).start()
    fenetre2 = Tk()
    canvas = Canvas(fenetre2, width=200, height=100)
    txt = canvas.create_text(75, 60, text=global_number, font="Arial 16 italic", fill="blue")
    canvas.pack()
    canvas2 = Canvas(fenetre2, width=200, height=100)
    txt = canvas2.create_text(75, 60, text=string2, font="Arial 16 italic", fill="blue")
    canvas2.pack()
    canvas2.update_idletasks
    #canvas2.update()
    fenetre2.mainloop()
    
    fenetre.quit()
    #sys.exit()

def check_for_color():
    global check, tempimg
    while True:
        if check:
            red_state = ''
            red_state = runmodel.main(tempimg)
            tempimg = ''
            check = False
            time.sleep(1)
        else:
            tempimg = ''
            check = False
            time.sleep(1)
            
def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	if perWidth != 0 :
            return ((knownWidth * focalLength) / perWidth)  * 2.54 / 96


old_distance,temp_Y1,temp_Y2, temp_X1,temp_X2 = 0 , 0 ,0 ,0, 0
X_light,X_car =0,0

def write(x, img):
    global X_light_arr, X_car_arr
    global counting, counter,check,tempimg, global_number,X_light,X_car
    global temp_Y1,temp_Y2, temp_X1,temp_X2, old_distance,fenetre2,canvas2
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    ver = 0
    #get width and height
    x1 = str(c1[0]).split('(')[1].split(',')[0]
    x2 = str(c1[1]).split('(')[1].split(',')[0]
    y1 = str(c2[0]).split('(')[1].split(',')[0]
    y2 = str(c2[1]).split('(')[1].split(',')[0]
    X = int(y1)-int(x1)
    Y = int(y2)-int(x2)
    rotating_1 = int(y2)-int(x1)
    rotating_2 = int(x2)-int(y1)
    surface_old = (temp_Y1-temp_X1)*(temp_Y2-temp_X2)
    #Focal length of the camera
    F = 674.25
    obj_width = 340.15
    obj_height = 1040.15
    car_height = obj_height*3
    cls = int(x[-1])
    temp_ditance_lights = 0
    label = "{0}".format(classes[cls])
    unit = ' cm'
    if ( str(label) == 'traffic light' and  str(label) == 'car') :
        check = True
        tempimg = img
    if str(label) == 'traffic light':
        X_light = X
        X_light_arr.append(X_light)
    elif str(label) == 'car':
        X_car = X
        X_car_arr.append(X_car)
    if X_light_arr[-1] != 0:
        if X_light_arr[-1] == X_light_arr[-2]:
            print('standing still ',old_distance)
            counting +=1
            counter +=1
            ver = 1
    if X_car_arr[-1] != 0:
        if X_light_arr[-1] == X_light_arr[-2]:
            print('the car is not moving')
        elif X_light_arr[-1] > X_light_arr[-2]:
            print('the car is slowing down')
        elif X_light_arr[-1] < X_light_arr[-2]:
            print('the car is moving faster than you')
        try:
            temp_ditance_lights = round(distance_to_camera(obj_height,F,X_light))
        except:
            print('pass')
        try:
            temp_ditance_car = round(distance_to_camera(car_height,F,X_car))
        except:
            print('pass')
        temp_ditance = temp_ditance_lights - temp_ditance_car
        print(temp_ditance)
        if temp_ditance > 1000:
            unit = ' m'
            temp_ditance = temp_ditance/100
            obj_height = obj_height/100
            car_height = obj_height/100
            print('the distance = '+str(label)+' is ',temp_ditance, unit)
            counting 
        elif temp_ditance < 205:
            if counting>100:
                now = datetime. now()
                current_time = str(now.strftime("%H:%M:%S")).replace(':','_')
                imsave('output/'+current_time+'.png', img)
                canvas2.update()
                global_number+=1
                print(global_number)
                counting = 0
            else:
                counting += 1
        temp_X1,temp_X2 = int(x1), int(x2)
        temp_Y1,temp_Y2 = int(y1), int(y2)
        old_distance = int(round(distance_to_camera(obj_height,F,X)))

    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    tem = cv2.rectangle(img, c1, c2,color, -1)
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    #sizesh =  img.shape
    #height, width = tem.shape[:2]
    return img



def plot_origin_image(image_np, boxes, classes, scores, category_index):

    # Size of the output images.
    IMAGE_SIZE = (12, 8)
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      np.squeeze(boxes),
      np.squeeze(classes).astype(np.int32),
      np.squeeze(scores),
      category_index,
      min_score_thresh=.5,
      use_normalized_coordinates=True,
      line_thickness=3)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)

    # save augmented images into hard drive
    # plt.savefig( 'output_images/ouput_' + str(idx) +'.png')
    plt.show()
    
def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
   
    parser.add_argument("--video", dest = 'video', help = 
                        "Video to run detection upon",
                        default = "video5.mp4", type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    return parser.parse_args()


    
def main_main():
    global classes, colors
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    num_classes = 80

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()
    

    
    
    num_classes = 80
    bbox_attrs = 5 + num_classes
    
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model(get_test_input(inp_dim, CUDA), CUDA)
            
    model.eval()
    
    videofile = args.video
    
    cap = cv2.VideoCapture(videofile)
    
    assert cap.isOpened(), 'Cannot capture source'
    
    frames = 0
    start = time.time()    
    while cap.isOpened():
        
        ret, frame = cap.read()
        if ret:
            
            img, orig_im, dim = prep_image(frame, inp_dim)
            im_dim = torch.FloatTensor(dim).repeat(1,2)                        
            
            
            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
            
            
            with torch.no_grad():   
                output = model(img, CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

            if type(output) == int:
                frames += 1
                #print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
            

        
            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
            
            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
            
            output[:,1:5] /= scaling_factor
    
            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
            
            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))
            
            list(map(lambda x: write(x, orig_im), output))
            
            
            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            #print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))         
        else:
            break

        
def interface():
    global fenetre
    fenetre = Tk()
    load = PIL.Image.open("ticket.png").convert("RGB")
    render = ImageTk.PhotoImage(load)
    img = Label(fenetre, image=render)
    img.image = render
    img.grid(row=1, column=1)
    # frame 2
    Frame2 = Frame(fenetre, relief=GROOVE)
    Frame2.grid(row=1, column=2)
    photo = PhotoImage(file=r"rename.png")
    Label(Frame2, text="Welcome to WatchOut", font=("Helvetica", 16)).pack(padx=10, pady=10)
    Button(fenetre, command=buttonClick, text='run your prototype', width=100, height=60, background='white',
           image=photo).grid(row=4, column=2)
    canvas = Canvas(fenetre, width=400, height=25).grid(row=2, column=2)
    canvas = Canvas(fenetre, width=400, height=50).grid(row=5, column=2)
    fenetre.mainloop()

    
if __name__ == '__main__':
    #main_main()
    #interface()
    Thread(target = interface).start()
    #Thread(target = main_main).start()
    Thread(target = check_for_color).start()
    #Thread(target = text_to_speech).start()


    

