# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 10:11:22 2018

@author: khalifazada
"""

# Object Detection using SSD

# importing libraries
import torch, cv2, imageio
from torch.autograd import Variable
# BaseTransform will adapt images for the pre-trained NN
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd

# implementing object detection function
# frame - image to use for object detection
# net - SSD Network
# transform - will transform images into the right format
def detect(frame, net, transform):
  #determine dims of frame
  height, width = frame.shape[:2] # only first 2 elements
  # apply image transformation
  frame_t = transform(frame)[0] # only 1st element
  # tensor from numpy array
  x = torch.from_numpy(frame_t).permute(2, 0, 1) # change color order RGB to BRG
  # create fake dim for batch input, convert to torch variable
  x = Variable(x.unsqueeze(0))
  # feed input x in to network, returns output y
  y = net(x)
  # create data attributes from y
  detections = y.data
  # normalization of object's position
  scale = torch.Tensor([width, height, width, height])
  # detections = [batch, number of classes, class occurence, (score,x0,y0,x1,y1)]
  # score in the above tuple is a threshold determining object recognized or not
  # loop through classes and identify number of occurence for each class
  for i in range(detections.size(1)):
    j = 0 # number of occurence
    while detections[0,i,j,0] >= 0.6:
      # normalized coordinates of occurrence j of class i
      pt = (detections[0, i, j, 1:] * scale).numpy()
      # draw rectangle
      cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255,0,0), 2)
      # print label on the rectangle
      # frame image to use
      # labelmap - index of the ith class
      # (int(pt[0]), int(pt[1])) - where to put the text
      # cv2.FONT_HERSHEY_SIMPLEX - font to use
      # text size
      # color
      # thickness
      # LINE_AA - display cont line
      cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
      j += 1
  return frame

# creating the SSD Neural Network
# define a network and specify test mode
net = build_ssd('test') 
# load_state_dict() - attribute weights to the pre-trained network
# torch.load() - is to convert net into a tensor
# map_location = lambda storage, loc: storage - re-maps location of all tensors to CPU
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))

# defining the transformations
# net.size() - target size of the images to feed to the network
# scale values to make sure that call values are in the right scale
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

# Object Detection on a video
# imageio will process video
# get_reader - loads the video
reader = imageio.get_reader('funny_dog.mp4')
# get fps sequence
fps = reader.get_meta_data()['fps']
# crate a writer that will assemble back the video
writer = imageio.get_writer('output.mp4', fps = fps)

# loop through video frames and label detected objects
for i, frame in enumerate(reader):
  frame = detect(frame, net.eval(), transform)
  writer.append_data(frame)
  print(i)

writer.close()