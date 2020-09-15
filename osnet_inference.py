'''
@LineStart: -------------------------------------------
@Copyright: © 2020, Pensees. All rights reserved.
@LineEnd: ----------------------------------------------
@Product: 
@Mode Name: 
@Author: [mahxn0]
@Date: 2020-09-01 18:41:53
@LastEditors: [mahxn0]
@LastEditTime: 2020-09-07 14:51:01
@Description: 
'''
import caffe
import cv2
import numpy as np
import math
caffe.set_mode_cpu()
class Osnet(object):
    def __init__(self):
        super().__init__()
        #self.net = caffe.Net("./osnet/osnet_0x_75.prototxt", "./osnet_0x_75.caffemodel", caffe.TEST)
        self.net = caffe.Net("./osnet/osnet_0x_25.prototxt",
                             "./osnet/osnet_0x_25.caffemodel", caffe.TEST)
        #conv1 = np.load("./conv1.npy")
        #print(conv1)
        #print(type(self.net.params['conv1'][0].data))
        #self.net.params['conv1'][0].data[...] = conv1
        
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))
        #self.transformer.set_mean('data', np.array([0.5, 0.5, 0.5]))
        self.net.blobs['data'].reshape(1,        # batch size
                                       3,         # 3-channel (BGR) images
                                       128,64)
    def forward(self,image):
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', image)
        #print(self.net.blobs['data'].data[...])
        out = self.net.forward()
        output = out['relu_blob101'][0]
        data = self.net.blobs['view_blob1'].data
        n=data.shape[0]
        c=data.shape[1]
        #h=data.shape[2]
        #w=data.shape[3]
        for _c in range(c):
            #for _h in range(h):
                print(data[0,:])
        #print(conv1_w)
        #print(self.net.params['bn_scale80'][0].data)
        return output
        

osnet = Osnet()
image = cv2.imread("./6025.jpg")
image =cv2.resize(image,(128,64))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = (image/256-0.5) / 0.5
#image=np.around(image,4)
output1 = osnet.forward(image)
output1=np.around(output1,4)
norm1 = np.linalg.norm(output1)
#print(output1)
print(norm1) 

# image2 = cv2.imread("./6025.jpg")
# #image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
# image2 = (image2 / 255 - 0.5) / 0.5
# image2 = np.around(image2, 4)
# print(image2)
# output2 = osnet.forward(image2)
# output2 = np.around(output2,4)
# #print(output2)
# norm2 = np.linalg.norm(output2)
# print(norm2)
# sim = np.dot(output1,output2.T)/(norm1*norm2)
# print("相似度=", sim)
