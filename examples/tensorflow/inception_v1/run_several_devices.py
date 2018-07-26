#! /usr/bin/env python3

# Copyright 2017 Intel Corporation. 
# The source code, information and material ("Material") contained herein is  
# owned by Intel Corporation or its suppliers or licensors, and title to such  
# Material remains with Intel Corporation or its suppliers or licensors.  
# The Material contains proprietary information of Intel or its suppliers and  
# licensors. The Material is protected by worldwide copyright laws and treaty  
# provisions.  
# No part of the Material may be used, copied, reproduced, modified, published,  
# uploaded, posted, transmitted, distributed or disclosed in any way without  
# Intel's prior express written permission. No license under any patent,  
# copyright or other intellectual property rights in the Material is granted to  
# or conferred upon you, either expressly, by implication, inducement, estoppel  
# or otherwise.  
# Any license under such intellectual property rights must be express and  
# approved by Intel in writing.

from mvnc import mvncapi as mvnc
import sys
import numpy
import cv2
from multiprocessing import Process, Queue as PQueue
from threading import Thread
from os import listdir
from os.path import isfile
from os.path import join 
import time

path_to_networks = './'
path_to_images = '../../data/images/'
graph_filename = 'graph'
#image_filename = path_to_images + 'nps_electric_guitar.png'


def getImg(image_filename):
    global path_to_networks

    #Load image size
    with open(path_to_networks + 'inputsize.txt', 'r') as f:
        reqsize = int(f.readline().split('\n')[0])
    #Load preprocessing data
    mean = 128 
    std = 1/128
    #Load image size
    with open(path_to_networks + 'inputsize.txt', 'r') as f:
        reqsize = int(f.readline().split('\n')[0])

    img = cv2.imread(image_filename).astype(numpy.float32)      
    img = cv2.resize(img, (reqsize, reqsize))
    return img


def getImages():
    global path_to_images, path_to_networks
    onlyfiles = [f for f in listdir(path_to_images) if isfile(join(path_to_images, f))]
    result = numpy.zeros((len(onlyfiles), 224, 224, 3))
    print("Batch len {}".format(len(onlyfiles)))
    for i, filename in enumerate(onlyfiles):
        image_filename = path_to_images + filename
        print(image_filename)
        img = getImg(image_filename)
        result[i] = img
    return result


def read_categories():
    #Load categories
    categories = []
    with open(path_to_networks + 'categories.txt', 'r') as f:
        for line in f:
            cat = line.split('\n')[0]
            if cat != 'classes':
                categories.append(cat)
        f.close()
        print('Number of categories:', len(categories))
    return categories



def runNCS():
    global path_to_networks, graph_filename
    #mvnc.SetGlobalOption(mvnc.GlobalOption.LOGLEVEL, 2)
    devHandle   = []
    graphHandle = []
    dispQ = []

    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print('No devices found')
        quit()
    else:
        print("There was found {}".format(len(devices)))

    # Create n Queues based on the number of sticks plugged in
    for devnum in range(len(devices)):
        dispQ.append(PQueue())

    #Load graph
    with open(path_to_networks + graph_filename, mode='rb') as f:
        graphfile = f.read()

    # *****************************************************************
    # Open the device and load the graph into each of the devices
    # *****************************************************************
    for devnum in range(len(devices)):
        print("***********************************************")
        devHandle.append(mvnc.Device(devices[devnum]))
        devHandle[devnum].OpenDevice()

        opt = devHandle[devnum].GetDeviceOption(mvnc.DeviceOption.OPTIMISATIONLIST)
        print("Optimisations:")
        print(opt)

        graphHandle.append(devHandle[devnum].AllocateGraph(graphfile))
        graphHandle[devnum].SetGraphOption(mvnc.GraphOption.ITERATIONS, 1)
        iterations = graphHandle[devnum].GetGraphOption(mvnc.GraphOption.ITERATIONS)
        print("Iterations:", iterations)

    print("***********************************************")
    print("Loaded Graphs")
    print("***********************************************\n\n\n")
    return devices, graphHandle, dispQ


def runparallel(imgarr, devices, graphHandle, dispQ=0):
    #print("STARTED RUNPARALLEL")
    numdevices = range(len(devices))
    numimgs = len(imgarr)

    batch_size = 3

    if len(devices) >=2:
        start_time = time.time()
        for devnum in range(len(devices)):
            #print("DEVICE", devnum, "loading data!")
            graphHandle[devnum].LoadTensor(imgarr[devnum*batch_size:(devnum+1)*batch_size], "user object")
            print(imgarr[devnum*batch_size:(devnum+1)*batch_size].shape)
        # *****************************************************************
        # Read the result from each of the devices
        # *****************************************************************
        print("Before done")
        for devnum in numdevices:
            tensor, userobj = graphHandle[devnum].GetResult()
            print("devnum {}".format(devnum))
        print("DONE")
        end_time = time.time()
        print("Total time {}".format(end_time - start_time))

    if len(devices) == 1:
        numdevices = 2
        start_time = time.time()
        for devnum in range(numdevices):
            #print("DEVICE", devnum, "loading data!")
            graphHandle[0].LoadTensor(imgarr[devnum*batch_size:(devnum+1)*batch_size], "user object")
            print(imgarr[devnum*batch_size:(devnum+1)*batch_size].shape)
            tensor, userobj = graphHandle[0].GetResult()
            print("devnum {}".format(devnum))
        print("DONE")
        end_time = time.time()
        print("Total time {}".format(end_time - start_time))


images = getImages()
devices, graphHandle, dispQ = runNCS()
runparallel(images, devices, graphHandle, dispQ)


def old():
    print('Start download to NCS...')
    graph.LoadTensor(img.astype(numpy.float16), 'user object')
    output, userobj = graph.GetResult()

    top_inds = output.argsort()[::-1][:5]

    print(''.join(['*' for i in range(79)]))
    print('inception-v1 on NCS')
    print(''.join(['*' for i in range(79)]))
    for i in range(5):
        print(top_inds[i], categories[top_inds[i]], output[top_inds[i]])

    print(''.join(['*' for i in range(79)]))
    graph.DeallocateGraph()
    device.CloseDevice()
    print('Finished')