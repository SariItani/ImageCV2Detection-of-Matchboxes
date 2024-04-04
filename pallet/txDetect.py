#!/usr/bin/env python
# coding: utf-8

# required packages
import socket
import sys
import time
import getopt
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import glob

# default configuration options
srv_addr = '127.0.0.1'   # server address
port = 19810             # server port
default_camera = 0       # camera device index
FRAME_WIDTH = 2592       # camera max width
FRAME_HEIGHT = 1944      # camera max height
debug = False            # control flag for debugging messages
save_folder = './grabs/'           # relative path for captured images
template_folder = './templates/'   # relative path for templates
file_prefix = 'pallet*'            # name pattern of template files
file_ext = '.jpg'                  # template files extension
scaleRange = [1.0,0.9,0.8]         # list of scale factors
gaussian_sigma = 9                 # sigma value for gaussian filtering
delta = 45                         # angle increment for rotated templates
threshold = 0.5                    # threshold for image matching function
RECV_BUF_SIZE = 8        # socket reply buffer size in byte
ACK_MSG = "DONE"         # confirmation message for robot action complete
EMPTY_MSG = "EMPTY"      # message to notify empty pallet to be replaced

# help function for program usage
def usage():
    print("usage: {} -h -d -s server_address -p server_port -t threshold -r angle".format(sys.argv[0]))
    print("h: display usage help")
    print("d: debug option to save all grabbed images with relevant detections")
    print("s: specify the server address; default option is localhost")
    print("p: specify the server port; default option is 19810")
    print("t: threshold for pattern matching; default option is 0.5")
    print("r: rotation angle for templates; default option is 45 degrees")
    sys.exit(2)

# non-maximum suppression algorithm
# Malisiewicz et al.
def NMSF(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

def matchTemplate(img_gray,template,threshold):
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    # x = loc[1] and y = loc[0]
    w, h = template.shape[::-1]
    coords = np.dstack([loc[1],loc[0],loc[1]+w,loc[0]+h]).squeeze()
    return coords

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    # getRotationMatrix2D needs coordinates in reverse order 
    # (width, height) compared to shape
    image_center = (width/2, height/2) 

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) 
    # and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), borderMode=cv2.BORDER_TRANSPARENT)
    return rotated_mat

# makeTemplates creates a list of template images rotated with all
# useful angles as of delta and scaling for extensive object matching
def makeTemplates(filelist,delta):
    template_list = []
    for filename in filelist:
        template = cv2.imread(filename)
        template = ndimage.gaussian_filter(template, sigma=gaussian_sigma)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        for scale in scaleRange:
            dim = ((int)(template.shape[1]*scale),(int)(template.shape[0]*scale))
            scaled = cv2.resize(template,dim)
            for alpha in range(0,360,delta):
                # rotate templates with angles
                rotated = rotate_image(scaled,alpha)
                template_list.append(rotated)
    
    return template_list
        
def getCenters(boxes,img):
    box_color = (255,0,0) # BGR
    radius = 15
    thickness = 3
    x_offs = 10
    y_offs = -5
    
    centers = []
    i = 1
    for l,t,r,b in boxes:
        l = int(l)
        t = int(t)
        r = int(r)
        b = int(b)
        # replace with a bullet in the cluster center
        #cv2.rectangle(img, (l,t), (r,b), box_color, 2)
        
        center_tuple = (int((l+r)/2),int((t+b)/2))
        cv2.circle(img, center_tuple, radius=radius, color=box_color, thickness=-1)
        org = (center_tuple[0] + x_offs, center_tuple[1] + y_offs)
        cv2.putText(img, str(i), org, cv2.FONT_HERSHEY_SIMPLEX, 3, box_color, thickness, cv2.LINE_AA)
        i += 1
        centers.append(center_tuple)
    
    return centers, img

def save_frame(newFrame):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    savefile = "{}grab{}.png".format(save_folder,timestr)
    cv2.imwrite(savefile,cv2.cvtColor(newFrame,cv2.COLOR_BGR2RGB))

def map2byte(centers_list):
    # converts the list of tuples representing centers into
    # bytes in utf-8 format for network transmission
    ct_str = ""
    for ctuple in centers_list: 
        ct_str += str(ctuple)+"-"
    byt = str(ct_str[:-1]).encode('utf-8')
    return byt

def process_frame(newFrame, t_list, debug):
    if debug:
        save_frame(newFrame)

    # find pattern and centers
    blurred = ndimage.gaussian_filter(newFrame, sigma=gaussian_sigma)

    # convert image to grayscale
    img_gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # initialize to none in order to send data over the net only
    # when some centers have been identified
    center_list = None

    # initialize empty box array to be filled with detected
    # boxes coordinates before non-maximum suppression
    boxes_found = np.empty([0,4])
    
    for template in t_list:
        coords = matchTemplate(img_gray,template,threshold)
        # add detected boxes to the array no matter how many they are
        boxes_found = np.vstack((boxes_found,coords))

    # perform non-maximum suppression to remove non-optimal boxes
    boxes = NMSF(np.array(boxes_found),0.1)
    if debug:
        cnt = 0 if len(boxes)==0 else boxes.shape[0]
        print("Number of boxes: {}".format(cnt))

    # superimpose boxes to original image and return the centers list
    if debug:
        print("Saving overlay")
        center_list, overlay = getCenters(boxes,newFrame)
        # save current frame to file
        save_frame(overlay)
    else:
        center_list, _ = getCenters(boxes,newFrame)

    return center_list

# parse command line for custom options
try:
    options, arguments = getopt.getopt(sys.argv[1:],'hds:p:t:r:',['help','debug','server=','port=','threshold=','rotation='])
    #assert len(sys.argv) > 1
except:
    usage()

for opt, arg in options:
    if opt == '-h':
        usage()
    if opt == '-d':
        debug = True
        print("Debug mode selected")
    elif opt in ("-s", "--server"):
        srv_addr = arg
    elif opt in ("-p", "--port"):
        port = int(arg)
    elif opt in ("-t", "--threshold"):
        threshold = float(arg)
    elif opt in ("-r", "--rotation"):
        delta = int(arg)
    else:
        usage()

# read the templates for pattern matching
flist = glob.glob(template_folder+file_prefix+file_ext)
if debug: print("Number of template files {}".format(len(flist)))

# read template files from list and create all required rotations
t_list = makeTemplates(flist,delta)
if debug: print("Number of template images {}".format(len(t_list)))

# get an image, detect boxes and send them over a TCP socket

readNewFrames = True  # set to false to stop grabbing
while readNewFrames:
    try:
        # initialize a capture device
        if debug: print("Capturing new frame ...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        # Capture frame
        ret, frame = cap.read()
        cap.release()
        if debug: print("Processing new frame ...")
        if ret:
            centers = process_frame(frame,t_list,debug)
            # send centers over the net to server application
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((srv_addr,port))

                if len(centers)>0:
                    # if centers have been detected, they are sent to robot
                    byt = map2byte(centers)
                    if debug: print("Sending centers over the net")
                else:
                    # if no centers are listed, empty pallet message is sent
                    byt = EMPTY_MSG.encode('utf-8')
                    if debug: print("Notifying EMPTY pallet ")

                nb = s.sendall(byt)
                if nb == 0:
                    print("error: no data sent")
                else:
                    # wait for robot confirmation or pallet replacement 
                    repl = s.recv(RECV_BUF_SIZE)
                    if repl.decode('utf-8') != ACK_MSG:
                        # force application to quit
                        raise KeyboardInterrupt
                s.close()
        else:
            print("Unable to access camera")
            readNewFrames = False
    except KeyboardInterrupt:
        readNewFrames = False
        # cap.release()
        # s.close()

sys.exit(0)

