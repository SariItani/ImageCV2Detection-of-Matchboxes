# template matching with non-max suppression (NMS)
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import subprocess

import glob

folder = './'
file_prefix = 'image?'
file_ext = '.jpg'

flist = glob.glob(folder+file_prefix+file_ext)


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
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
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


def matchTemplate(img_gray, template, threshold):
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    # x = loc[1] and y = loc[0]
    w, h = template.shape[::-1]
    coords = np.dstack([loc[1], loc[0], loc[1]+w, loc[0]+h]).squeeze()
    return coords


def prepareTemplates(template_folder, delta):
    template_list = []
    # Get all PNG files in the template folder
    template_files = glob.glob(template_folder + "/*.png")

    for template_file in template_files:
        # Read template image in grayscale
        template = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)
        template_list.append(template)

        for angle in range(delta, 360, delta):
            rotated_template = rotate_image(template, angle)
            template_list.append(rotated_template)

    return template_list


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 2 dimensions
    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origin) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(
        mat, rotation_mat, (bound_w, bound_h), borderMode=cv2.BORDER_TRANSPARENT)
    return rotated_mat


template_folder = 'templates'  # Folder containing the PNG template images
delta = 45  # Rotation increment

template_list = prepareTemplates(template_folder, delta)

# Now, you have the template images in the template_list for comparison with objects in the input image.


# it creates a list of template images rotated with all
# useful angles for extensive object matching
def makeTemplates(filename, delta):
    template = cv2.imread(filename)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_list = []
    for alpha in range(0, 360, delta):
        # rotate templates with angles
        rotated = rotate_image(template, alpha)
        template_list.append(rotated)

    return template_list


def makeBoxes(boxes, img):
    box_color = (0, 0, 255)  # BGR
    radius = 5
    thickness = 3
    x_offs = 10
    y_offs = -5

    centers = []
    i = 1
    for l, t, r, b in boxes:
        l = int(l)
        t = int(t)
        r = int(r)
        b = int(b)
        # replace with a bullet in the cluster center
        cv2.rectangle(img, (l, t), (r, b), box_color, 2)

        center_tuple = (int((l+r)/2), int((t+b)/2))
        cv2.circle(img, center_tuple, radius=radius,
                   color=box_color, thickness=-1)
        org = (center_tuple[0] + x_offs, center_tuple[1] + y_offs)
        cv2.putText(img, str(i), org, cv2.FONT_HERSHEY_SIMPLEX,
                    3, box_color, thickness, cv2.LINE_AA)
        i += 1
        centers.append(center_tuple)

    return centers


SELECTED = 3
threshold = 0.3
delta = 90

# read template file and create all required rotations
templatefile_0 = 'box_template1.png'
templatefile_1 = 'box_template_back1.png'
t_list_0 = makeTemplates(templatefile_0, delta)
t_list_1 = makeTemplates(templatefile_1, delta)

t_list = t_list_0 + t_list_1

#t_list = makeTemplates(templatefile, delta)
#t_list.append(cv2.imread('box_template2.png', 0))
#t_list.append(cv2.imread('box_template3.png', 0))
#t_list.append(cv2.imread('box_template4.png', 0))
#t_list.append(cv2.imread('box_template5.png', 0))
#t_list.append(cv2.imread('box_template6.png', 0))
#t_list.append(cv2.imread('box_template7.png', 0))
#t_list.append(cv2.imread('box_template8.png', 0))
#t_list.append(cv2.imread('box_template9.png', 0))
#t_list.append(cv2.imread('box_template10.png', 0))


# replace the following 2 lines with camera grabbing
imagefile = 'image6.jpg'
img_rgb = cv2.imread(imagefile)


# convert image to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# initialize empty box array to be filled with detected
# boxes coordinates before non-maximum suppression
boxes_found = np.empty([0, 4])
for template in t_list:
    coords = matchTemplate(img_gray, template, threshold)
    # add detected boxes to the array no matter how many they are
    boxes_found = np.vstack((boxes_found, coords))

# perform non-maximum suppression to remove non-optimal boxes
boxes = NMSF(np.array(boxes_found), 0.1)
# boxes = boxes_found

# superimpose boxes to original image and return the centers list
center_list = makeBoxes(boxes, img_rgb)

# Show result
plt.figure(figsize=(20, 7))
plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))

# Draw bounding boxes and centers
for box, center in zip(boxes, center_list):
    l, t, r, b = box
    cv2.rectangle(img_rgb, (l, t), (r, b), (0, 0, 255), 2)
    cv2.circle(img_rgb, center, radius=5, color=(0, 0, 255), thickness=-1)

plt.axis('off')
plt.show()
