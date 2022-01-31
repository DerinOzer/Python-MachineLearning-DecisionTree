import numpy as np;
import scipy as sp;
from matplotlib import pyplot as plt
from scipy import ndimage
import imageio
import skimage as sk;from skimage import measure
import os
from sklearn import *

training_directory="Learning/"
unknown_images_directory="Unknown Images/"
recognized_directory="Recognition/"

if not os.path.exists(recognized_directory) : os.mkdir(recognized_directory)

connectivity = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float64)

features_list,label_list=[],[]
label_to_name={}
label=0

def compute_features(img):
    measures = sk.measure.regionprops(img)
    allongement_objet = measures[0].major_axis_length / measures[0].minor_axis_length
    im_filled = sp.ndimage.morphology.binary_fill_holes(img) * 255
    labelled_tmp, nb_trou_objet = sp.ndimage.measurements.label((im_filled - img), connectivity)
    return allongement_objet,nb_trou_objet


for filename in os.listdir(training_directory):
    if filename[0]!='.':
        class_name=filename.split("_")[0]
        image_fullfilename=os.path.join(training_directory,filename)
        image=imageio.imread(image_fullfilename)
        axis_ratio,nb_holes=compute_features(image)
        features_list+=[np.array([axis_ratio,nb_holes])]
        label_to_name[label]=class_name
        label_list+=[np.asarray([label])]
        label+=1

label_array=np.vstack(tuple(label_list))
features_array=np.vstack(tuple(features_list))
clf=tree.DecisionTreeClassifier()
clf.fit(features_array,label_array)

for filename in os.listdir(unknown_images_directory):
    if filename[0] != '.':
        image_fullfilename=os.path.join(unknown_images_directory,filename)
        image=imageio.imread(image_fullfilename)
        axis_ratio, nb_holes = compute_features(image)
        label=clf.predict(np.array([[axis_ratio,nb_holes]]))[0]
        name=label_to_name[label]
        index=0
        output_fullfilename=os.path.join(recognized_directory, name+"_"+str(index)+".png")
        while os.path.isfile(output_fullfilename):
            index+=1
            output_fullfilename=os.path.join(recognized_directory, name+"_"+str(index)+".png")
        imageio.imsave(output_fullfilename,image)
