import os
import time
from sys import getsizeof
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import zoom
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import csv
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

#####FingerLines CAD feature prediction#####
#Latest update: 05/16/2021
#The purpose of this script is to train a neural network to classify 
#features in CAD files. In this script, the raw CAD .stl files are loaded
#and the data is represented into finger lines. Two neural nets are built
#using this representation; one using sklearn and one using keras/tensorflow.
#Note: currently, this routine only supports classification of a single feature.

start = time.time()
#Change to target directory
directory = r"" #Insert path to directory with train and test samples
hard_ex_directory = r"" #Insert path to directory with hard test examples

def get_faces(filename):
    #This function gets the (x,y,z) coordinate pair of all faces in the input
    #.stl file.
    
    infile = open(filename) 
    data = infile.readlines()
    faces=[]
    for i,line in enumerate(data):
        if "outer loop" in line:
            vertices = []
            j = i + 1
            while j <= (i+3):
                vertex_line = data[j].split()
                vertex_x = vertex_line[1]
                vertex_y = vertex_line[2]
                vertex_z = vertex_line[3]
                vertices.append(float(vertex_x))
                vertices.append(float(vertex_y))
                vertices.append(float(vertex_z))
                j += 1
            faces.append(vertices)
    
    infile.close()
    return faces

def get_all_data(directory, hard_test=False):
    #This function reads in all .stl files in contained in the specified
    #directory.
    
    dataset = {}
    if directory.find("_") >=0:
        cad_type = directory[directory.find("_")+1:]
    for filename in os.scandir(directory):
        
        if filename.is_file() and (filename.path.endswith(".STL")
                                   or filename.path.endswith(".stl")) :
            if hard_test:
                cad_type_no = os.path.splitext(filename.name)[0]
                dataset[cad_type_no] = get_faces(filename)
            else:
                cad_type_no = os.path.splitext(filename.name)[0]
                dataset[cad_type, cad_type_no] = get_faces(filename)
        elif not filename.is_file():
            new_dataset = get_all_data(directory + "\\" + filename.name)
            for filename in new_dataset:
                dataset[filename] = new_dataset[filename]            
            
    return dataset

def centroid(points):
    #This functions the centroid of three (x,y,z) points.
    
    centroid = []
    centroid.append((points[0]+points[3]+points[6])/3)
    centroid.append((points[1]+points[4]+points[7])/3)
    centroid.append((points[2]+points[5]+points[8])/3)
    return centroid

def return_line(centroids, axis, norm_points):
    #Returns a normalized, truncated set of points (with noise added) "as the
    #finger runs" through the input axis.
       
    if axis.lower() == "x":
        axis = 0
        f1 = 1
        f2 = 2
    elif axis.lower() == "y":
        axis = 1
        f1 = 0
        f2 = 2
    elif axis.lower() == "z":
        axis = 2
        f1 = 0
        f2 = 1
    else:
        return "axis must be x, y, or x"
    
    sorted_centroids = sorted(centroids, key=lambda point: point[axis])
    axis_points = []
    f1_points = []
    f2_points = []
    for point in sorted_centroids:
        axis_point = point[axis]
        while axis_point in axis_points:
            axis_point += 0.0001
        axis_points.append(axis_point)
        f1_points.append(point[f1])
        f2_points.append(point[f2])
    
    #Adjusting point lists to same size
    resample_ratio = norm_points/len(axis_points)
    norm_axis_points = zoom(axis_points, resample_ratio)
    norm_f1_points = zoom(f1_points, resample_ratio)
    norm_f2_points = zoom(f2_points, resample_ratio)
    
    #Normalizing point lists to max of 1
    norm_axis_points = np.array(norm_axis_points)
    norm_axis_points = norm_axis_points/max(norm_axis_points)
    norm_f1_points = np.array(norm_f1_points)
    norm_f1_points = norm_f1_points/max(norm_f1_points)
    norm_f2_points = np.array(norm_f2_points)
    norm_f2_points = norm_f2_points/max(norm_f2_points)
    
    #Adding random Gaussian noise
    norm_axis_points += np.random.normal(0,0.01, norm_points)
    norm_f1_points += np.random.normal(0,0.01, norm_points)
    norm_f2_points += np.random.normal(0,0.01, norm_points)
    
    #Making all values between 0 and 1
    for points in [norm_axis_points, norm_f1_points, norm_f2_points]:
        points[points<0] = 0
        points[points>1] = 1
        
    return list(norm_axis_points), list(norm_f1_points), list(norm_f2_points)

#####Read in dataset#####
print("Compiling dataset...")
dataset = get_all_data(directory)
compilation_runtime = round(time.time() - start, 4)
print('Dataset compiled in', compilation_runtime,'seconds.')
# size = getsizeof(dataset)/1000000
# print('Dataset size is', size, 'MB')

#####Analysis of input dataset parameters#####
# size_data = {}
# for (cad_type, cad_type_no) in dataset.keys():
#     if cad_type not in list(size_data.keys()):
#         size_data[cad_type] = []
#     size_data[cad_type].append(len(dataset[cad_type, cad_type_no]))
# print("\n***Dataset characteristics***")
# print("feature type [mean faces, min faces, max faces]\n")
# size_analysis = {}
# max_faces = 0
# for cad_type in size_data.keys():
#     size_analysis[cad_type] = []
#     size_analysis[cad_type].append(np.mean(size_data[cad_type]))
#     size_analysis[cad_type].append(np.min(size_data[cad_type]))
#     size_analysis[cad_type].append(np.max(size_data[cad_type]))
#     if max_faces < np.max(size_data[cad_type]):
#         max_faces = np.max(size_data[cad_type])
#     print(cad_type, size_analysis[cad_type])
# print('\nMax faces is', max_faces)

#####Reduce all points to centroids#####
centroids = {}
for (cad_type, cad_type_no) in dataset.keys():
    centroids[cad_type, cad_type_no] = []
    for points_set in dataset[cad_type, cad_type_no]:
        centroids[cad_type, cad_type_no].append(centroid(points_set))
        
#####FingerLines Representation#####
print('\nApproach 2: Finding representations as the finger runs.')

#Save representations to finger_lines dict
finger_lines = {}
points_per_set = 100
features = []
for (feature, designator) in centroids.keys():
    if feature not in features:
        features.append(feature)
    finger_lines[(feature, designator)] = []
    x_line = return_line(centroids[(feature, designator)], 'x', points_per_set)
    y_line = return_line(centroids[(feature, designator)], 'y', points_per_set)
    z_line = return_line(centroids[(feature, designator)], 'z', points_per_set)
    representations = [x_line, y_line, z_line]
    for representation in representations:
        for line in representation:
            for value in line:
                finger_lines[(feature, designator)].append(value)

#Classification key
feature_nos = {'Oring': 0, 'through_hole': 1, 'blind_hole': 2,
              'triangular_passage': 3, 'rectangular_passage': 4,
              'circular_through_slot': 5, 'triangular_through_slot': 6,
              'rectangular_through_slot': 7, 'rectangular_blind_slot': 8,
              'triangular_pocket': 9, 'rectangular_pocket': 10,
              'circular_end_pocket': 11, 'triangular_blind_step': 12,
              'circular_blind_step': 13, 'rectangular_blind_step': 14,
              'rectangular_through_step': 15, '2sides_through_step': 16,
              'slanted_through_step': 17, 'chamfer': 18, 'round': 19,
              'v_circular_end_blind_slot': 20, 'h_circular_end_blind_slot': 21,
              '6sides_passage': 22,'6sides_pocket': 23}

#Make representation plots
designators = ['501', '575', '850']
figures = {}
axes = {}
for (feature_no, feature) in enumerate(features):
    figures[feature], axes[feature] = plt.subplots(nrows=3, ncols=3, figsize=(8,6))
    figures[feature].suptitle(feature)
    for (col, designator) in enumerate(designators):
        all_points = finger_lines[(feature, str(feature_nos[feature])+'_'+designator)]
        xaxis = all_points[0:99]
        xf1 = all_points[100:199]
        xf2 = all_points[200:299]
        yaxis = all_points[300:399]
        yf1 = all_points[400:499]
        yf2 = all_points[500:599]
        zaxis = all_points[600:699]
        zf1 = all_points[700:799]
        zf2 = all_points[800:899]
        axes[feature][0, col].plot(xaxis, xf1, xaxis, xf2)
        axes[feature][0, col].set_title(str(feature_nos[feature])+ '_' + designator + ' x')
        axes[feature][1, col].plot(yaxis, yf1, yaxis, yf2)
        axes[feature][1, col].set_title(str(feature_nos[feature])+ '_' + designator + ' y')
        axes[feature][2, col].plot(zaxis, zf1, zaxis, zf2)
        axes[feature][2, col].set_title(str(feature_nos[feature])+ '_' + designator + ' z')
    plt.tight_layout()
    plt.savefig(feature+'.png')

#Write finger_lines to csv file
with open('finger_lines.csv', 'w', newline = '') as csvfile:
        finger_writer = csv.writer(csvfile, delimiter = ',')
        for (feature, designator) in finger_lines.keys():
            label = [feature, designator]
            finger_writer.writerow(label+finger_lines[(feature, designator)])
approach_2_time = round(time.time() - start - compilation_runtime, 4)
print('Approach 2 complete in', approach_2_time, 'seconds.')

#####Creation of train/val/test datasets#####
print('\nCreating train/test datasets...')
y = []
X = []
for (feature, designator) in finger_lines.keys():
    y.append(feature)
    X.append(finger_lines[(feature, designator)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
print('Dataset creation complete.')

#####Sklearn DNN model#####
train_start = time.time()
print('\nTraining neural network classifier with sklearn...')
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(50, 50),
                    random_state=42, max_iter=200, validation_fraction=0.15,
                    activation='relu')
clf.fit(X_train, y_train)
train_time = round(time.time() - train_start, 4)
print('Sklearn DNN training complete in', train_time, 'seconds.')
accuracy = clf.score(X_test, y_test)
print('Model accuracy is', accuracy)

#####Keras DNN model#####
ker_train_start = time.time()
print('\nTraining neural network classifier with keras...')

#One-hot encoding
encoder = LabelEncoder()
encoder.fit(y)
y_encoded = encoder.transform(y)
y_oh = np_utils.to_categorical(y_encoded)
X_oh = np.array(X)
X_train_oh, X_test_oh, y_train_oh, y_test_oh = train_test_split(X_oh, y_oh, test_size=0.15, random_state=42)

#Model Definition and Training
ker = Sequential()
ker.add(Dense(50, input_dim=900, activation='relu'))
ker.add(Dense(50, activation='relu'))
ker.add(Dense(24, activation='softmax'))
ker.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
ker.fit(X_train_oh, y_train_oh, epochs=25, batch_size=10, validation_split=0.15)
_, ker_accuracy = ker.evaluate(X_test_oh, y_test_oh)
ker_train_time = round(time.time() - ker_train_start, 4)
print('Keras DNN training complete in', ker_train_time, 'seconds.')
print('Keras Model Accuracy:', ker_accuracy)

#####Confusion matrix build#####
confusion = {}
print('\nBuilding confusion matrix...')
for actual_feature in features:
    confusion[actual_feature] = {}
    for predicted_feature in features:
        confusion[actual_feature][predicted_feature] = 0
for (sample, actual_feature) in zip(X_test, y_test):
    predicted_feature = clf.predict(np.array(sample).reshape(1,-1))[0]
    confusion[actual_feature][predicted_feature] += 1

with open('confusion_matrix.csv', 'w', newline = '') as csvfile:
        confusion_writer = csv.writer(csvfile, delimiter = ',')
        confusion_writer.writerow([' ']+list(features))
        for actual_feature in confusion.keys():
            label = [actual_feature]
            values = [confusion[actual_feature][predicted_feature] for predicted_feature in features]
            confusion_writer.writerow(label+values)

#Generating plot
predictions = clf.predict(X)
confmat = confusion_matrix(y,predictions)
df_cm = pd.DataFrame(confmat, index = [i for i in range(1,25)],
                  columns = [i for i in range(1,25)])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, fmt='d')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print('Confusion matrix built.')

#####Saving DNN weights#####
print('\nSaving DNN weights...')
for (n, matrix) in enumerate(clf.coefs_):
    with open(str(n)+'_weights.csv', 'w', newline='') as weights_file:
        weights_writer = csv.writer(weights_file, delimiter=',')
        for row in matrix:
            weights_writer.writerow(row)

for (n, column) in enumerate(clf.intercepts_):
    with open(str(n)+'_bias.csv', 'w', newline='') as bias_file:
        bias_writer = csv.writer(bias_file, delimiter=',')
        for row in column:
            bias_writer.writerow([row])

#####Hard Test Examples#####
print('\nTrying hard test examples...')

#Reading in dataset and getting centroids
hard_dataset = get_all_data(hard_ex_directory)
hard_centroids = {}
for example in hard_dataset.keys():
    hard_centroids[example] = []
    for points_set in hard_dataset[example]:
        hard_centroids[example].append(centroid(points_set))

#Representing data as fingerlines
hard_finger_lines = {}
points_per_set = 100
for example in hard_centroids.keys():
    hard_finger_lines[example] = []
    x_line = return_line(hard_centroids[example], 'x', points_per_set)
    y_line = return_line(hard_centroids[example], 'y', points_per_set)
    z_line = return_line(hard_centroids[example], 'z', points_per_set)
    representations = [x_line, y_line, z_line]
    for representation in representations:
        for line in representation:
            for value in line:
                hard_finger_lines[example].append(value)

#Using sklearn DNN model to make predictions
total_samples = 0
correct_predictions = 0
hard_predictions = []
hard_true = []
for (example, designator) in hard_finger_lines.keys():
    feature_no = re.search('_(.*)_', designator)
    feature_no = feature_no.group(1)
    for (name, number) in feature_nos.items():
        if number == int(feature_no):
            feature_type = name
    x = np.array(hard_finger_lines[example, designator])
    x = x.reshape(1, -1)
    prediction = clf.predict(x)
    hard_predictions.append(prediction)
    hard_true.append(feature_type)
    total_samples += 1
    if feature_type == prediction[0]:
        correct_predictions += 1
        
print('Hard example accuracy is', round(correct_predictions/total_samples*100,2),'%.')

with open('hard_finger_lines.csv', 'w', newline = '') as csvfile:
        finger_writer = csv.writer(csvfile, delimiter = ',')
        for (feature, designator) in hard_finger_lines.keys():
            label = [feature, designator]
            finger_writer.writerow(label+hard_finger_lines[(feature, designator)])

hard_confmat = confusion_matrix(hard_true, hard_predictions)
unique_features = list(np.unique(list(np.unique(hard_true)) + list(np.unique(hard_predictions))))
hard_df_cm = pd.DataFrame(hard_confmat, index = [feature_nos[feature] for feature in unique_features].sort(),
                     columns = [feature_nos[feature] for feature in unique_features].sort())
hard_df_cm = hard_df_cm.sort_index(axis=0)
hard_df_cm = hard_df_cm.sort_index(axis=1)
plt.figure(figsize = (10,7))
sn.heatmap(hard_df_cm, annot=True, fmt='d')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#####Program end#####
total_end = time.time()
total_runtime = round(total_end - start, 4)
print('\n\nProgram complete in', total_runtime,'seconds.')