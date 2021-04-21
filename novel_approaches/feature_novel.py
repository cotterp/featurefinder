#Last Update: 04/21/21

import os
import time
from sys import getsizeof
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import zoom
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import csv

start = time.time()

def get_faces(filename):
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

def get_all_data(directory):
    dataset = {}
    if directory.find("_") >=0:
        cad_type = directory[directory.find("_")+1:]
    for filename in os.scandir(directory):
        
        if filename.is_file() and (filename.path.endswith(".STL")
                                   or filename.path.endswith(".stl")) :
            cad_type_no = os.path.splitext(filename.name)[0]
            dataset[cad_type, cad_type_no] = get_faces(filename)
        elif not filename.is_file():
            new_dataset = get_all_data(directory + "\\" + filename.name)
            for filename in new_dataset:
                dataset[filename] = new_dataset[filename]            
            
    return dataset

def centroid(points):
    centroid = []
    centroid.append((points[0]+points[3]+points[6])/3)
    centroid.append((points[1]+points[4]+points[7])/3)
    centroid.append((points[2]+points[5]+points[8])/3)
    return centroid

def return_line(centroids, axis, norm_points):
    #Returns an ordered set of (x,y,z) centroid points "as the finger runs"
    #across the specified axis.
       
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
    
    resample_ratio = norm_points/len(axis_points)
    norm_axis_points = zoom(axis_points, resample_ratio)
    norm_f1_points = zoom(f1_points, resample_ratio)
    norm_f2_points = zoom(f2_points, resample_ratio)
    
    return norm_axis_points, norm_f1_points, norm_f2_points

#Choose directory and use above functions to put all data into a dict
print("Compiling dataset...")
#Change to your directory
directory = r"C:\Users\micha\Documents\Python Scripts\MIT\2.169\dataset\stl"
dataset = get_all_data(directory)
compilation_runtime = round(time.time() - start, 4)
print('Dataset compiled in', compilation_runtime,'seconds.')
size = getsizeof(dataset)/1000000
print('Dataset size is', size, 'MB')

#Analyze the size of our various datasets
size_data = {}
for (cad_type, cad_type_no) in dataset.keys():
    if cad_type not in list(size_data.keys()):
        size_data[cad_type] = []
    size_data[cad_type].append(len(dataset[cad_type, cad_type_no]))
print("\n***Dataset characteristics***")
print("feature type [mean faces, min faces, max faces]\n")
size_analysis = {}
max_faces = 0
for cad_type in size_data.keys():
    size_analysis[cad_type] = []
    size_analysis[cad_type].append(np.mean(size_data[cad_type]))
    size_analysis[cad_type].append(np.min(size_data[cad_type]))
    size_analysis[cad_type].append(np.max(size_data[cad_type]))
    if max_faces < np.max(size_data[cad_type]):
        max_faces = np.max(size_data[cad_type])
    print(cad_type, size_analysis[cad_type])
print('\nMax faces is', max_faces)

#Reduce each point to centroid
centroids = {}
for (cad_type, cad_type_no) in dataset.keys():
    centroids[cad_type, cad_type_no] = []
    for points_set in dataset[cad_type, cad_type_no]:
        centroids[cad_type, cad_type_no].append(centroid(points_set))
        
# #Approach Number 1: Add in zeros to max_faces size vectors
# max_length = max_faces*3*3
# print('\nApproach 1: Sizing all vectors to', max_length,'length.')
# xoryorz_dataset = {}
# for (cad_type, cad_type_no) in dataset.keys():
#     xoryorz_dataset[cad_type, cad_type_no] = []
#     for vertex_set in dataset[cad_type, cad_type_no]:
#         for xoryorz in vertex_set:
#             xoryorz_dataset[cad_type, cad_type_no].append(xoryorz)
#     while len(xoryorz_dataset[cad_type, cad_type_no]) < max_length:
#         xoryorz_dataset[cad_type, cad_type_no].append(0)
approach_1_time = round(time.time() - start-compilation_runtime,4)
# print('Approach 1 complete in',approach_1_time,'seconds.')


#Approach Number 2: "Feeling with your finger" representation
print('\nApproach 2: Finding representations as the finger runs.')
finger_lines = {}
points_per_set = 100
for (feature, designator) in centroids.keys():
    finger_lines[(feature, designator)] = []
    x_line = return_line(centroids[(feature, designator)], 'x', points_per_set)
    y_line = return_line(centroids[(feature, designator)], 'y', points_per_set)
    z_line = return_line(centroids[(feature, designator)], 'z', points_per_set)
    representations = [x_line, y_line, z_line]
    for representation in representations:
        for line in representation:
            for value in line:
                finger_lines[(feature, designator)].append(value)
# plt.plot(finger_lines[0], finger_line[1], finger_line[0], finger_line[2])
#Write finger_lines to csv file
with open('finger_lines.csv', 'w', newline = '') as csvfile:
        finger_writer = csv.writer(csvfile, delimiter = ',')
        for (feature, designator) in finger_lines.keys():
            label = [feature, designator]
            finger_writer.writerow(label+finger_lines[(feature, designator)])
approach_2_time = round(time.time() - start - compilation_runtime - approach_1_time, 4)
print('Approach 2 complete in', approach_2_time, 'seconds.')

print('\nCreating train/test datasets...')
y = []
X = []
for (feature, designator) in finger_lines.keys():
    y.append(feature)
    X.append(finger_lines[(feature, designator)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
print('Dataset creation complete.')

train_start = time.time()
print('\nTraining neural network classifier...')
#https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier.predict
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(25, 25),
                    random_state=1, max_iter=200)
clf.fit(X_train, y_train)
train_time = round(time.time() -train_start, 4)
print('Neural network classifier training complete in', train_time, 'seconds.')
accuracy = clf.score(X_test, y_test)
print('Model accuracy is', accuracy)

confusion = {}
print('\nBuilding confusion matrix...')
features = list(np.unique(np.array(y_train)))
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
print('Confusion matrix built.')

total_end = time.time()
total_runtime = round(total_end - start, 4)
print('\n\nProgram complete in', total_runtime,'seconds.')