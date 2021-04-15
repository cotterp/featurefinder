import binvox_rw
import os
import sys
import numpy as np

#################################### input data ###########################################

path = ""
def rotations6(polycube):
    # imagine shape is pointing in axis 0 (up)

    # 4 rotations about axis 0
    yield polycube

    # rotate 180 about axis 1, now shape is pointing down in axis 0
    # 4 rotations about axis 0
    yield rot90(polycube, 2, axis=1)

    # rotate 90 or 270 about axis 1, now shape is pointing in axis 2
    # 8 rotations about axis 2
    yield rot90(polycube, axis=1)
    yield rot90(polycube, -1, axis=1)

    # rotate about axis 2, now shape is pointing in axis 1
    # 8 rotations about axis 1
    yield rot90(polycube, axis=0)
    yield rot90(polycube, -1, axis=0)

def rot90(m, k=1, axis=2):
    """Rotate an array by 90 degrees in the counter-clockwise direction around the given axis"""
    m = np.swapaxes(m, 2, axis)
    m = np.rot90(m, k)
    m = np.swapaxes(m, 2, axis)
    return m




def gen_binary(path):
    dataset=np.zeros((1,64*64*64+1))
    for file in os.listdir(path):
        suffix=file.split('.')[-1]
        if suffix=='binvox':
            name = file.split('_')
            label = np.array([int(name[0])])
            print(label)

            f = open(path+file, 'rb')
            model = binvox_rw.read_as_3d_array(f)
            model_num = model.data*1
            model_rot = rotations6(model_num)
            for rot_dir in model_rot:
                model_vector = np.reshape(rot_dir,(1,64*64*64))[0]
                data = np.append(label,model_vector)
                dataset=np.vstack((dataset,data))
            # print(np.shape(model_num))# 1/0
            # model_vector = np.reshape(model_num,(1,64*64*64))[0]
            #
            # data = np.append(label,model_vector)
            # dataset=np.vstack((dataset,data))

    dataset=dataset[1: ,:]
    dataset = np.array(dataset,dtype=np.uint8) # convert float64 to uint8
    np.random.shuffle(dataset)  # shuffle input
    return dataset


for i in range(0,11):
    inputPath=path+str(i)+"\\"
    # inputPath=path
    dataset = gen_binary(inputPath)
    print(dataset)
    print(np.shape(dataset))
    dataset.tofile(path+"dataset\\"+"test"+str(i)+".bin")
