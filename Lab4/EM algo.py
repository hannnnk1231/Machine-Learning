import numpy as np
import struct

def idx3_decode(file):
    data=open(file, 'rb').read()
    # > for Big-endian, iiii for 4 integers, each size=4
    fmt='>iiii'
    offset=0
    magic_number, image_numbers, height, width=struct.unpack_from(fmt,data,offset)
    image_size=height*width
    offset+=struct.calcsize(fmt)
    # B for unsigned byte, size=1
    fmt='>'+str(image_size)+'B'
    images=np.empty((image_numbers,height*width))
    for i in range(image_numbers):
        images[i]=np.array(struct.unpack_from(fmt,data,offset)).reshape((height*width))
        offset+=struct.calcsize(fmt)
    return images,image_numbers

def idx1_decode(file):
    data=open(file, 'rb').read()
    # > for Big-endian, ii for 2 integers, each size=4
    fmt='>ii'
    offset=0
    magic_number, label_numbers=struct.unpack_from(fmt,data,offset)
    offset+=struct.calcsize(fmt)
    # B for unsigned byte, size=1
    fmt='>B'
    labels=np.empty(label_numbers)
    for i in range(label_numbers):
        labels[i]=struct.unpack_from(fmt,data,offset)[0]
        offset+=struct.calcsize(fmt)
    return labels,label_numbers

def E_step(X,P,W):
    for n in range(60000):
        temp = lamda.copy()
        for k in range(10):
            for d in range(784):
                if (X[n,d]==1):
                    if(P[d,k]==0):
                        temp[k] *= 0.0001
                    else: 
                        temp[k] *= P[d,k]
                else:
                    if(P[d,k]==1):
                        temp[k] *= 0.0001
                    else:
                        temp[k] *= 1-P[d,k]
        for k in range(10):
            if(np.sum(temp)==0):
                W[n,k] = temp[k]/0.0001
            else:
                W[n,k] = temp[k]/np.sum(temp)
    return W

def M_step(X,W,P,lamda):
    sigma_w = np.sum(W,axis=0)
    lamda = sigma_w/60000
    for k in range(10):
        for d in range(784):
            P[d][k] = np.dot(np.transpose(X)[d],np.transpose(W)[k])
            if(sigma_w[k]==0):
                P[d][k] /= 0.0001
            else:
                P[d][k] /= sigma_w[k]
    return P,lamda

def print_imagination():
    for k in range(10):
        print('\nclass {}:'.format(k))
        for d in range(784):
            if d%28==0 and d!=0:
                print('')
            if P[d,k]>0.5:
                print('1',end='')
            else:
                print('0',end='')

def print_labeled_imagination(r):
    for i,k in enumerate(r):
        print('labeled class {}:'.format(i))
        for d in range(784):
            if d%28==0 and d!=0:
                print('')
            if P[d,int(k)]>0.5:
                print('1',end='')
            else:
                print('0',end='')
        print('\n')
    print("\n----------------------------------------------------")

def confusion(r):
    confusion_matrix = np.zeros((10,3))
    error = 0
    for n in range(60000):
        temp = lamda.copy()
        for k in range(10):
            for d in range(784):
                if (X[n,d]==1):
                    if(P[d,k]==0):
                        temp[k] *= 0.0001
                    else: 
                        temp[k] *= P[d,k]
                else:
                    if(P[d,k]==1):
                        temp[k] *= 0.0001
                    else:
                        temp[k] *= 1-P[d,k]
        temp_index, = np.where(r==np.argmax(temp))[0]
        if(int(train_label[n])==temp_index):
            confusion_matrix[int(train_label[n]),0]+=1
        else:
            confusion_matrix[int(train_label[n]),1]+=1
            confusion_matrix[temp_index,2]+=1
    for k in range(10):
        print("\nConfusion Matrix:") 
        print("                Predict number {} Predict not number {}".format(k,k))
        print("Is number {}           {}                {}".format(k,confusion_matrix[k,0],confusion_matrix[k,1]))
        print("Isn\'t number {}       {}                {}".format(k,confusion_matrix[k,2],60000-np.sum(confusion_matrix[k])))
        print("\nSensitivity (Successfully predict number {}): {:.5f}".format(k,confusion_matrix[k,0]/(confusion_matrix[k,0]+confusion_matrix[k,1])))
        print("Specificity (Successfully predict not number {}): {:.5f}".format(k,(60000-np.sum(confusion_matrix[k]))/(confusion_matrix[k,2]+60000-np.sum(confusion_matrix[k]))))
        print("\n----------------------------------------------------")
        error+=confusion_matrix[k,1]+confusion_matrix[k,2]
    return error

def clustering():
    table = np.zeros((10,10))
    label_class_relation = np.zeros(10)
    for n in range(60000):
        temp = lamda.copy()
        for k in range(10):
            for d in range(784):
                if (X[n,d]==1):
                    if(P[d,k]==0):
                        temp[k] *= 0.0001
                    else: 
                        temp[k] *= P[d,k]
                else:
                    if(P[d,k]==1):
                        temp[k] *= 0.0001
                    else:
                        temp[k] *= 1-P[d,k]
        table[int(train_label[n]),np.argmax(temp)]+=1
    for k in range(10):
        index = np.unravel_index(np.argmax(table, axis=None), table.shape)
        label_class_relation[index[0]] = index[1]
        for j in range(0, 10):
            table[index[0]][j] = 0
            table[j][index[1]] = 0
    print_labeled_imagination(label_class_relation)
    return confusion(label_class_relation)


train_image_path='train-images.idx3-ubyte'
train_label_path='train-labels.idx1-ubyte'
train_image,train_image_number=idx3_decode(train_image_path)
train_label,train_label_number=idx1_decode(train_label_path)
train_image = train_image//128

X = train_image.copy()
lamda = np.full((10,1),0.1,dtype=np.float64) # init prob for every class
P = np.random.rand(28*28,10) # init prob for every pixel of every class
P_prev = P.copy()
W = np.zeros((60000,10)) # init w for every pic for every class 

iteration = 0
while(1):
    iteration += 1
    W = E_step(X,P,W)
    P,lamda = M_step(X,W,P,lamda)
    print_imagination()
    diff = np.linalg.norm(P-P_prev)
    print("\nNo. of Iteration: {}, Difference: {}".format(iteration, diff))
    print("\n----------------------------------------------------")
    if diff < 0.001:
        break
    P_prev = P
print("\n----------------------------------------------------")
print("----------------------------------------------------")
error = clustering()
print('Total iteration to coverage: {}'.format(iteration))
print('Total error rate: {}'.format(error/600000))