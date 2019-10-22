import struct
import numpy as np
import matplotlib.pyplot as plt

def print_result(posterior,ans):
    print('Posterior (in log scale):')
    for i in range(10):
        print(i,': ',posterior[i])
    print('Prediction: {}, Ans: {}\n'.format(np.argmin(posterior),int(ans)))

def print_imagination_continuous(mean):
    print('Imagination of numbers in Bayesian classifier:\n')
    for i in range(10):
        print(i,':')
        for p in range(28*28):
            if mean[i][p]<128:
                print('0', end=' ')
            else:
                print('1', end=' ')
            if p%28==27:
                print('')
        print('')

def print_imagination(train_result):
    print('Imagination of numbers in Bayesian classifier:\n')
    for i in range(10):
        print(i,':')
        for p in range(28*28):
            no=yes=0
            for j in range(16):
                if(train_result[i][p][j]!=0):
                    no+=train_result[i][p][j]
                else:
                    no+=0.0001/train_label_count[i]
            for j in range(16,32):
                if(train_result[i][p][j]!=0):
                    yes+=train_result[i][p][j]
                else:
                    yes+=0.0001/train_label_count[i]
            if no>yes:
                print('0', end=' ')
            else:
                print('1', end=' ')
            if p%28==27:
                print('')
        print('')

def test_continuous(mean,varience,N):
    error=0
    for i in range(N):
        posterior=np.copy(prior)
        for l in range(10):
            for p in range(28*28):
                posterior[l]+=np.log(1/np.sqrt(2*np.pi*varience[l][p]))-(((test_image[i][p]-mean[l][p])**2)/(2*varience[l][p]))-np.log(train_label_count[l])
        posterior/=np.sum(posterior)
        print_result(posterior,test_label[i])
        if(np.argmin(posterior)!=test_label[i]):
            error+=1
    print_imagination_continuous(mean)
    print('Error rate: ',float(error/N))

def train_continuous(N):
    mean=np.zeros((10,28*28))
    varience=np.zeros((10,28*28))
    for i in range(N):
        for p in range(28*28):
            mean[int(train_label[i])][p]+=train_image[i][p]
            varience[int(train_label[i])][p]+=train_image[i][p]**2
    for l in range(10):
        mean[l]=mean[l]/train_label_count[l]
        varience[l]/=train_label_count[l]
        varience[l]-=mean[l]**2
        for p in range(28*28):
            if(varience[l][p]==0):
                varience[l][p]=2250
    return mean, varience

def test_discrete(train_result,N):
    error=0
    for i in range(N):
        posterior=np.copy(prior)
        for label in range(10):
            for p in range(28*28):
                if(train_result[label][p][int(test_image[i][p]//8)]!=0):
                    posterior[label]+=np.log(train_result[label][p][int(test_image[i][p]//8)])
                else:
                    posterior[label]+=np.log(0.0001/train_label_count[label])
        posterior=posterior/np.sum(posterior)
        print_result(posterior,test_label[i])
        if(np.argmin(posterior)!=test_label[i]):
            error+=1
    print_imagination(train_result)
    print('Error rate: ',float(error/N))

def train_discrete(N):
    image_bin_count=np.zeros((10,28*28,32))
    probability=np.zeros((10,28*28,32))
    for i in range(N):
        for p in range(28*28):
            image_bin_count[int(train_label[i])][p][int(train_image[i][p]//8)]+=1
    for label in range(10):
        probability[label]=image_bin_count[label]/train_label_count[label]
    return probability

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

train_image='train-images.idx3-ubyte'
train_label='train-labels.idx1-ubyte'
test_image='t10k-images.idx3-ubyte'
test_label='t10k-labels.idx1-ubyte'

train_image,train_image_number=idx3_decode(train_image)
test_image,test_image_number=idx3_decode(test_image)
train_label,train_label_number=idx1_decode(train_label)
test_label,test_label_number=idx1_decode(test_label)

train_label_count=np.zeros(10)
for label in train_label:
    train_label_count[int(label)]+=1
    
prior=np.log(train_label_count/train_label_number)

if(int(input('Toggle option: (0 for discrete mode, 1 for continuous mode) \n'))):
    mean,varience=train_continuous(train_image_number)
    test_continuous(mean,varience,test_image_number)
else:
    train_result=train_discrete(train_image_number)
    test_discrete(train_result,test_image_number)
