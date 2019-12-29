import imageio
import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel(X,image_number):
    sigma_s = -0.02
    sigma_c = -0.02
    W = np.zeros((10000,10000))
    for i in range(10000):
        x1 = i//100
        y1 = i%100
        for j in range(i+1,10000):
            x2 = j//100
            y2 = j%100
            if image_number==1:
                w = np.exp(sigma_s*((x1-x2)**2+(y1-y2)**2))*np.exp(sigma_c*(np.linalg.norm(im[x1,y1]-im[x2,y2])**2))
            else:
                w = np.exp(sigma_s*((x1-x2)**2+(y1-y2)**2))*np.exp(sigma_c*(np.linalg.norm(im2[x1,y1]-im2[x2,y2])**2))
            W[i,j] = W[j,i] = w
    return W

def init(U,method,k,image_number):
    means = np.zeros((k,k))
    if method == 0:
        print('Init method: random')
        temp = np.random.randint(10000,size=k)
        for i in range(k):
            means[i] = U[temp[i]]
    else:
        print('Init method: kmeans++')
        temp = np.random.randint(10000)
        means[0] = U[temp]
        for i in range(1,k):
            dist = np.zeros(10000)
            for j in range(10000):
                temp = np.zeros(i)
                for p in range(i):
                    temp[p] = np.linalg.norm(U[j]-means[p])
                dist[j] = np.min(temp)
            dist = dist/dist.sum()
            temp = np.random.choice(10000,1,p=dist)
            means[i,:] = U[temp,:]
    return means

def classify(U,means,k,image_number):
    res = np.zeros(10000)
    for i in range(10000):
        temp = np.zeros(k)
        for j in range(k):
            temp[j] = np.linalg.norm(U[i]-means[j])
        res[i] = np.argmin(temp)
    return res

def update(U,res,k,image_number):
    means = np.zeros((k,k))
    count = np.zeros(k)
    for i in range(10000):
        cluster = int(res[i])
        count[cluster] += 1
        means[cluster] += U[i]
    for i in range(k):
        means[i] /= count[i]
    return means

def draw(res,image_number,iteration,k,method,cut):
    if method == 0:
        method = 'random'
    else:
        method = 'kmeans++'
    plt.title("Spectral, {} means after {} iters, Image {}\ninit method: {}, cut method: {}".format(k,iteration,image_number,method,cut))
    if image_number==1:
        plt.imshow(im.astype('int'))
    else:
        plt.imshow(im2.astype('int'))
    plt.imshow(res.reshape(100,100),alpha = 0.8)
    plt.savefig('Spectral_{}means_image{}_{}_{}_{}iters.png'.format(k,image_number,cut,method,iteration))

def k_means(k,U,image_number,cut_method):
    print("k={}".format(k))
    iteration = np.zeros(2,dtype=np.int)
    for i in range(2):
        res_prev = np.random.randint(k,size=10000)
        means= init(U,i,k,image_number)
        while(1):
            iteration[i]+=1
            diff = 0
            res = classify(U,means,k,image_number)
            new_means = update(U,res,k,image_number)
            draw(res,image_number,iteration[i],k,i,cut_method)
            for j in range(10000):
                if res[j] != res_prev[j]:
                    diff += 1
            print('iteration: {}, diff: {}'.format(iteration[i],diff))
            if diff <10 and iteration[i]!=1:
                break
            res_prev = res
            means = new_means

im = imageio.imread('image1.png').astype('float32')
im2 = imageio.imread('image2.png').astype('float32')
W1 = rbf_kernel(im,1)
W2 = rbf_kernel(im2,2)
D1 = np.diag(np.sum(W1,axis=1))
D2 = np.diag(np.sum(W2,axis=1))
L1 = D1-W1
L2 = D2-W2

# Ratio-Cut
eigen_values_1, eigen_vectors_1 = np.linalg.eig(L1)
eigen_values_2, eigen_vectors_2 = np.linalg.eig(L2)
eigen_vectors_1_sort = eigen_vectors_1[:,np.argsort(eigen_values_1)]
eigen_vectors_2_sort = eigen_vectors_2[:,np.argsort(eigen_values_2)]
for k in range(2,5):
    U1 = (eigen_vectors_1_sort[:,1:k+1])
    U2 = (eigen_vectors_2_sort[:,1:k+1])
    k_means(k,U1,1,'ratio cut')
    k_means(k,U2,2,'ratio cut')

# Normal-Cut
D1_inv = np.linalg.inv(D1**0.5)
D2_inv = np.linalg.inv(D2**0.5)
L1_sym = np.dot(np.dot(D1_inv,L1),D1_inv)
L2_sym = np.dot(np.dot(D2_inv,L2),D2_inv)
eigen_values_normal_1, eigen_vectors_normal_1 = np.linalg.eig(L1_sym)
eigen_values_normal_2, eigen_vectors_normal_2 = np.linalg.eig(L2_sym)
eigen_vectors_normal_1_sort = eigen_vectors_normal_1[:,np.argsort(eigen_values_normal_1)]
eigen_vectors_normal_2_sort = eigen_vectors_normal_2[:,np.argsort(eigen_values_normal_2)]
for k in range(2,5):
    U1_normal = (eigen_vectors_normal_1_sort[:,1:k+1])
    U2_normal = (eigen_vectors_normal_2_sort[:,1:k+1])
    k_means(k,U1_normal,1,'normal cut')
    k_means(k,U2_normal,2,'normal cut')
