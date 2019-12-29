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

def init(method,k,image_number):
    if method == 0:
        print('Init method: random')
        return np.random.randint(k,size=10000)
    else:
        print('Init method: kmeans++')
        res = np.zeros(10000)
        means = np.zeros(k)
        means[0] = np.random.randint(10000)
        for i in range(1,k):
            dist = np.zeros(10000)
            for j in range(10000):
                temp = np.zeros(i)
                x = j//100
                y = j%100
                for p in range(i):
                    xm = int(means[p]//100)
                    ym = int(means[p]%100)
                    if image_number==1:
                        temp[p] = (xm-x)**2+(ym-y)**2+np.linalg.norm(im[xm,ym]-im[x,y])**2
                    else:
                        temp[p] = (xm-x)**2+(ym-y)**2+np.linalg.norm(im2[xm,ym]-im2[x,y])**2
                dist[j] = np.min(temp)
            dist = dist/dist.sum()
            means[i] = np.random.choice(10000,1,p=dist)
        for i in range(10000):
            temp = np.zeros(k)
            x=i//100
            y=i%100
            for j in range(k):
                xm = int(means[j]//100)
                ym = int(means[j]%100)
                if image_number==1:
                    temp[j] = (xm-x)**2+(ym-y)**2+np.linalg.norm(im[xm,ym]-im[x,y])**2
                else:
                    temp[j] = (xm-x)**2+(ym-y)**2+np.linalg.norm(im2[xm,ym]-im2[x,y])**2
            res[i] = np.argmin(temp)
        return res

def second_term(kernel_data,res_prev,node,cluster):
    temp = 0
    for i in range(10000):
        if res_prev[i]==cluster:
            temp += kernel_data[node][i]
    return temp

def third_term(kernel_data,res_prev,k):
    C = np.zeros(k)
    for c in range(k):
        for i in range(10000):
            for j in range(i+1,10000):
                if res_prev[i]==c and res_prev[j]==c :
                    C[c] += kernel_data[i][j]
    return C

def classify(kernel_data,res_prev,k):
    res = np.zeros(10000)
    unique, counts = np.unique(res_prev, return_counts=True)
    third = third_term(kernel_data,res_prev,k)
    for j in range(10000):
        temp = np.zeros(k)
        for c in range(k):
            temp[c]+=(0-2*second_term(kernel_data,res_prev,j,c)/counts[c]+third[c]/counts[c]**2)
        res[j] = np.argmin(temp)
    return res

def draw(res,image_number,iteration,k,method):
    if method == 0:
        method = 'random'
    else:
        method = 'kmeans++'
    plt.title("Kernel {} means after {} iters, Image {}\ninit method: {}".format(k,iteration,image_number,method))
    if image_number==1:
        plt.imshow(im.astype('int'))
    else:
        plt.imshow(im2.astype('int'))
    plt.imshow(res.reshape(100,100),alpha = 0.8)
    plt.savefig('Kernel{}means_{}iters_image{}_{}.png'.format(k,iteration,image_number,method))

def kernel_kmean(kernel_data,image_number,k=2):
    print("k={}".format(k))
    iteration = np.zeros(2,dtype=np.int)
    for i in range(0,2):
        res_prev = init(i,k,image_number)
        draw(res_prev,image_number,0,k,i)
        while(1):
            iteration[i]+=1
            diff = 0
            res = classify(kernel_data,res_prev,k)
            draw(res,image_number,iteration[i],k,i)
            for j in range(10000):
                if res[j] != res_prev[j]:
                    diff += 1
            print('iteration: {}, diff: {}'.format(iteration[i],diff))
            if diff < 50:
                break
            res_prev = res

im = imageio.imread('image1.png').astype('float32')
im2 = imageio.imread('image2.png').astype('float32')
W1 = rbf_kernel(im,1)
W2 = rbf_kernel(im2,2)
for k in range(2,5):
    kernel_kmean(W1,1,k=k)
    kernel_kmean(W2,2,k=k)