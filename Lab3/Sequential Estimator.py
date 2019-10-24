import numpy as np

def univariate_gaussian_data_generator():
    new = np.sum(np.random.uniform(0,1,12))-6
    new = mean+new*np.sqrt(varience)
    return new

def sequential_estimator():
    N=1
    new = univariate_gaussian_data_generator()
    print("Add data point: ", new)
    old_mean = new
    old_var = 0
    print("Mean = {}  Varience = {}".format(old_mean, old_var))
    while(1):
        N+=1
        new = univariate_gaussian_data_generator()
        print("Add data point: ", new)
        new_mean = (old_mean*(N-1)+new)/N
        new_var = ((N-1)*(old_var+old_mean**2)+new**2)/N-new_mean**2
        print("Mean = {}  Varience = {}".format(new_mean, new_var))
        if(abs(new_mean-old_mean)<0.001 and abs(new_var-old_var)<0.001):
            break
        else:
            old_mean = new_mean
            old_var = new_var


mean = float(input("Please assign mean: "))
varience= float(input("Please assign varience: "))
print("Data point source function: N({}, {})\n".format(mean,varience))
sequential_estimator()