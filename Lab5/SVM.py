import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("/usr/local/lib/python3.6/site-packages/libsvm/")
from svmutil import *
from subprocess import *
from scipy.spatial.distance import pdist, cdist, squareform

X_train = pd.read_csv('X_train.csv',header=None).values
Y_train = pd.read_csv('Y_train.csv',header=None).values.reshape(-1)
X_test = pd.read_csv('X_test.csv',header=None).values
Y_test = pd.read_csv('Y_test.csv',header=None).values.reshape(-1)

##############  Problem 1  ############## 

kernels = ['linear','polynomial','RBF']
for i in range(3):
    print('Kernel: '+kernels[i])
    params = '-q -t '+ str(i)
    model = svm_train(Y_train,X_train,params)
    svm_predict(Y_test,X_test, model)
print('\n------------------------------------------------------------------------------------------\n')
##############  Problem 2  ############## 

c_begin,c_end,c_step = (-5,5,1) # Cost
g_begin,g_end,g_step = (-5,5,1) # Gamma
d_begin,d_end,d_step = (2,4,1)  # degree(poly. kernel)
r_begin,r_end,r_step = (0,2,1)  # coef.(poly. kernel)
best_acc = [0,0,0]
best_params = [0,(0,0,0,0),(0,0)]
RBF_res = []
for c in range(c_begin,c_end+c_step,c_step):
    
    # Linear kernel
    params = '-q -v 5 -s 0 -t 0 -c {}'.format(2**c)
    acc = svm_train(Y_train,X_train,params)
    print('[Linear] {} {} (best: c={} acc={})'.format(c,acc,best_params[0],best_acc[0]))
    if acc > best_acc[0]:
        best_acc[0] = acc
        best_params[0] = 2**c
        
    # Polynomial kernel
    for g in range(g_begin,g_end+g_step,g_step):
        for d in range(d_begin,d_end+d_step,d_step):
            for r in range(r_begin,r_end+r_step,r_step):
                params = '-q -v 5 -s 0 -t 1 -c {} -g {} -d {} -r {}'.format(2**c,2**g,d,r)
                acc = svm_train(Y_train,X_train,params)
                print('[Polynomial] {} {} {} {} {} (best: c={} g={} d={} r={} acc={})'.format(c,g,d,r,acc,best_params[1][0],best_params[1][1],best_params[1][2],best_params[1][3],best_acc[1]))
                if acc > best_acc[1]:
                    best_acc[1] = acc
                    best_params[1] = (2**c,2**g,d,r)
        
    # RBF kernel
    for g in range(g_begin,g_end+g_step,g_step):
        params = '-q -v 5 -s 0 -t 1 -c {} -g {}'.format(2**c,2**g)
        acc = svm_train(Y_train,X_train,params)
        print('[RBF] {} {} {} (best: c={} g={} acc={})'.format(c,g,acc,best_params[2][0],best_params[2][1],best_acc[2]))
        if acc > best_acc[2]:
            best_acc[2] = acc
            best_params[2] = (2**c,2**g) 
        RBF_res.append((c,g,acc))
        
print("\nBest parameters of linear, polynomial, RBF kernel respectively: \n{}".format(best_params))
print("Best accuracy of linear, polynomial, RBF kernel respectively: \n{}\n".format(best_acc))

begin_level = round(max(x[2] for x in RBF_res)) - 3
step_size = 0.5
gnuplot = Popen('/usr/local/bin/gnuplot',stdin = PIPE,stdout=PIPE,stderr=PIPE).stdin
gnuplot.write(b"set term png transparent small linewidth 2 medium enhanced\n")
gnuplot.write("set output \"OUT.png\"\n".encode())
gnuplot.write(b"set xlabel \"log2(C)\"\n")
gnuplot.write(b"set ylabel \"log2(gamma)\"\n")
gnuplot.write("set xrange [{0}:{1}]\n".format(c_begin,c_end).encode())
gnuplot.write("set yrange [{0}:{1}]\n".format(g_begin,g_end).encode())
gnuplot.write(b"set contour\n")
gnuplot.write("set cntrparam levels incremental {0},{1},100\n".format(begin_level,step_size).encode())
gnuplot.write(b"unset surface\n")
gnuplot.write(b"unset ztics\n")
gnuplot.write(b"set view 0,0\n")
gnuplot.write("set title \"HW5 - handwritten digit\"\n".encode())
gnuplot.write(b"unset label\n")
gnuplot.write("set label \"Best log2(C) = {0}  log2(gamma) = {1}  accuracy = {2}%\" \
  at screen 0.5,0.85 center\n". \
  format(np.log2(best_params[2][0]), np.log2(best_params[2][1]), best_acc[2]).encode())
gnuplot.write("set label \"C = {0}  gamma = {1}\""
  " at screen 0.5,0.8 center\n".format(best_params[2][0], best_params[2][1]).encode())
gnuplot.write(b"set key at screen 0.9,0.95\n")
gnuplot.write(b"splot \"-\" with lines\n")

RBF_res.sort(key = lambda x:(x[0], -x[1]))

prevc = RBF_res[0][0]
for line in RBF_res:
    if prevc != line[0]:
        gnuplot.write(b"\n")
        prevc = line[0]
    gnuplot.write("{0[0]} {0[1]} {0[2]}\n".format(line).encode())
gnuplot.write(b"e\n")
gnuplot.write(b"\n") # force gnuplot back to prompt when term set failure
gnuplot.flush()

# Best Linear
params = '-q -t 0 -c {}'.format(best_params[0])
model = svm_train(Y_train,X_train,params)
svm_predict(Y_test,X_test,model)

# Best Poly
params = '-q -t 1 -c {} -g {} -d {} -r {}'.format(best_params[1][0],best_params[1][1],best_params[1][2],best_params[1][3])
model = svm_train(Y_train,X_train,params)
svm_predict(Y_test,X_test,model)

# Best RBF
params = '-q -t 2 -c {} -g {}'.format(best_params[2][0],best_params[2][1])
model = svm_train(Y_train,X_train,params)
svm_predict(Y_test,X_test,model)
print('\n------------------------------------------------------------------------------------------\n')

##############  Problem 3  ############## 
gamma = -0.03125

linear_train = np.dot(X_train,X_train.T)
RBF_train = squareform(np.exp(gamma*pdist(X_train,'sqeuclidean')))
x_train_kernel = np.hstack((np.arange(1, 5001).reshape((5000, 1)),np.add(linear_train,RBF_train)))

linear_test = np.dot(X_test,X_train.T)
RBF_test = np.exp(gamma*cdist(X_test,X_train,'sqeuclidean'))
x_test_kernel = np.hstack((np.arange(1, 2501).reshape((2500, 1)),np.add(linear_test,RBF_test)))

model = svm_train(Y_train,x_train_kernel,'-q -t 4')
svm_predict(Y_test,x_test_kernel,model)