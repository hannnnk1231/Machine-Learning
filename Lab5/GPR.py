import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def kernel(Xn, Xm, l=1.0, sigma=1.0, alpha=-1.0):
    dist = Xn**2-2*np.dot(Xn,Xm.T)+(Xm**2).flatten()
    return sigma**2*(1+dist/(2*alpha*(l**2)))**(-alpha)

def GPR(X,X_train,Y_train,l=1.0,sigma=1.0,alpha=1.0,noise=1/5):
    C = kernel(X_train,X_train,l,sigma,alpha) + (noise**2)*np.eye((len(X_train)))
    C_inv = np.linalg.inv(C)
    Ks = kernel(X_train,X,l,sigma,alpha)
    Kss = kernel(X,X,l,sigma,alpha) + (noise**2)
    MU = np.dot(np.dot(Ks.T,C_inv),Y_train)
    COV = Kss - np.dot(np.dot(Ks.T,C_inv),Ks)
    return MU,COV

def log_likelihood(theta):
    C = kernel(x,x,l=theta[0],sigma=theta[1],alpha=theta[2]) + (noise**2)*np.eye((len(x)))
    return 0.5*(np.log(np.linalg.det(C)) + np.dot(np.dot(y.T,np.linalg.inv(C)),y) + len(x)*np.log(2*np.pi))

def plot_GPR(mu,cov):
    plt.plot(x,y,'bx')
    plt.plot(x_pred,mu,'r') 
    cert = 1.96 * np.sqrt(np.diag(cov))
    plt.fill_between(x_pred.ravel(), mu.ravel() + cert, mu.ravel() - cert, alpha=0.2)

x = []
y = []
noise = 1/5
with open('input.data.txt') as f:
    d = f.readlines()
    for l in d:
        _l = l.split()
        x.append(float(_l[0]))
        y.append(float(_l[1]))
x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)
x_pred = np.arange(-60,60,0.5).reshape(-1,1)

########  Problem 1  #######

mu,cov = GPR(x_pred,x,y,1,1,1,1/5)
plot_GPR(mu,cov)
plt.show()

########  Problem 2  #######

res = minimize(log_likelihood,[1,1,1])
print('Opt theta: ',res.x)
opt_l,opt_sigma,opt_alpha = res.x
mu_opt,cov_opt = GPR(x_pred,x,y,opt_l,opt_sigma,opt_alpha,noise)
plot_GPR(mu_opt,cov_opt)
plt.show()

########  Other observations  #######

parameters=[
    (0.5,1,1,1/5),
    (5,1,1,1/5),
    (1,0.5,1,1/5),
    (1,5,1,1/5),
    (1,1,0.1,1/5),
    (1,1,5,1/5),
    (1,1,1,1/5),
    (1,1,1,5)
]
plt.figure(figsize=(20,15))
for i,(l,sigma,alpha,noise) in enumerate(parameters):
    mu_test, cov_test = GPR(x_pred,x,y,l,sigma,alpha,noise)
    plt.subplot(4,2,i+1)
    plt.title('l={} sigma={} alpha={} noise={}'.format(l,sigma,alpha,noise))
    plot_GPR(mu_test,cov_test)
plt.show()


