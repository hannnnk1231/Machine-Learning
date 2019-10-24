import numpy as np
import matplotlib.pyplot as plt

def univariate_gaussian_data_generator():
    temp = np.sum(np.random.uniform(0,1,12))-6
    temp = 0 + temp*np.sqrt(error_gen)
    return temp

def polynomial_basis_linear_model_generator():
    x = float(np.random.uniform(-1,1,1))
    y = 0
    for i in range(len(W)):
        y += W[i]*(x**i)
    return x, y + univariate_gaussian_data_generator()

b = float(input("Precision for initial prior: "))
n = int(input("Basis number: "))
error_gen = float(input("Varience for error(generator): "))
W = np.array([float(s) for s in input("{}X1 W vector: ".format(n)).split(",")])

N = 0
x = []
y = []
data_mean = 0
prev_data_mean = 0
a = 0
prev_m = np.zeros((n,1))
prev_S = np.zeros((n,n))
while(1):
    N += 1
    new_x, new_y = polynomial_basis_linear_model_generator()
    x.append(new_x)
    y.append(new_y)
    print("Add data point ({:.5f}, {:.5f})\n".format(new_x, new_y))
    
    data_mean = ((N-1)*data_mean+new_y)/N
    a = (((N-1)*(a+prev_data_mean**2)+new_y**2)/N - data_mean**2)
    if a == 0:
        a = 0.001
    else:
        a = 1/a
    
    A = np.array([[new_x**j for j in range(n)]])
    if N == 1:
        S_inv = np.linalg.inv(a*np.transpose(A)*A+b*np.identity(n))
        m = a*np.dot(S_inv,np.transpose(A))*new_y
    else:
        S = np.linalg.inv(S_inv)
        S_inv = np.linalg.inv(a*np.dot(np.transpose(A),A)+S)
        m = np.dot(S_inv, (a*np.transpose(A)*new_y+np.dot(S,m)))
    
    print("Posterior mean:")
    for i in range(m.shape[0]):
        print("{:16.10f}".format(m[i][0]))
    print("")
    
    print("Posterior varience:")
    for i in range(S_inv.shape[0]):
        for j in range(S_inv.shape[1]):
            if j == S_inv.shape[1]-1:
                print("{:16.10f}".format(S_inv[i][j]))
            else:
                print("{:16.10f},".format(S_inv[i][j]), end="")
    print("")
    
    predictive_distribution_mean = np.dot(A,m)[0][0]
    predictive_distribution_varience = 1/a + np.dot(A,np.dot(S_inv,np.transpose(A)))[0][0]
    print("Predictive distribution ~ N({:.5f}, {:.5f})".format(predictive_distribution_mean,predictive_distribution_varience))
    print("--------------------------------------------------")
    
    if (N>=100 and np.linalg.norm(m-prev_m)<0.0001 and np.linalg.norm(S_inv-prev_S)<0.0001):
        break
    
    prev_data_mean = data_mean
    prev_m = m
    prev_S = S_inv
    
    if (N==10):
        m_10 = m
        S_10 = S_inv
        a_10 = a
    if (N==50):
        m_50 = m
        S_50 = S_inv
        a_50 = a

# Vis

plt.rcParams["figure.figsize"] = (15,10)

plt.subplot(221)
plt.title("Ground truth")
plt.xlim(-2,2)
plt.ylim(-15,20)
data_x = np.linspace(-2,2,30)
plt.plot(data_x,np.poly1d(np.flip(W))(data_x),'k-')
plt.plot(data_x,np.poly1d(np.flip(W))(data_x)+error_gen,'r-')
plt.plot(data_x,np.poly1d(np.flip(W))(data_x)-error_gen,'r-')

plt.subplot(222)
plt.title("Predict result")
plt.xlim(-2,2)
plt.ylim(-15,20)
plt.plot(x,y,'bo')
data_x = np.linspace(-2,2,50)
predict_y = np.poly1d(np.flip(np.reshape(m,n)))(data_x)
plt.plot(data_x,predict_y,'k-')
for i in range(50):
    A = np.array([[data_x[i]**j for j in range(n)]])
    predict_y[i] += (1/a+np.dot(A,np.dot(S_inv,np.transpose(A)))[0][0])
plt.plot(data_x,predict_y,'r-')
for i in range(50):
    A = np.array([[data_x[i]**j for j in range(n)]])
    predict_y[i] -= 2*(1/a+np.dot(A,np.dot(S_inv,np.transpose(A)))[0][0])
plt.plot(data_x,predict_y,'r-')

plt.subplot(223)
plt.title("After 10 incomes")
plt.xlim(-2,2)
plt.ylim(-15,20)
plt.plot(x[:10],y[:10],'bo')
data_x = np.linspace(-2,2,50)
predict_y = np.poly1d(np.flip(np.reshape(m_10,n)))(data_x)
plt.plot(data_x,predict_y,'k-')
for i in range(50):
    A = np.array([[data_x[i]**j for j in range(n)]])
    predict_y[i] += (1/a_10+np.dot(A,np.dot(S_10,np.transpose(A)))[0][0])
plt.plot(data_x,predict_y,'r-')
for i in range(50):
    A = np.array([[data_x[i]**j for j in range(n)]])
    predict_y[i] -= 2*(1/a_10+np.dot(A,np.dot(S_10,np.transpose(A)))[0][0])
plt.plot(data_x,predict_y,'r-')

plt.subplot(224)
plt.title("After 50 incomes")
plt.xlim(-2,2)
plt.ylim(-15,20)
plt.plot(x[:50],y[:50],'bo')
data_x = np.linspace(-2,2,50)
predict_y = np.poly1d(np.flip(np.reshape(m_50,n)))(data_x)
plt.plot(data_x,predict_y,'k-')
for i in range(50):
    A = np.array([[data_x[i]**j for j in range(n)]])
    predict_y[i] += (1/a_50+np.dot(A,np.dot(S_50,np.transpose(A)))[0][0])
plt.plot(data_x,predict_y,'r-')
for i in range(50):
    A = np.array([[data_x[i]**j for j in range(n)]])
    predict_y[i] -= 2*(1/a_50+np.dot(A,np.dot(S_50,np.transpose(A)))[0][0])
plt.plot(data_x,predict_y,'r-')
plt.show()