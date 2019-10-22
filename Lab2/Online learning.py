from math import factorial

def online_learning(a,b,cases):
    for i in range(len(cases)):
        print('case {}: {}'.format(i+1,cases[i]))
        num_1=num_0=0
        for outcome in cases[i]:
            if outcome=='1':
                num_1+=1
            else:
                num_0+=1
        N=len(cases[i])
        likelihood=(factorial(N)/(factorial(num_1)*factorial(N-num_1))*((num_1/N)**num_1)*(num_0/N)**num_0)
        print('Likelihood: {}'.format(likelihood))
        print('Beta prior: a = {} b = {}'.format(a,b))
        a+=num_1
        b+=num_0
        print('Beta posterior: a = {} b = {}\n'.format(a,b))

def read_test_file(file_path):
    cases=[]
    with open(file_path) as f:
        for x in f:
            cases.append(x.strip())
    return cases

file_path=input('File path:')
a=int(input('a for initial beta prior: '))
b=int(input('b for initial beta prior: '))
print('')
cases=read_test_file(file_path)
online_learning(a,b,cases)