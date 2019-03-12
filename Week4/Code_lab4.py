#importing various necessary modules
import numpy as np
from scipy.integrate import *
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

#the below two lines define the required functions which return the respective values of the functions when given an input
fune = lambda t : np.e**t
func = lambda t : np.cos(np.cos(t))

#Question 1

plt.figure(1);plt.subplot(211)
plt.title('Function: cos(cos(x))')
plt.plot(np.linspace(-2*np.pi,4*np.pi,100),func(np.linspace(-2*np.pi,4*np.pi,100)),label='original function')#plot the actual function over -2pi to 4pi
plt.plot(np.linspace(-2*np.pi,0,100),func(np.linspace(-2*np.pi,0,100)),'g',label='expected fourier series')#plot the function which we expect i.e a periodic function with perioid 2pi over -2pi and 4pi
plt.plot(np.linspace(0,2*np.pi,100),func(np.linspace(0,2*np.pi,100)),'g')
plt.plot(np.linspace(2*np.pi,4*np.pi,100),func(np.linspace(0,2*np.pi,100)),'g')
plt.grid();plt.legend()

plt.subplot(212)
plt.semilogy(np.linspace(-2*np.pi,4*np.pi,100),fune(np.linspace(-2*np.pi,4*np.pi,100)),label = 'original function')#plot the actual function over -2pi to 4pi
plt.semilogy(np.linspace(-2*np.pi,0,100),fune(np.linspace(0,2*np.pi,100)),'g',label='expected fourier series')#plot the function which we expect i.e a periodic function with perioid 2pi over -2pi and 4pi
plt.semilogy(np.linspace(0,2*np.pi,100),fune(np.linspace(0,2*np.pi,100)),'g')
plt.semilogy(np.linspace(2*np.pi,4*np.pi,100),fune(np.linspace(0,2*np.pi,100)),'g')
plt.yscale('log');plt.xlabel('Function : exp(x)');plt.legend();plt.grid()

#Question 2
x = 'fune'
u = lambda t,k : eval(x)(t)*np.cos(k*t)# define the function u(x) = f(x)cos(nx) 
v = lambda t,k : eval(x)(t)*np.sin(k*t)# define the function v(x) = f(x)sin(nx)

#finding the coefficients for exp(x) using integration
ae_fs = [quad(u,0,2*np.pi,k)[0]/(np.pi) if k!=0 else quad(u,0,2*np.pi,k)[0]/(2*np.pi) for k in range(0,26)]
be_fs = [quad(v,0,2*np.pi,k)[0]/(np.pi) for k in range(1,26)]
ce_fs = [ae_fs[0]]
for i in range(0,len(be_fs)): 
    ce_fs.append(ae_fs[i+1])
    ce_fs.append(be_fs[i])  
    
x = 'func'
#finding the coefficients of cos(cos(x)) using integration
ac_fs = [quad(u,0,2*np.pi,k)[0]/(np.pi) if k!=0 else quad(u,0,2*np.pi,k)[0]/(2*np.pi) for k in range(0,26)]
bc_fs = [quad(v,0,2*np.pi,k)[0]/(np.pi) for k in range(1,26)]
cc_fs = [ac_fs[0]]
for i in range(0,len(bc_fs)): 
    cc_fs.append(ac_fs[i+1])
    cc_fs.append(bc_fs[i])    

#Question 3
plt.figure(2)
plt.title('Function: exp(x) coeff vs n semilogy')
plt.semilogy(abs(np.array(ce_fs)),'ro',label = 'coeffs from integration')
plt.grid()

plt.figure(3)
plt.title('Function: exp(x) coeff vs n loglog')
plt.loglog(abs(np.array([1]+ce_fs)),'ro',label ='coeff from integration')
plt.grid()

plt.figure(4)
plt.title('Function: cos(cos(x)) coeff vs n semilogy')
plt.semilogy(abs(np.array(cc_fs)),'ro',label = 'coeff from integration')
plt.grid()

plt.figure(5)
plt.title('Function: cos(cos(x)) coeff vs n loglog')
plt.loglog(abs(np.array([1]+cc_fs)),'ro',label = 'coeff from integration')
plt.grid()

#Question 5
# calculating the coefficients for both the functions using the lstsq method
x=np.linspace(0,2*np.pi,401)
x=x[:-1] 
b=fune(x)
A=np.zeros((400,51))
A[:,0]=1
for k in range(1,26):
    A[:,2*k-1]=np.cos(k*x) 
    A[:,2*k]=np.sin(k*x)

c1=lstsq(A,b)[0]

b=func(x)
A=np.zeros((400,51))
A[:,0]=1
for k in range(1,26):
    A[:,2*k-1]=np.cos(k*x)
    A[:,2*k]=np.sin(k*x)

c2=lstsq(A,b)[0]

#finding the maximum deviation in coefficients found using both the methods 
print('the maximum deviation in exp(x) is  : {} and is at {}'.format(max(np.array(abs((np.array(ce_fs))-(c1)))),list(abs((np.array(ce_fs))-(c1))).index(max(np.array(abs((np.array(ce_fs))-(c1)))))))
print('the maximum deviation in cos(cos(x)) is  : {} and is at {}'.format(max(np.array(abs((np.array(cc_fs))-(c2)))),list(abs((np.array(cc_fs))-(c2))).index(max(np.array(abs((np.array(cc_fs))-(c2)))))))

plt.figure(2)
plt.semilogy(abs(np.array(c1)),'go',label='coeff from lstsq')
plt.legend()

plt.figure(3)
plt.loglog(abs(np.array([1]+list(c1))),'go',label = 'coeff from lstsq' )
plt.legend()

plt.figure(4)
plt.semilogy(abs(np.array(c2)),'go',label = 'coeff from lstsq' )
plt.legend()

plt.figure(5)
plt.loglog(abs(np.array([1]+list(c2))),'go',label = 'coeff from lstsq')
plt.legend()

#the below plot represents the expected plot against the actual plot of the functions
plt.figure(6)
plt.subplot(212)
plt.title('Function : exp(x)')
plt.semilogy(x,fune(x),'r',label='expected')
plt.semilogy(x,A.dot(c1),'g',label='obtained from lstsq')
plt.legend()

plt.subplot(211)
plt.title('Function : cos(cos(x)')
plt.plot(x,func(x),'r',label='expected')
plt.plot(x,A.dot(c2),'g',label='obtained from lstsq')
plt.legend()

plt.show()
