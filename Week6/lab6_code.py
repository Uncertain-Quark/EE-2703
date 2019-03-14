import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp

#Question1
#define the transfer functions 
H = sp.lti([-0.5],[-0.5+1.5j,-0.5-1.5j],1)
X = sp.lti([-0.5],[-0.5+1.5j,-0.5-1.5j,-1.5j,1.5j],1)
#calculate the time domain counterparts
t1,x1=sp.impulse(H,None,np.linspace(0,10,201))
t2,x2=sp.impulse(X,None,np.linspace(0,50,1001))

plt.figure(1)
plt.subplot(2,1,1)
plt.title('Original function cos(1.5t)e^(-0.5t)u(t)')
plt.plot(t1,x1)

plt.subplot(2,1,2)
plt.title('Solution to the differential function')
plt.plot(t2,x2)

#Question2
#define the respective transfer functions
H2 = sp.lti([-0.05],[-0.05+1.5j,-0.05-1.5j],1)
X2 = sp.lti([-0.05],[-0.05+1.5j,-0.05-1.5j,-1.5j,1.5j],1) # decay constant = 0.05
#compute the time domain counterparts of the transfer functions
t1,x1=sp.impulse(H2,None,np.linspace(0,10,201))
t2,x2=sp.impulse(X2,None,np.linspace(0,50,1001))

plt.figure(2)
plt.subplot(2,1,1)
plt.title('Original function cos(1.5t)e^(-0.05t)u(t)')
plt.plot(t1,x1)

plt.subplot(2,1,2)
plt.title('solution to the differential equation')
plt.plot(t2,x2)

plt.show()


# Question 3 
a = 1.4j
tran_func = sp.lti([],[-1.5j,1.5j],1)
for i in range(1,6) :
    t1,tran_func_cont=sp.impulse(tran_func,None,np.linspace(0,50,501))
    input_func = sp.lti([-0.05],[-a-0.05,+a-0.05],1)
    t,y,svec = sp.lsim(input_func,tran_func_cont,np.linspace(0,50,501))#define the convolution between the system transfer function and input
    plt.figure(3)
    plt.title('Varying input frequency')
    plt.plot(t,y,label='frequency at {}'.format(a))
    plt.legend()
    a = a + 0.05j
plt.show()    

# Question 4 
# Using the hand calculated values of X(s) and Y(s), we define the LTI systems and proceed
X_func = sp.lti([1,0,2],[1,0,3,0])
t_cont,X_cont = sp.impulse(X_func,None,np.linspace(0,20,1001))
plt.figure(4) 
plt.subplot(1,2,1)
plt.plot(t_cont,X_cont)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Displacement of spring in X')

Y_func = sp.lti([2],[1,0,3,0])
t_cont,Y_cont = sp.impulse(Y_func,None,np.linspace(0,20,1001)) 
plt.subplot(1,2,2)
plt.plot(t_cont,Y_cont)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Displacement of spring in Y')
plt.show()

#Question 5
# transfer function of the electrical system
H_5 = sp.lti([10**12],[1,10**8,10**12])
w,mod,phi = H_5.bode()# computing the Bode plot

plt.figure(5)
plt.subplot(1,2,1)
plt.semilogx(w,mod)
plt.title('Bode plot of magnitude')
plt.xlabel('frequency')
plt.ylabel('magnitude(H(s))')
plt.subplot(1,2,2)
plt.semilogx(w,phi)
plt.title('Bode plot of phase')
plt.xlabel('frequency')
plt.ylabel('phase(H(s))') 
plt.show()

# Question 6
T_small = np.linspace(0,30e-6,10001)
input_func_6 = lambda t : np.cos(10**3 * t) - np.cos(10**6 * t)
t_6,y_6,vsec = sp.lsim(H_5,input_func_6(T_small),T_small) # convolution of the input and the transfer function 

plt.figure(6)
plt.plot(t_6,y_6)
plt.title('response of system to input with time in order of micro seconds')
plt.xlabel('Time')
plt.ylabel('Voltage across capacitor')
plt.show()

T_large = np.linspace(0,30e-3,10001)
t_7,y_7,vsec = sp.lsim(H_5,input_func_6(T_large),T_large)

plt.figure(7)
plt.plot(t_7,y_7)
plt.title('response of system to input with time in order of milli seconds')
plt.xlabel('Time')
plt.ylabel('Voltage across capacitor')
plt.show()
