import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import sympy as sy

# lowpass function
def lowpass(R1,R2,C1,C2,G,Vi):
    s=sy.symbols('s')
    A=sy.Matrix([[0,0,1,-1/G],[-1/(1+s*R2*C2),1,0,0], [0,-G,G,1],[-1/R1-1/R2-s*C1,1/R2,0,s*C1]])
    b=sy.Matrix([0,0,0,-Vi/R1])
    V = A.inv()*b
    return (A,b,V)

#highpass Function
def highpass(R1,R2,C1,C2,G,Vi):
    s=sy.symbols('s')
    A=sy.Matrix([[0,0,1,-1/G],[-(s*R2*C2)/(1+s*R2*C2),1,0,0], [0,-G,G,1],[-1/R1-s*C2-s*C1,s*C2,0,1/R1]])
    b=sy.Matrix([0,0,0,-Vi*s*C1])
    V = A.inv()*b
    return (A,b,V)
    
#converting Vo from sympy.core to sp.lti        
def convert(Vs) :
    Vs = sy.simplify(Vs)    
    num,den = sy.fraction(Vs)
    num = np.array(sy.Poly(num,s).all_coeffs(),dtype = np.complex64)
    den = np.array(sy.Poly(den,s).all_coeffs(),dtype = np.complex64)
    Vo_sig = sp.lti(num,den)
    print(Vo_sig)
    return Vo_sig     

A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,1)
print( 'G=1000')
Vo=V[3]
print (Vo)
w=np.logspace(0,8,801)
ss=1j*w
s = sy.symbols('s')
hf=sy.lambdify(s,Vo,'numpy')
v=hf(ss)
plt.loglog(w,abs(v),lw=2)
plt.grid(True)
plt.show() 

#this part of the code evaluates the bode plot of Vo
Vo_signal = convert(Vo)
w , mod , phi = Vo_signal.bode(np.linspace(1,10**8,10**5+1))
plt.subplot(2,1,1)
plt.semilogx(w,mod)
plt.grid()
plt.title('Magnitude Plot - Low Pass')
plt.subplot(2,1,2)
plt.semilogx(w,phi)
plt.title('Phase Plot - Low Pass')
plt.grid()
plt.show()

#calculates the impulse response of lowpass
t,Vo_t = sp.impulse(Vo_signal,None,np.linspace(0,100e-6,20001))
plt.plot(t,Vo_t)
plt.title('Impulse response low pass')
plt.show()

#calculates the step response of the lowpass
t,unit_res = sp.step(Vo_signal,None,np.linspace(0,100e-6,200001),None)
plt.plot(t,unit_res)
plt.title('unit step response low pass')
plt.show()

#calculates the response to sinusoidal input to lowpass
t = np.linspace(0,6/2000,2001)
t,sin_res,svec = sp.lsim(Vo_signal,np.sin(2000*np.pi*t)+np.cos(2*1e6*np.pi*t),t)
plt.plot(t,sin_res)
plt.title('response to sinusoids')
plt.show()

#calulates the response to damped sinusoids to lowpass
t = np.linspace(0,1/20,4001)
t,sin_res,svec = sp.lsim(Vo_signal,np.exp(-1e2*t)*np.cos(2*1e3*np.pi*t),t)
plt.plot(t,sin_res)
plt.title('Response to Damped Low Pass')
plt.show()    

#Highpass Filter
A,b,V=highpass(10000,10000,1e-9,1e-9,1.586,1)
print( 'G=1000')
Vo=V[3]
print (Vo)
w=np.logspace(0,8,801)
ss=1j*w
s = sy.symbols('s')
hf=sy.lambdify(s,Vo,'numpy')
v=hf(ss)
plt.loglog(w,abs(v),lw=2)
plt.grid(True)
plt.show() 
Vo_signal = convert(Vo)
w , mod , phi = Vo_signal.bode(np.linspace(1,10**8,10**5+1))

#computes bode plot of high pass
plt.subplot(2,1,1)
plt.semilogx(w,mod)
plt.title('Magnitude plot - High Pass')
plt.grid()
plt.subplot(2,1,2)
plt.title('Phase Plot - High Pass')
plt.semilogx(w,phi)
plt.grid()
plt.show()

# computes the impulse response of the highpass
t,Vo_t = sp.impulse(Vo_signal,None,np.linspace(0,100e-6,20001))
plt.plot(t,Vo_t)
plt.title('Impulse Response high pass')
plt.show()

# computes the step response of the highpass
t,unit_res = sp.step(Vo_signal,None,np.linspace(0,100e-6,200001),None)
plt.plot(t,unit_res)
plt.title('unit step response High Pass')
plt.show()

# computes the sinusoidal response of the highpass
t = np.linspace(0,6/(1e6),2001)
t,sin_res,svec = sp.lsim(Vo_signal,np.sin(2000*np.pi*t)+np.cos(2*1e6*np.pi*t),t)
plt.plot(t,sin_res)
plt.title('Response to Sinusoids High Pass')
plt.show()

# computes the damped sinusoidal response of the highpass
t = np.linspace(0,1e-5,4001)
t,sin_res,svec = sp.lsim(Vo_signal,np.exp(-1e5*t)*np.cos(2*1e6*np.pi*t),t)
plt.plot(t,sin_res)
plt.title('Response to Damped High Pass')
plt.show()   

# The below code computes the unit step response by passing the Vi as 1/s 
A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,1/s)
Vo=V[3]
Vo_signal = convert(Vo)
t,Vo_time = sp.impulse(Vo_signal,None,np.linspace(0,100e-6,20001))
plt.plot(t,Vo_time)
plt.grid()
plt.title('step response using lowpass(1/s)')
plt.plot()
plt.show()

A,b,V=highpass(10000,10000,1e-9,1e-9,1.586,1/s)
Vo=V[3]
Vo_signal = convert(Vo)
t,Vo_time = sp.impulse(Vo_signal,None,np.linspace(0,100e-6,20001))
plt.plot(t,Vo_time)
plt.grid()
plt.title('step response using Highpass(1/s)')
plt.plot()
plt.show()