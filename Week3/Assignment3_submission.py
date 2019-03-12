import numpy as np
from pylab import *
import scipy.special as sp

data=[np.loadtxt('fitting.dat',usecols=(i)) for i in range(0,10)]
def g(t,A,B):
    return A*sp.jn(2,t)+B*t 
def stdev(data,t,A0,B0):
    return math.sqrt(sum([(data[i]-g(t[i],A0,B0))**2 for i in range(0,len(data))])/len(data)) 
t = np.array(data[0])       
A0=1.05
B0=-0.105

figure('fig1')
[plot(t,data[i],label=r'$\sigma=$' + str(round(stdev(data[i],t,A0,B0),3))) for i in range(1,len(data))]    
true_value = g(t,1.05,-0.105)
plot(t,true_value,label='True Value')
grid(True);legend(loc='upper right');xlabel('t');ylabel('f(t)');savefig('fig1_fin.png')

figure('fig2')
errorbar(data[0][::5],data[1][::5],stdev(data[1],data[0],A0,B0),fmt='ro',label='errorbar')
plot(t,true_value,label='True Value')
grid(True);xlabel('t');legend(loc='upper right');savefig('fig2_fin.png')

j = [sp.jn(2,i) for i in t]
M = c_[j,t]
G = np.matmul(c_[[sp.jn(2,i) for  i in t],t],np.array([A0,B0]))
print ('equal') if max([G[i]-true_value[i] for i in range(0,len(G))])<1e-10 else print('not equal')


A_calc_list=np.array([np.linalg.lstsq(M,data[i],rcond=-1)[0][0] for i in range(1,len(data))])
B_calc_list=np.array([np.linalg.lstsq(M,data[i],rcond=-1)[0][1] for i in range(1,len(data))])    
A = linspace(0,2,21);B= linspace(-0.2,0,21)
epsilon=np.array([sum([(data[1][z]-(A[i]*sp.jn(2,t[z])+B[j]*t[z]))**2 for z in range(len(data[1]))])/len(data[1]) for i in range(len(A)) for j in range(len(B))])        
epsilon.shape=(len(A),len(B))
figure('fig3')
con= contour(A,B,epsilon)
plot([A_calc_list[0]],[B_calc_list[0]],'ro')
clabel(con,inline=1,fontsize=10);xlabel('A');ylabel('B');savefig('fig3_fin.png')

std_dev = [stdev(data[i],data[0],A0,B0) for i in range(1,len(data))]
figure('fig4')
plot(std_dev,abs(A_calc_list-A0),'ro',label='Aerr')
plot(std_dev,abs(A_calc_list-A0),'r',linestyle='--')
plot(std_dev,abs(B_calc_list-B0),'go',label='Berr')
plot(std_dev,abs(B_calc_list-B0),'g',linestyle='--')
legend();grid(True);xlabel('sigma');ylabel('error');savefig('fig4_fin.png')

figure('fig5')
stem(std_dev,abs(A_calc_list-A0),'r','ro',label='Aerr')
stem(std_dev,abs(B_calc_list-B0),'g','go',label='Berr')
legend();xscale('log');yscale('log');ylabel('error(log scale)');xlabel('sigma(log scale)');grid(True);savefig('fig5_fin.png')
show()   
