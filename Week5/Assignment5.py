#import the necessary modules
import numpy as np
import sys
from pylab import *
import mpl_toolkits.mplot3d.axes3d as p3

# initialise the parameters
Nx = 25;Ny=25;radius=8;Niter=1500

potential = np.zeros([Ny,Nx])
x = np.array(np.linspace(-0.5,0.5,Nx))
y = np.array(np.linspace(0.5,-0.5,Nx))

X,Y = meshgrid(x,y)
# c is the tuple composed of the coordinates of the points with potential 1 
c = np .where(Y*Y + X*X <= 0.35*0.35)

potential[c] = 1.0
error=[]
# iteration which calculates the steady state potential array
for  i in range(Niter) :
    oldpot = potential.copy()
    potential[1:-1,1:-1] = 0.25*(potential[1:-1,0:-2]+potential[1:-1,2:]+potential[0:-2,1:-1]+potential[2:,1:-1])
    potential[1:-1,0]=potential[1:-1,1]
    potential[1:-1,Nx-1]=potential[1:-1,Nx-2]
    potential[0,1:-1]=potential[1,1:-1]
    potential[0,0]=potential[0,1]
    potential[0,Nx-1]=potential[0,Nx-2]
    potential[Ny-1,1:-1]=0.0
    potential[c]=1.0
    error.append((abs(oldpot-potential)).max())

# plots the various error graphs
figure(1)
semilogy(list(range(Niter)),error,'r',label = 'original error calculated')    
title('error plot with error list') 
# initialise funstions for the fitting
flog = lambda t : log(t)
fexp = lambda t : e**t

error_log = flog(error)
a_log = np.zeros((Niter,2)) # a_log*x_fit = error_log. Hence forming the matrix A
a_log[:,0]=1
for i in range(1,Niter+1): 
    a_log[i-1,1] = i
x_fit = lstsq(a_log,error_log,rcond = None)[0] # calculating the x_fit
error_obtained = a_log.dot(x_fit) # calculating the error obtained by fitting curve

semilogy(fexp(error_obtained), 'b',label = 'fit considering entire range of data') # plotting the error plot for fitted data

error_log_500 = flog(error[500:]) # this block of code computes the error and fits the data for 500 point onwards
a_log_500 = np.zeros((Niter-500,2))
a_log_500[:,0]=1
for i in range(501,Niter+1):
    a_log_500[i-501,1] = i
x_fit_500 = lstsq(a_log_500,error_log_500,rcond = None)[0]
error_obtained_500 = a_log_500.dot(x_fit_500)

semilogy(list(range(501,Niter+1)),fexp(error_obtained_500), 'g',label = 'fit considering 500 data point onwards')
legend()

fig2 = figure(2) # plot the 3d plot for the obtained potential as a function of X and Y coordinates
ax = p3.Axes3D(fig2)
title('POTENTIAL GRAPH')
surf = ax.plot_surface(X,Y,potential,rstride=1,cstride=1,cmap=cm.jet)
ax.set_xlabel('X coordinates')
ax.set_ylabel('Y Coordinates')

c= list(c) # b1 and b2 are composed of the scaled version of c to [-0.5,0.5]
b1 = c[0].copy()
b1.dtype = np.float32
b2 = c[1].copy()
b2.dtype = np.float32
for i in range(len(c[0])) :
    b1[i]=x[(c[0])[i]]
for j in range(len(c[1])) :
    b2[j]=y[(c[1])[j]]

figure(3)# plots the contour of the potential
title('contour plot for potential')
cs = contour(X,Y,potential)
clabel(cs)
scatter(X,Y,s =3)
scatter(b1,b2,s=3,color='r')

currentx = np.zeros([Ny,Nx])# initialise the current density vectors
currenty = np.zeros([Ny,Nx])
currentx[:,1:Nx-2] =0.5*( potential[:,0:Nx-3]-potential[:,2:Nx-1] )
currenty[1:Ny-2,:] =-0.5*( potential[0:Ny-3,:]-potential[2:Ny-1,:] )

figure(4)# plots the current density calculated using the quiver function as the current density is a vector quantity
title('Current density plots')
quiver(x,y,currentx,currenty,scale= 4)
scatter(b1,b2,s=6,color='r')
show()
