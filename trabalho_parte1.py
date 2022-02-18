#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 02:03:05 2020

@author: laura
"""


######## Bibliotecas #########

import numpy as np  # biblioteca numérica

import matplotlib.pyplot as plt  # biblioteca gráfica
from mpl_toolkits.mplot3d import Axes3D  # biblioteca para gráficos 3D
from matplotlib import cm # biblioteca para gráficos
##############################


######## Constantes globais ########

global xc,yc,sigmax,sigmay
xc,yc,sigmax,sigmay=1.0,2.0,0.75,0.5

####################################


######## Funções ########

def f(x,y): # função a ser minimizada
    A=-(((x-xc)**2.0)/(2.0*(sigmax**2.0)))-(((y-yc)**2.0)/(2.0*(sigmay**2.0)))
    f1=1-np.exp(A)
    f2=(1.0/25.0)*((x-xc)**2.0+(y-yc)**2.0)
    return f1+f2

def gradf(X): # gradiente de f
    x=X[0]
    y=X[1]
    G=np.zeros(2)
    A=-(((x-xc)**2.0)/(2.0*(sigmax**2.0)))-(((y-yc)**2.0)/(2.0*(sigmay**2.0)))
    g1=((x-xc)/(sigmax**2.0))*np.exp(A)+(2.0/25.0)*(x-xc)
    g2=((y-yc)/(sigmay**2.0))*np.exp(A)+(2.0/25.0)*(y-yc)
    G[0]=g1
    G[1]=g2
    return G

def hessianaf(X): # hessiana de f
    x=X[0,0]
    y=X[1,0]
    H=np.zeros((2,2))
    A=-(((x-xc)**2.0)/(2.0*(sigmax**2.0)))-(((y-yc)**2.0)/(2.0*(sigmay**2.0)))
    h11=np.exp(A)*(1.0/(sigmax**2.0)) - np.exp(A)*(((x-xc)**2.0)/(sigmax**4.0))+(2.0/25.0)
    h12=np.exp(A)*((x-xc)*(yc-y)/((sigmax*sigmay)**2.0))
    h21=np.exp(A)*((xc-x)*(y-yc)/((sigmax*sigmay)**2.0))
    h22=np.exp(A)*(1.0/(sigmay**2.0)) - np.exp(A)*(((y-yc)**2.0)/(sigmay**4.0))+(2.0/25.0)
    H[0,0]=h11
    H[0,1]=h12
    H[1,0]=h21
    H[1,1]=h22
    return H
    
def determinante(H): # determinante da hessiana
    det=H[0,0]*H[1,1]-H[0,1]*H[1,0]
    return det

def metododejacobi(H,b): # método iterativo de Jacobi
    k=1
    n=100
    x0=np.ones((2,1))
    y=np.zeros((2,1))
    xi=x0
    TOL=10.0**(-8.0) # tolerância do método de Jacobi
    while(k<=n):
        for i in range(2):
            soma=0.0
            for j in range(2):
                if(j!=i):
                    soma=soma+H[i,j]*x0[j,0]
            y=-x0
            xi[i,0]=(1.0/H[i,i])*(-soma+b[i])
            y=xi+y
            print("x(",i+1,")=",xi)
            norma=((y[0,0])**2.0+(y[1,0])**2.0)**(1.0/2.0)
            print("||x(",i+1,")-x(",i,")||=",norma)
        if(norma<TOL):
            print("O método de Jacobi convergiu após",k,"iterações.")
            return xi
        k=k+1
        x0[:,:]=xi[:,:]
    if(k>=n):
        print("Número máximo de iterações atingido.")
        return xi
 
def metododenewton(X0,tol,N): # método de Newton   
    k=1
    F=np.zeros((2,1))
    Gf=np.zeros((2,1))
    H=np.zeros((2,2))
    G1=np.zeros((2,N))
    Ge=np.zeros(N)
    G2=np.zeros((2,N))
    X=np.zeros((2,1))
    Xb=np.zeros((2,1))
    Xb[0,0],Xb[1,0]=1.0,2.0
    Y=np.zeros((2,1))
    Y2=np.zeros((2,1))
    E=np.zeros(N)
    Y2=X0-Xb
    E[0]=(((Y2[0,0])**2.0+(Y2[1,0])**2.0)**(1.0/2.0))
    X[0],X[1]=X0[0],X0[1]
    Y=X
    d=1.0
    while(k<N):
        print("X(",k-1,") = ",X)
        Gf=gradf(X)
        G1[0,k-1],G1[1,k-1]=X[0,0],X[1,0]
        G2[:,k-1]=Gf
        F[0,0],F[1,0]=-Gf[0],-Gf[1]
        H=hessianaf(X)
        print("Hf(X(",k-1,"))=",H)
        print("gradf(X(",k-1,"))=",Gf)
        print("f(X(",k-1,"))=",f(X[0,0],X[1,0]))
        d=determinante(H)
        print("Det(Hf(X",k-1,"))",d)
        if(d!=0.0):
            Y=metododejacobi(H,F)
        X=X+Y
        Y2=X-Xb
        E[k]=(((Y2[0,0])**2.0+(Y2[1,0])**2.0)**(1.0/2.0))
        norma=(((Y[0,0])**2.0+(Y[1,0])**2.0)**(1.0/2.0))
        Ge[k-1]=norma
        print("Y(",k-1,") = ",Y[:,0])
        print("||Y(",k-1,")|| = ",(((Y[0,0])**2.0+(Y[1,0])**2.0)**(1.0/2.0)))
        if(norma<tol):
            G1[:,k]=X[:,0]
            G2[:,k]=gradf(X)
            if((H[0,0]>0.0)and(d>0.0)):
                print("X(",k-1,")=",X,"É ponto de mínimo!!!")
                print("f(X)=",f(X[0,0],X[1,0]))
                print("Condição de mínimo: det(H)=",d,">0 e fxx=",H[0,0],">0" )
                return X, G1, G2, k, Ge, E
        k=k+1
    if(k>=N):
        print("Número máximo de iterações excedido.")
        return -1
     



##########################
    

######## Execução do experimento ########


tol=10.0**(-8.0) # tolerância

N=100 # número máximo de iterações


X0=np.zeros(2) # ponto inicial
X1=np.zeros(2)

G1=np.zeros((2,N))
G2=np.zeros((2,N))
Ge=np.zeros(N)
E=np.zeros(N)
k=0




X1,G1,G2,k,Ge,E=metododenewton(X0,tol,N)



G3=np.zeros(k+1)
g1=np.zeros((2,k+1))
g2=np.zeros((2,k+1))
ge=np.zeros(k)
e=np.zeros(k-1)
I=np.zeros(k)
Ie=np.zeros(k-1)
for i in range(k+1):
    G3[i]=f(G1[0,i],G1[1,i])
    g1[:,i]=G1[:,i]
    g2[:,i]=G2[:,i]
    
    if(i<k):
        ge[i]=np.log(Ge[i])
        I[i]=i
    if(i<k-1):
        e[i]=np.log(E[i])
        Ie[i]=i


#########################################


######## Plotando os gráficos ########


# domínio: [a,b]x[a,b], b=4, a=-2, número de divisões=60
n=60
dx=np.zeros(n+1)
dy=np.zeros(n+1)
a=-2.0
b=4.0
g3=np.zeros((n+1,n+1))

for i in range(n+1):
    dx[i]=a+i*((b-a)/n)
    dy[i]=a+i*((b-a)/n)
    
for i in range(n+1):    
    for j in range(n+1):
        g3[i,j]=f(dx[i],dy[j])
        

Mx,My=np.meshgrid(dx,dy)
fig=plt.figure()

ax0=fig.add_subplot(111,projection='3d')
ax0.scatter(g1[0,:],g1[1,:],G3,c='red')
ax0.plot(g1[0,:],g1[1,:],G3)
ax0.plot_surface(My,Mx,g3, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.2, antialiased=False)  
plt.show() # plotando f


pontos=plt.scatter(I,ge)
linha,=plt.plot(I,ge,label="ln(||X(k)-X(k-1)||)",color='purple',ls='-')
plt.xlabel("k")
plt.ylabel("ln(||X(k)-X(k-1)||)")
plt.legend(handles=[linha],loc='best')
plt.show() # plotando o log do erro

pontose=plt.scatter(Ie,e)
linha,=plt.plot(Ie,e,label="ln(||X(k)-X||)",color='green',ls='-')
plt.xlabel("k")
plt.ylabel("ln(||X(k)-X||)")
plt.legend(handles=[linha],loc='best')
plt.show() # plotando o log do erro


######################################
