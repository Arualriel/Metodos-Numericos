#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 02:03:07 2020

@author: laura
"""


######## Bibliotecas #########

import numpy as np  # biblioteca numérica
import time # biblioteca para marcar o tempo
import matplotlib.pyplot as plt  # biblioteca gráfica

##############################


######## Constantes globais ########

global P,L,k
P,L,k=15.0,1.0,100.0

####################################


######## Funções ########

def f(T): # função a qual queremos achar o zero
    t1=T[0]
    t2=T[1]
    t3=T[2]
    t4=T[3]
    f1=k*t1-k*(t2-t1)-((7.0*P*L)/2.0)*np.cos(t1)
    f2=k*(t2-t1)-k*(t3-t2)-((5.0*P*L)/2.0)*np.cos(t2)
    f3=k*(t3-t2)-k*(t4-t3)-((3.0*P*L)/2.0)*np.cos(t3)
    f4=k*(t4-t3)-((P*L)/2.0)*np.cos(t4)
    F=np.zeros(4)
    F[0],F[1],F[2],F[3]=f1,f2,f3,f4
    return F



def jacobianaf(T): # jacobiana de f
    J=np.zeros((4,4))
    for i in range(4):
        if(i<3):
            J[i,i+1]=-k
            J[i+1,i]=-k
            J[i,i]=2.0*k+(((7.0-i*2.0)*P*L)/2.0)*np.sin(T[i])
        elif(i==3):
            J[i,i]=k+(((7.0-i*2.0)*P*L)/2.0)*np.sin(T[i])
    return J



def metododejacobi(J,b): # método iterativo de Jacobi
    l=1
    n=100
    t0=np.zeros(4)
    t=np.zeros(4)
    ti=t0
    TOL=10.0**(-8.0) # tolerância do método de Jacobi
    while(l<=n):
        for i in range(4):
            soma=0.0
            for j in range(4):
                if(j!=i):
                    soma=soma+J[i,j]*t0[j]
            t=-t0
            ti[i]=(1.0/J[i,i])*(-soma+b[i])
            t=t+ti
            print("Teta(",i+1,")=",ti)
            norma=((t[0])**2.0+(t[1])**2.0+(t[2])**2.0+(t[3])**2.0)**(1.0/2.0)
            print("||Teta(",i+1,")-Teta(",i,")||=",norma)
        if(norma<TOL):
            print("JACOBI")
            print("Convergiu depois de ",l,"iterações.")
            return ti
        l=l+1
        t0=ti
    if(l>=n):
        print("Número máximo de iterações atingido.")
        return ti
 

def metododenewton(T0,tol,N): # método de Newton 
    l=1
    F=np.zeros(4)
    J=np.zeros((4,4))
    Ge=np.zeros(N)
    T=np.zeros(4)
    Y=np.zeros(4)
    T[0],T[1],T[2],T[3]=T0[0],T0[1],T0[2],T0[3]
    Y=T
    while(l<N):
        print("Teta(",l-1,") = ",T)
        F=f(T)
        F[0],F[1],F[2],F[3]=-F[0],-F[1],-F[2],-F[3]
        J=jacobianaf(T)
        print("determinante",np.linalg.det(J))
        print("Jf(T(",l-1,"))=",J)
        print("f(T(",l-1,"))=",F)
        Y=metododejacobi(J,F)
        T=T+Y
        print("Y(",l-1,") = ",Y)
        norma=(((Y[0])**2.0+(Y[1])**2.0+(Y[2])**2.0+(Y[3])**2.0)**(1.0/2.0))
        Ge[l-1]=norma
        print("||Y(",l-1,")|| = ",norma)
        if(norma<tol):
            print("NEWTON")
            print("T(",l-1,")=",T,"É a aproximação de zero!!!")
            print("f(T)=",f(T))
            return T, l, Ge
        l=l+1
    if(l>=N):
        print("Número máximo de iterações excedido.")
        return T, -1
   
     
def metodoquasinewton(T0,tol,N): # método quasi-Newton
    A0=np.zeros((4,4))
    A=A0
    Ge=np.zeros(N)
    v=np.zeros((4,1))
    w=np.zeros((4,1))
    y=np.zeros((4,1))
    z=np.zeros((4,1))
    s=np.zeros((4,1))
    ut=np.zeros((1,4))
    p=0.0
    T=np.zeros((4,1))
    A0=jacobianaf(T0)
    v[:,0]=f(T0)
    A=np.linalg.inv(A0) # inversa de A0
    s=-np.dot(A,v)
    norma=((s[0,0])**2.0+(s[1,0])**2.0+(s[2,0])**2.0+(s[3,0])**2.0)**(1.0/2.0)
    Ge[0]=norma
    T[:,0]=T0+s[:,0]
    l=2
    print("||s(",l-2,")||=",norma)
    print("Teta(",l-2,")=",T0)
    print("f(T(",l-2,"))=",v)
    print("A(T(",l-2,"))=",A)
    while(l<=N):
        print("T(",l-1,")=",T)
        w[:,:]=v[:,:]
        v[:,0]=f(T)
        print("f(T(",l-1,"))=",v)
        y=v-w
        z=-np.dot(A,y)
        p=-np.dot(s.T,z)
        ut=np.dot(s.T,A)
        A=A+(1.0/p)*(np.dot((s+z),ut))
        print("A(T(",l-1,"))=",A)
        s=-np.dot(A,v)
        T=T+s
        norma=((s[0,0])**2.0+(s[1,0])**2.0+(s[2,0])**2.0+(s[3,0])**2.0)**(1.0/2.0)
        Ge[l-1]=norma
        print("||s(",l-1,")||=",norma)
        if(norma<tol):
            print("QUASI-NEWTON")
            print("Teta(",l-1,")=",T,"É a aproximação de zero!!!")
            print("f(T)=",f(T[:,0]))
            return T, l, Ge
        l=l+1
    if(l>N):
        print("Número máximo de iterações atingido.")
        return T,-1
        

##########################
    

######## Execução do experimento ########


tol=10.0**(-8.0) # tolerância

N=100# número máximo de iterações


T0=np.zeros(4) # ponto inicial
T1=np.zeros(4)
T2=np.zeros(4)
Te=np.zeros(4)
E1=np.zeros(N)
E2=np.zeros(N)
for i in range(4):
    T0[i]=np.pi/4.0

l1=0
l2=0

t=time.time()
T1,l1,E1=metododenewton(T0,tol,N)
print("tempo de execução",time.time()-t)
print("Convergiu depois de ",l1," iterações.")


t=time.time()
T2,l2,E2=metodoquasinewton(T0,tol,N)
print("tempo de execução",time.time()-t)
print("Convergiu depois de ",l2," iterações.")
 

Ge1=np.zeros(l1)
Ge2=np.zeros(l2)
I1=np.zeros(l1)
I2=np.zeros(l2)

for i in range(l1):
    Ge1[i]=np.log(E1[i])
    I1[i]=i


for i in range(l2):
    Ge2[i]=np.log(E2[i])
    I2[i]=i

print(T1,T2)

Te[:]=T1[:]-T2[:,0]
normae=((Te[0])**2.0+(Te[1])**2.0+(Te[2])**2.0+(Te[3])**2.0)**(1.0/2.0)
print("Distância entre as aproximações: ",normae)
#########################################


######## Plotando os gráficos ########

pontos1=plt.scatter(I1,Ge1)
linha1,=plt.plot(I1,Ge1,label="ln(||Teta(l)-Teta(l-1)||)",color='red',ls='-')
plt.xlabel("l")
plt.ylabel("ln(||T(l)-T(l-1)||)")
plt.legend(handles=[linha1,pontos1],loc='best')
plt.show() # plotando o log do erro para o método de Newton



pontos2=plt.scatter(I2,Ge2)
linha2,=plt.plot(I2,Ge2,label="ln(||Teta(l)-Teta(l-1)||)",color='blue',ls='-')
plt.xlabel("l")
plt.ylabel("ln(||T(l)-T(l-1)||)")
plt.legend(handles=[linha2,pontos2],loc='best')
plt.show() # plotando o log do erro para o método quasi-Newton


######################################