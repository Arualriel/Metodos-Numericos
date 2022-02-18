import numpy as np


def U(t,u):
    return -0.5*u+2+t

def Euler(t,u,h):
    return u+h*U(t,u)

def EulerM(t,u,h):
    return u+(h/2.0)*(U(t,u)+U(t+h,Euler(t,u,h)))


h1=10.0**(-1.0)
h2=10.0**(-2.0)
h3=10.0**(-3.0)
h4=10.0**(-4.0)
h5=10.0**(-5.0)

ue=6.85224527770107
u10=8.0
u20=8.0
t0=0.0
h=h5
while(t0<1.0):
    u10=Euler(t0,u10,h)
    #print("euler",t0,u10)
    t0=t0+h



t0=0.0

while(t0<1.0):
    u20=EulerM(t0,u20,h)
    #print("euler modificado",t0,u20)
    t0=t0+h
    

print("erro",np.abs(u10-ue),np.abs(u20-ue))

