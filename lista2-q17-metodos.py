import math
import numpy as np

x=5.25
expo=0.0
temp=0.0
aux=0
E=np.exp(-5.25)
for i in range(20):
    temp=((x**i)/(math.factorial(i)))
    #print(temp)
    aux=int(temp*10.0**5.0)
    aux1=aux-(aux/5)*5
    print('gggg',aux1)
    temp=float(aux)*(10.0**(-5.0))
    #print(temp)
    #print('aqui',expo,temp,expo-temp,expo+temp)
    if(i%2!=0):
        expo=float(int((expo-temp)*10.0**5.0))*10.0**(-5.0)
    else:
        expo=float(int((expo+temp)*10.0**5.0))*10.0**(-5.0)
    print(expo)
pexpo=0.0

temp=0.0
aux=0
for i in range(20):
    temp=(x**i)/(math.factorial(i))
    #print(temp)
    aux=int(temp*10.0**5.0)
    temp=float(aux)*(10.0**(-5.0))
    pexpo=float(int((pexpo+temp)*10.0**5.0))*10.0**(-5.0)
    #print(temp, i)

print(expo,E,1/pexpo, pexpo, 1/expo)
