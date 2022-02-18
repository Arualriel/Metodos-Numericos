import numpy as np

B1=np.matrix([[1.0/2.0,1.0/4.0],[1.0/4.0,1.0]])
B2=np.matrix([[1.0,2.0],[3.0,4.0]])
B3=np.matrix([[1.0,3.0],[0.0,1.0]])
B4=np.matrix([[1.0,1.0],[-1.0,1.0]])

InvB=np.zeros((2,2))

for i in range(16):
    InvB=InvB+(np.identity(2, dtype=float)-B4)**(i)
    if(((i+1)%4==0)and((i+1)!=12)):
        print(InvB)

print(np.linalg.inv(B3))
