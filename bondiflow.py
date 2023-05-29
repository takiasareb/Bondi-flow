#Bondi Transonic Behaviour
#Saikat Bera

import numpy as np
import matplotlib.pyplot as plt

#constants
E = 0.002
gm = 1.4
n = 1/(gm-1) 
N = 10000

#critical values
rc = (n - 1.5)/(2*E)
print("critical radius : ",rc)
ac = 1/((2*rc)**0.5)
uc = ac

#roots (slopes) of dudr at r=r_c
a = - (2 + 1/n)
b = - (2/n) * ((2/(rc**3))**0.5)
c = (1 - (2/n))/(rc**3)

dudr_c = np.zeros(2)
dudr_c[0] = (- b + np.sqrt( b**2 - 4*a*c) )/(2*a)
dudr_c[1] = (- b - np.sqrt( b**2 - 4*a*c) )/(2*a)

"""
#IGNORE:
dadr_c_mod = np.sqrt( (3/4) * (1/(rc**3)) * (1/(n*(2*n-1))) )
dadr_c = [ dadr_c_mod, -dadr_c_mod ]
"""

#slopes of dadr at r=r_c
dadr_c = np.zeros(2)
for i in range(2):
    dadr_c[i] = - ( 1/(rc**2) + ac * dudr_c[i] )/ (2*n*ac)
    #dadr_c[i] = - (0.5*dudr_c[i] + 2*uc/rc)/(n*uc/ac)
    
#print("Slopes of u at r=rc : ", dudr_c)
#print("Slopes of a at r=rc : ", dadr_c)

#equations
def dudr(u,a,r):
    return (1/(r**2) - (2*(a**2))/r)/((a**2)/u - u)
def dadr(u,a,r):
    return ( 1/(2*n*(r**2)) - (u**2)/(n*r) )/((u**2)/a - a)

#rk4 method 
def rk4(dudr, dadr, u0, a0, r0, rn):
    h = (rn - r0)/N

    #initializing r,u and a arrays
    r = [r0]
    u = [u0]
    a = [a0]

    i=0
    while i < N :
    
        k1 = h * dudr(u0,a0,r0)
        l1 = h * dadr(u0,a0,r0)
        k2 = h * dudr(u0 + k1/2, a0 + l1/2, r0 + h/2)
        l2 = h * dadr(u0 + k1/2, a0 + l1/2, r0 + h/2)
        k3 = h * dudr(u0 + k2/2, a0 + l2/2, r0 + h/2)
        l3 = h * dadr(u0 + k2/2, a0 + l2/2, r0 + h/2)
        k4 = h * dudr(u0 + k3, a0 + l3, r0 + h)
        l4 = h * dadr(u0 + k3, a0 + l3, r0 + h)
    
        k = (k1 + 2*k2 + 2*k3 + k4)/6
        l = (l1 + 2*l2 + 2*l3 + l4)/6
        
        if (u0+k)/(a0+l) < 1 and u[-1]/a[-1]>(u0+k)/(a0+l) and r[0]!=rc+dr and r[0] != rc-dr and r[0]!=r2u[-2500]:
            break
    
        r.append(r0 + h)
        u.append(u0 + k)
        a.append(a0 + l)
    
        i = i + 1
        r0 = r0 + h
        u0 = u0 + k
        a0 = a0 + l

    return [u,a,r]

#difference from rc
dr = 0.00001

#subsonic region at r>rc 
range_2_d = rk4(dudr, dadr, uc + dudr_c[0]*dr, ac + dadr_c[0]*dr, rc + dr, 100*rc)
r2d = range_2_d[2]
a2d = range_2_d[1]
u2d = range_2_d[0]
m2d = []
for i in range(len(r2d)):
    m2d.append(u2d[i]/a2d[i])

#supersonic region at r>rc 
range_2_u = rk4(dudr, dadr,uc + dudr_c[1]*dr, ac + dadr_c[1]*dr, rc + dr, 100*rc)

r2u = range_2_u[2]
a2u = range_2_u[1]
u2u = range_2_u[0]
m2u = []
for i in range(len(r2u)):
    m2u.append(u2u[i]/a2u[i])
    
#subsonic region at r<rc 
range_1_d = rk4(dudr, dadr, uc + dudr_c[0]*dr, ac + dadr_c[0]*dr, rc - dr, rc/100)
r1d = range_1_d[2]
a1d = range_1_d[1]
u1d = range_1_d[0]
m1d = []
for i in range(len(r1d)):
    m1d.append(u1d[i]/a1d[i])

#supersonic region at r<rc 
range_1_u = rk4(dudr, dadr, uc + dudr_c[1]*dr, ac + dadr_c[1]*dr, rc - dr, rc/100)
r1u = range_1_u[2]
a1u = range_1_u[1]
u1u = range_1_u[0]
m1u = []
for i in range(len(r1u)):
    m1u.append(u1u[i]/a1u[i])
    
#supersonic non-transonic
range_u = rk4(dudr, dadr, u2u[-1], a2u[-1], r2u[-2500] , rc/100)
ru = range_u[2]
au = range_u[1]
uu = range_u[0]
mu = []
for i in range(len(ru)):
    mu.append(uu[i]/au[i])

#subsonic non-transonic
range_d = rk4(dudr, dadr, u2d[-1] , a2d[-1], r2d[-2500] , rc/100)
rd = range_d[2]
ad = range_d[1]
ud = range_d[0]
md = []
for i in range(len(rd)):
    md.append(ud[i]/ad[i])
    
#non-physical r>rc upper part
range_2_un = rk4(dudr, dadr,u2u[-2000], a2u[-2000], rc*100, rc)

r2u_n = range_2_un[2]
a2u_n = range_2_un[1]
u2u_n = range_2_un[0]
m2u_n = []
for i in range(len(r2u_n)):
    m2u_n.append(u2u_n[i]/a2u_n[i])

#non-physical  r>rc lower part
range_2_dn = rk4(dudr, dadr, u2d[-2000], a2d[-2000], rc*100, r2u_n[-1])

r2d_n = range_2_dn[2]
a2d_n = range_2_dn[1]
u2d_n = range_2_dn[0]
m2d_n = []
for i in range(len(r2d_n)):
    m2d_n.append(u2d_n[i]/a2d_n[i])
    
#non-physical r<rc upper part
range_1_un = rk4(dudr, dadr,u1u[-10], a1u[-10], rc/100, rc)

r1u_n = range_1_un[2]
a1u_n = range_1_un[1]
u1u_n = range_1_un[0]
m1u_n = []
for i in range(len(r1u_n)):
    m1u_n.append(u1u_n[i]/a1u_n[i])
    
#non-physical  r<rc lower part
range_1_dn = rk4(dudr, dadr, u1d[-10], a1d[-10], rc/100, r1u_n[-1])

r1d_n = range_1_dn[2]
a1d_n = range_1_dn[1]
u1d_n = range_1_dn[0]
m1d_n = []
for i in range(len(r1d_n)):
    m1d_n.append(u1d_n[i]/a1d_n[i])
    

#creating the data file
file = open("bondi2u.dat", "w")
for i in range(len(r2u)):
    file.write(str(r2u[i]) + "\t" + str(u2u[i]) + "\t" + str(a2u[i]) + "\t" + str(m2u[i]) + "\n")
file.close()

file = open("bondi2d.dat", "w")
for i in range(len(r2d)):
    file.write(str(r2d[i]) + "\t" + str(u2d[i]) + "\t" + str(a2d[i]) + "\t" + str(m2d[i]) + "\n") 
file.close()

file = open("bondi1u.dat", "w")
for i in range(len(r1u)):
    file.write(str(r1u[i]) + "\t" + str(u1u[i]) + "\t" + str(a1u[i]) + "\t" + str(m1u[i]) + "\n") 
file.close()

file = open("bondi1d.dat", "w")
for i in range(len(r1d)):
    file.write(str(r1d[i]) + "\t" + str(u1d[i]) + "\t" + str(a1d[i]) + "\t" + str(m1d[i]) + "\n") 
file.close()

file = open("bondi2u_n.dat","w")
for i in range(len(r2u_n)):
    file.write(str(r2u_n[i]) + "\t" + str(u2u_n[i]) + "\t" + str(a2u_n[i]) + "\t" + str(m2u_n[i]) + "\n") 
file.close()

file = open("bondi2d_n.dat","w")
for i in range(len(r2d_n)):
    file.write(str(r2d_n[i]) + "\t" + str(u2d_n[i]) + "\t" + str(a2d_n[i]) + "\t" + str(m2d_n[i]) + "\n") 
file.close()

file = open("bondi1u_n.dat","w")
for i in range(len(r1u_n)):
    file.write(str(r1u_n[i]) + "\t" + str(u1u_n[i]) + "\t" + str(a1u_n[i]) + "\t" + str(m1u_n[i]) + "\n") 
file.close()

#plotting
plt.plot(r1u,m1u, label="supersonic $r<r_c$ region")
plt.plot(r1d,m1d, label="subsonic $r<r_c$ region")
plt.plot(r2u,m2u, label="supersonic $r>r_c$ region")
plt.plot(r2d,m2d, label="subsonic $r>r_c$ region")
plt.plot(rd,md, label="non-transonic subsonic region")
plt.plot(ru,mu,label="non-transonic supersonic region")
plt.plot(r2u_n,m2u_n)
plt.plot(r2d_n,m2d_n)
plt.plot(r1u_n,m1u_n)
plt.plot(r1d_n,m1d_n)
plt.xscale("log")
#plt.yscale("log")
plt.title("Mach number VS Radial distance\nE = " + str(E) + " and γ = " + str(gm))
plt.xlabel("Radial distance --->")
plt.ylabel("Mach number --->")
#plt.legend()
plt.show()
