from scipy.integrate import quad
from scipy.integrate import odeint 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.linalg import inv, det, solve 
import math
import sounddevice as sd


# def f(x):
# 	return x**2

# result, error = quad(f, 0, 1)

# print(f"l'integrale de f(x) entre 0 et 1 est {result} avec une erreur de {error}")






# def model(y,t):
# 	dydt = -y +t 
# 	return dydt

# y0 = 1

# t = np.linspace(0,10,100)

# y = odeint(model, y0, t)


# plt.plot(t,y)
# plt.xlabel('t')
# plt.ylabel('f(x)')
# plt.show()



 

# A = np.array([[1,2],[3,4]])
# print(f"La matrice A=\n{A}")
# print()

# A_inv = inv(A)
# print(f"l'inverse de A est : \n{A_inv}")
# print()

# A_det = det(A)
# print(f"le determinant de A est : \n{A_det}")
# print()

# b = np.array([1,2])
# x = solve(A,b)

# print(f"La solution du système est : x = {x[0]}, y = {x[1]}")




#Exercice 1 
# def f(x): 
# 	return math.exp((-x**2))

# result, error = quad(f, 0, 2)

# print(f"l'integrale de f(x) entre 0 et 2 est {result} avec une erreur de {error}")


#Exercice 2 
#Résoudre l’équation différentielle dy/dt = -2y + sin(t) avec la condition initiale y(0) = 0, sur l’intervalle de temps [0, 10].

# def equadiff(y,t) : 
# 	dydt = -2*y + math.sin(t)
# 	return dydt

# y0 = 0

# t = np.linspace(0,10,100)

# y = odeint(equadiff, y0, t)


# plt.plot(t,y)
# plt.xlabel('t')
# plt.ylabel('f(x)')
# plt.show()


#Exercice 3 
# A = np.array([[3,4],[2,-1]])

# b = np.array([7,1])

# x = solve(A,b)

# print(f"La solution du système est : x = {x[0]}, y = {x[1]}")

# def model(S):
# 	Sp = np.array(S[1], -(9/L)*math.sin(S[0]))
# 	return Sp


# def model(y,t):
# 	dydt = -y +t 
# 	return dydt

# y0 = 10

# t = np.linspace(0,10,100)

# y = odeint(model, y0, t)


# plt.plot(t,y)
# plt.xlabel('t')
# plt.ylabel('f(x)')
# plt.show()

g = 9.81
l = 1
theta_deg = 10 
theta = theta_deg * np.pi/180
t0 = 0
tfinal = 10

s_initt = [theta, 0]
t = np.linspace(t0, tfinal, 100)

def model(S,t):
	Sp = [S[1], -(9/l)*math.sin(S[0])]
	return Sp

S=odeint(model, s_initt,t)

print(S)


plt.plot(t,S[:,0], color = "red")
plt.xlabel("Temps")
plt.ylabel("Position")
plt.show()

