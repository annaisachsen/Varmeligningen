import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


alpha = 1  
L_x = 1
L_y = 1
T = 10

N_x = 16
N_y = 16 #Hvis N_x og N_y er større enn 16 så blir det ustabilt med N_t= 10000
N_t = 10000

h_x = L_x / (N_x - 1)
h_y = L_y / (N_y - 1)
k = T / (N_t - 1)
print("h_x: " + str(h_x))
print("h_y: " + str(h_y))
print("k: " + str(k))

gamma_x = alpha * k / h_x**2
gamma_y = alpha * k / h_y**2

x = np.linspace(0, L_x, N_x)
y = np.linspace(0, L_y, N_y)

u = np.zeros((N_x, N_y, N_t))

if gamma_x >=1/4:
    print("FUCKED GAMMA_X " + str(gamma_x))
    print("k er: " + str(k))
else: 
    print(gamma_x)
    
if gamma_y >=1/4:
    print("FUCKED GAMMA_Y " + str(gamma_y))
    print("k er: " + str(k))
else: 
    print(gamma_y)

#Initalbetingelser
for i in range(1, N_x - 1):
    for j in range(1, N_y - 1):
        u[i, j, 0] = np.sin(np.pi * x[i]) * np.sin(np.pi * y[j])
        
u[:, 0, :] = 0  # Venstre side
u[:, -1, :] = 0  # Høyre side
u[0, :, :] = 0  # Topp
u[-1, :, :] = 0  # Bunn


# Eksplisitt 
for n in range(0, N_t - 1):
    for i in range(1, N_x - 1):
        for j in range(1, N_y - 1):
            u[i, j, n+1] = u[i, j, n] + gamma_x * (u[i+1, j, n] - 2*u[i, j, n] + u[i-1, j, n]) \
                                         + gamma_y * (u[i, j+1, n] - 2*u[i, j, n] + u[i, j-1, n])

fig, ax = plt.subplots()
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    global cax
    cax = ax.imshow(u[:, :, 0], extent=(0, L_x, 0, L_y), origin='lower', vmin=np.min(u), vmax=np.max(u))
    fig.colorbar(cax)
    return cax, time_text

def update(frame):
    cax.set_data(u[:, :, frame])
    time_text.set_text('Tid: {:.2f}s'.format(frame * k))
    return cax, time_text

# Lager animasjonen
ani = animation.FuncAnimation(fig, update, frames=N_t, init_func=init, blit=False, interval=50)


plt.title("Temperaturfordeling over tid")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

