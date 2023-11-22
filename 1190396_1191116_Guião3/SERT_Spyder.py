from pyfirmata import Arduino, util

import time
import numpy as np
import matplotlib.pyplot as plt

#%%
board = Arduino('COM3')

iterator = util.Iterator(board)
iterator.start()

# Entradas analógicas
T_S1 = board.get_pin('a:0:i') # A0
T_S2 = board.get_pin('a:1:i') # A1

# Saídas PWM
S1 = board.get_pin('d:10:p')  # Pino 10
S2 = board.get_pin('d:9:p')   # Pino 9

time.sleep(1)

# Função de conversão
def TEMP(val):
    return (val*5000.0-500.0)/10.0

# Funções temperaturas
def T1():
    return TEMP(T_S1.read())

def T2():
    return TEMP(T_S2.read())

#%% Resposta ao degrau
Ts = 1 # período de amostragem
tf = 600.0 # tempo final (10) - 400.0
n = int(np.round(tf/Ts+1)) # número de amostras
temp1 = np.zeros(n) # temperatura S1
temp2 = np.zeros(n) # temperatura S2
t = np.linspace(0,n-1,n)*Ts # tempo
u_s1 = 0.8 # 50% PWM - alternar com o u_s2 - quando é 0 os valores do sensor S1 não são considerádos
u_s2 = 0.6   # 50% PWM - alternar com o u_s1 - quando é 0 os valores do sensor S2 não são considerádos
# Entrada de controlo
u1 = np.ones(n)*u_s1
u2 = np.ones(n)*u_s2
print('u1   T1   u2   T2')
S1.write(u_s1)
S2.write(u_s2)
for i in range(n):
    # ler temperatura
    temp1[i] = T1()
    temp2[i] = T2()
    print(str(u1[i])+'   '+str(temp1[i])+'   '+str(u2[i])+'   '+str(temp2[i]))
    time.sleep(Ts)

# desligar S1
S1.write(0.0)   
# desligar S2
S2.write(0.0)
board.exit() # termina comunicação com placa

#%% Guardar dados

dados = np.vstack((t,temp1,temp2)).T
np.savetxt('dados_reglinear.txt',dados,delimiter=',',\
        header='t,u,T1',comments='')



#%% Gráficos
if u_s1!=0:
    plt.figure()
    ax = plt.subplot(2,1,1)
    ax.grid()
    plt.plot(t,temp1,'k-',label='T1')
    plt.ylabel('Temperatura ($^oC$)')
    plt.legend(loc='best')
    ax = plt.subplot(2,1,2)
    ax.grid()
    plt.plot(t,u1,'b-',label='u')
    plt.ylabel('Controlo (%)')
    plt.xlabel('tempo (s)')
    plt.legend()
    plt.savefig('degrau_S1.png')
    plt.show()

if u_s2!=0:
    plt.figure()
    ax = plt.subplot(2,1,1)
    ax.grid()
    plt.plot(t,temp2,'k-',label='T2')
    plt.ylabel('Temperatura ($^oC$)')
    plt.legend(loc='best')
    ax = plt.subplot(2,1,2)
    ax.grid()
    plt.plot(t,u2,'b-',label='u')
    plt.ylabel('Controlo (%)')
    plt.xlabel('tempo (s)')
    plt.legend()
    plt.savefig('degrau_S2.png')
    plt.show()