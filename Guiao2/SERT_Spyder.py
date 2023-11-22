from pyfirmata import Arduino, util
import time
import numpy as np
import matplotlib.pyplot as plt

#%%
board = Arduino('COM5')

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

def startP():
    S1.write(1.0)
    S2.write(0.0)
    temp_t1 = T1()
    temp_t2 = T2()
    while temp_t1 < 30.0:
        temp_t1 = T1()
        temp_t2 = T2()
        print('Start P, T1: '+str(temp_t1))
        time.sleep(1)

    S1.write(0.0)
    while temp_t2 > 25.0:
        temp_t1 = T1()
        temp_t2 = T2()
        print('Start P, T2: '+str(temp_t2))
        time.sleep(1)
    while temp_t1 > 25.0:
        temp_t1 = T1()
        print('Start P, T1: '+str(temp_t1))
        time.sleep(1)

def wait_P():
    S1.write(0.0)
    S2.write(0.0)
    temp_t1 = T1()
    temp_t2 = T2()
    while temp_t2 > 25.0:
        temp_t1 = T1()
        temp_t2 = T2()
        print('Wait P, T2: '+str(temp_t2))
        time.sleep(1)
    while temp_t1 > 25.0:
        temp_t1 = T1()
        print('Wait P, T1: '+str(temp_t1))
        time.sleep(1)

def startPI():
    S1.write(0.0)
    S2.write(0.0)
    temp_t1 = T1()
    temp_t2 = T2()
    while temp_t1 > 25.0:
        temp_t1 = T1()
        temp_t2 = T2()
        print('Start PI, T1: '+str(temp_t1))
        time.sleep(1)
    S2.write(1.0)
    while temp_t2 < 30.0:
        temp_t2 = T2()
        temp_t1 = T1()
        print('Start PI, T2: '+str(temp_t2))
        time.sleep(1)

    S2.write(0.0)
    while temp_t1 > 25.0:
        temp_t1 = T1()
        temp_t2 = T2()
        print('Start PI, T1: '+str(temp_t1))
        time.sleep(1)
    while temp_t2 > 25.0:
        temp_t1 = T1()
        temp_t2 = T2()
        print('Start PI, T2: '+str(temp_t2))
        time.sleep(1)

def wait_PI():
    S1.write(0.0)
    S2.write(0.0)
    temp_t2 = T2()
    while temp_t2 > 25.0:
        temp_t2 = T2()
        print('Wait PI, T2: '+str(temp_t2))
        time.sleep(1)

def P_control(yref,Kp,n):
    temp0 = T1()
    e = np.zeros(n)
    u = np.ones(n)*25.0
    temp = np.ones(n)*temp0

    for i in range(n):

        e[i] = yref[i] - temp[i]
        u[i] = Kp*e[i]/100
        if u[i] > 1.0:
            u[i] = 1.0
        if u[i] < 0.0:
            u[i] = 0.0 
        if temp[i] > 70.0:
            u[i] = 0.0
        S1.write(u[i])
        if i != n-1:
            temp[i+1] = T1()
        else:
            temp[i] = T1()
        print('P Control, T1: '+str(temp[i]) + ' amostra: ' +str(i) + ' Kp: ' +str(Kp) + ' u: ' +str(u[i]) + ' e: ' +str(e[i]))
        time.sleep(1)

    dados = np.array([temp,u,e])
    return dados       

def PI_control(yref,Kp,Ti,n,Ts):
    temp0 = T2()
    e = np.zeros(n)
    u = np.ones(n)*25.0
    temp = np.ones(n)*temp0
    P = np.zeros(n)
    I = np.zeros(n)

    for i in range(n):

        e[i] = yref[i] - temp[i]
        P[i] = Kp*e[i]
        u[i] = (P[i] + I[i])/100
        if i != n-1:
            I[i+1] = I[i] + e[i]*Kp*Ts/Ti
        else:
            I[i] = I[i]    

        if u[i] > 1.0:
            u[i] = 1.0
        if u[i] < 0.0:
            u[i] = 0.0 
        if temp[i] > 70.0:
            u[i] = 0.0
        S2.write(u[i])
        if i != n-1:
            temp[i+1] = T2()
        else:  
            temp[i] = T2()
        print('PI Control, T2: '+str(temp[i]) + ' amostra: ' +str(i) + ' Kp :'+str(Kp) + ' Ti: ' +str(Ti)  + ' u: ' +str(u[i]) + ' e: ' +str(e[i]))
        time.sleep(1)

    dados = np.array([temp,u,e])
    return dados

def PI_AW(yref,Kp,Ti,n,Ts,Ka):        
    temp0 = T2()
    e = np.zeros(n)
    u = np.ones(n)*25.0
    v = np.ones(n)*25.0
    temp = np.ones(n)*temp0
    P = np.zeros(n)
    I = np.zeros(n)

    for i in range(n):

        e[i] = yref[i] - temp[i]
        P[i] = Kp*e[i]
        v[i] = (P[i] + I[i])/100
   

        u[i] = v[i]

        if u[i] > 1.0:
            u[i] = 1.0
        if u[i] < 0.0:
            u[i] = 0.0 
        if temp[i] > 70.0:
            u[i] = 0.0
        
        if i != n-1:
            I[i+1] = I[i] + e[i]*Kp*Ts/Ti + Ts*Ka*(u[i]-v[i])*100
        else:
            I[i] = I[i] 

        S2.write(u[i])
        if i != n-1:
            temp[i+1] = T2()
        else:  
            temp[i] = T2()
        print('PI Control, T2: '+str(temp[i]) + ' amostra: ' +str(i) + ' Ti: ' +str(Ti) + ' Ka: '+str(Ka) + ' u: ' +str(u[i]) + ' e: ' +str(e[i]))
        time.sleep(1)
    for i in range(n):
        P[i] = P[i]/100
        I[i] = I[i]/100
    dados = np.array([temp,u,e,P,I])
    return dados
    
#%% Resposta ao degrau
Ts = 1 # período de amostragem

tf_s1 = 200.0 # tempo final 
n_s1 = int(np.round(tf_s1/Ts+1)) # número de amostras
t_s1 = np.linspace(0,n_s1-1,n_s1)*Ts # tempo

tf_s2 = 600.0 # tempo final 
n_s2 = int(np.round(tf_s2/Ts+1)) # número de amostras
t_s2 = np.linspace(0,n_s2-1,n_s2)*Ts # tempo

yref_s1 = np.ones(n_s1)*50.0 # referência
yref_s2 = np.ones(n_s2)*50.0 # referência

#%% Controlador P

kp_1p = 1
kp_5p = 5  
kp_10p = 10
kp_20p = 20

startP()
dados_1p = P_control(yref_s1,kp_1p,n_s1)
wait_P()
dados_5p = P_control(yref_s1,kp_5p,n_s1)
wait_P()
dados_10p = P_control(yref_s1,kp_10p,n_s1)
wait_P()
dados_20p = P_control(yref_s1,kp_20p,n_s1)





#%% Controlador PI

kp_pi = 5
ti_20pi = 20
ti_50pi = 50
ti_100pi = 100

startPI()
dados_20pi = PI_control(yref_s2,kp_pi,ti_20pi,n_s2,Ts)
wait_PI()
dados_50pi = PI_control(yref_s2,kp_pi,ti_50pi,n_s2,Ts)
wait_PI()
dados_100pi = PI_control(yref_s2,kp_pi,ti_100pi,n_s2,Ts)







#%% Controlador PI Ziegler-Nichols

k_zn = 0.6136      # ganho
tau_zn = 102.9552  # constante de tempo
tau_d_zn = 16.5712 # constante de tempo de atraso
Kp_zn = (0.9*tau_zn)/(k_zn*tau_d_zn) # ganho do controlador
Ti_zn = tau_d_zn/0.3                 # constante de tempo integral


wait_PI()
dados_ZN = PI_control(yref_s2,Kp_zn,Ti_zn,n_s2,Ts)

#%% Controlador PI IMC

k_imc = 0.6136      # ganho
tau_imc = 102.9552  # constante de tempo
tau_d_imc = 16.5712 # constante de tempo de atraso

tau_c_2 = tau_imc/2
tau_c_5 = tau_imc/5 
tau_c_10 = tau_imc/10
Kp_imc_2 = (1 / k_imc) * (tau_imc / (tau_c_2 + tau_d_imc))   # ganho do controlador
Kp_imc_5 = (1 / k_imc) * (tau_imc / (tau_c_5 + tau_d_imc))   # ganho do controlador
Kp_imc_10 = (1 / k_imc) * (tau_imc / (tau_c_10 + tau_d_imc)) # ganho do controlador
Ti_imc = tau_imc                                             # constante de tempo integral

wait_PI()
dados_IMC_2 = PI_control(yref_s2,Kp_imc_2,Ti_imc,n_s2,Ts)
wait_PI()
dados_IMC_5 = PI_control(yref_s2,Kp_imc_5,Ti_imc,n_s2,Ts)
wait_PI()
dados_IMC_10 = PI_control(yref_s2,Kp_imc_10,Ti_imc,n_s2,Ts)


#%% Controlador PI Anti-Windup
Ka_02 = 0.02                         # Ka = 1/Tt
ka_1 = 0.1
ka_2 = 0.2
Kp_aw = Kp_zn                        # ganho do controlador
Ti_aw = Ti_zn                        # constante de tempo integral

wait_PI()
dados_AW_02 = PI_AW(yref_s2,Kp_aw,Ti_aw,n_s2,Ts,Ka_02)
wait_PI()
dados_AW_1 = PI_AW(yref_s2,Kp_aw,Ti_aw,n_s2,Ts,ka_1)
wait_PI()
dados_AW_2 = PI_AW(yref_s2,Kp_aw,Ti_aw,n_s2,Ts,ka_2)


# desligar S1
S1.write(0.0)   
# desligar S2
S2.write(0.0)
board.exit() # termina comunicação com placa

#%% Guardar dados
#if u_s1!=0:
#    dados = np.vstack((t,u1,temp1)).T
#    np.savetxt('dados_S1.txt',dados,delimiter=',',\
#            header='t,u,T1',comments='')

#if u_s2!=0:
#    dados = np.vstack((t,u2,temp2)).T
#    np.savetxt('dados_S2.txt',dados,delimiter=',',\
#            header='t,u,T2',comments='')

#%% gráficos

plt.figure()
ax = plt.subplot(2,1,1)
ax.grid()
plt.plot(t_s1,yref_s1,'k-',label='Referência')
plt.plot(t_s1,dados_1p[0],'b-',label='T1 (kp=1)')
plt.plot(t_s1,dados_5p[0],'g-',label='T1 (kp=5)')
plt.plot(t_s1,dados_10p[0],'r-',label='T1 (kp=10)')
plt.plot(t_s1,dados_20p[0],'y-',label='T1 (kp=20)')
plt.ylabel('Temperatura ($^oC$)')
plt.legend(loc='best')
ax = plt.subplot(2,1,2)
ax.grid()
plt.plot(t_s1,dados_1p[1],'k-',label='u (kp=1)')
plt.plot(t_s1,dados_5p[1],'b-',label='u (kp=5)')
plt.plot(t_s1,dados_10p[1],'r-',label='u (kp=10)')
plt.plot(t_s1,dados_20p[1],'g-',label='u (kp=20)')
plt.ylabel('Controlo (%)')
plt.xlabel('tempo (s)')
plt.legend()
plt.savefig('Guiao2/controlo_P.png')
plt.show()

plt.figure()
ax = plt.subplot(2,1,1)
ax.grid()
plt.plot(t_s1,dados_1p[2],'k-',label='e (kp=1)')
plt.plot(t_s1,dados_5p[2],'b-',label='e (kp=5)')
plt.plot(t_s1,dados_10p[2],'r-',label='e (kp=10)')
plt.plot(t_s1,dados_20p[2],'g-',label='e (kp=20)')
plt.ylabel('Erro ($^oC$)')
plt.xlabel('tempo (s)')
plt.legend()
plt.savefig('Guiao2/erro_P.png')
plt.show()

plt.figure()
ax = plt.subplot(2,1,1)
ax.grid()
plt.plot(t_s2,yref_s2,'k-',label='Referência')
plt.plot(t_s2,dados_20pi[0],'b-',label='T2 (Ti=20)')
plt.plot(t_s2,dados_50pi[0],'g-',label='T2 (Ti=50)')
plt.plot(t_s2,dados_100pi[0],'r-',label='T2 (Ti=100)')
plt.ylabel('Temperatura ($^oC$)')
plt.legend(loc='best')
ax = plt.subplot(2,1,2)
ax.grid()
plt.plot(t_s2,dados_20pi[1],'k-',label='u (Ti=20)')
plt.plot(t_s2,dados_50pi[1],'b-',label='u (Ti=50)')
plt.plot(t_s2,dados_100pi[1],'r-',label='u (Ti=100)')
plt.ylabel('Controlo (%)')
plt.xlabel('tempo (s)')
plt.legend()
plt.savefig('Guiao2/controlo_PI.png')
plt.show()

plt.figure()
ax = plt.subplot(2,1,1)
ax.grid()
plt.plot(t_s2,yref_s2,'k-',label='Referência')
plt.plot(t_s2,dados_ZN[0],'b-',label='T2 (ZN)')
plt.ylabel('Temperatura ($^oC$)')
plt.legend(loc='best')
ax = plt.subplot(2,1,2)
ax.grid()
plt.plot(t_s2,dados_ZN[1],'k-',label='u (ZN)')
plt.ylabel('Controlo (%)')
plt.xlabel('tempo (s)')
plt.legend()
plt.savefig('Guiao2/controlo_PI_ZN.png')
plt.show()

plt.figure()
ax = plt.subplot(2,1,1)
ax.grid()
plt.plot(t_s2,yref_s2,'k-',label='Referência')
plt.plot(t_s2,dados_IMC_2[0],'b-',label='T2 (tau_c=2)')
plt.plot(t_s2,dados_IMC_5[0],'g-',label='T2 (tau_c=5)')
plt.plot(t_s2,dados_IMC_10[0],'r-',label='T2 (tau_c=10)')
plt.ylabel('Temperatura ($^oC$)')
plt.legend(loc='best')
ax = plt.subplot(2,1,2)
ax.grid()
plt.plot(t_s2,dados_IMC_2[1],'k-',label='u (tau_c=2)')
plt.plot(t_s2,dados_IMC_5[1],'b-',label='u (tau_c=5)')
plt.plot(t_s2,dados_IMC_10[1],'r-',label='u (tau_c=10)')
plt.ylabel('Controlo (%)')
plt.xlabel('tempo (s)')
plt.legend()
plt.savefig('Guiao2/controlo_PI_IMC.png')
plt.show()

plt.figure()
ax = plt.subplot(2,1,1)
ax.grid()
plt.plot(t_s2,yref_s2,'k-',label='Referência')
plt.plot(t_s2,dados_AW_02[0],'b-',label='T2 (Ka = 0.02)')
plt.plot(t_s2,dados_AW_1[0],'g-',label='T2 (Ka = 0.1)')
plt.plot(t_s2,dados_AW_2[0],'r-',label='T2 (Ka = 0.2)')
plt.ylabel('Temperatura ($^oC$)')
plt.legend(loc='best')
ax = plt.subplot(2,1,2)
ax.grid()
plt.plot(t_s2,dados_AW_02[1],'k-',label='u (Ka = 0.02)')
plt.plot(t_s2,dados_AW_1[1],'b-',label='u (Ka = 0.1)')
plt.plot(t_s2,dados_AW_2[1],'r-',label='u (Ka = 0.2)')
plt.plot(t_s2,dados_AW_02[3],'k--',label='P (Ka = 0.02)')
plt.plot(t_s2,dados_AW_1[3],'b--',label='P (Ka = 0.1)')
plt.plot(t_s2,dados_AW_2[3],'r--',label='P (Ka = 0.2)')
plt.plot(t_s2,dados_AW_02[4],'k:',label='I (Ka = 0.02)')
plt.plot(t_s2,dados_AW_1[4],'b:',label='I (Ka = 0.1)')
plt.plot(t_s2,dados_AW_2[4],'r:',label='I (Ka = 0.2)')
plt.ylabel('Controlo (%)')
plt.xlabel('tempo (s)')
plt.legend()
plt.savefig('Guiao2/controlo_PI_AW.png')
plt.show()

plt.figure()
ax = plt.subplot(2,1,1)
ax.grid()
plt.plot(t_s2,yref_s2,'k-',label='Referência')
plt.plot(t_s2,dados_ZN[0],'b-',label='T2 (ZN)')
plt.plot(t_s2,dados_AW_02[0],'g-',label='T2 (Ka = 0.02)')
plt.plot(t_s2,dados_AW_1[0],'r-',label='T2 (Ka = 0.1)')
plt.plot(t_s2,dados_AW_2[0],'y-',label='T2 (Ka = 0.2)')
plt.ylabel('Temperatura ($^oC$)')
plt.legend(loc='best')
plt.savefig('Guiao2/controlo_PI_ZN_AW.png')
plt.show()
