import matplotlib.pyplot as plt
import numpy as np
from qutip import *
import scipy.integrate as integrate
import pandas as pd
import scipy.constants as sc
from tqdm import tqdm

import scipy.signal as ss
# QuTiP Hamiltonian functions

def hamiltonian(Ec, Ej, N, ng):
    """
    Return the charge qubit hamiltonian as a Qobj instance.

    The size of the Hamiltonian matrix will be 2*N + 1
    """
    m = np.diag(4 * Ec * (np.arange(-N,N+1)-ng)**2) + 0.5 * Ej * (np.diag(-np.ones(2*N), 1) + np.diag(-np.ones(2*N), -1))
    return Qobj(m)

def Hc(Ec, Ej, N, ng):
    """
    Reutrn the charging term of the charge qubit hamiltonian as a Qobj instance
    """
    m = np.diag(4 * Ec * (np.arange(-N,N+1)-ng)**2) 
    
    return Qobj(m)

def Hj(Ec, Ej, N, ng):
    """
    Reutrn the Josephson term of the charge qubit hamiltonian as a Qobj instance
    """
    
    m = 0.5 * Ej * (np.diag(-np.ones(2*N), 1) +  np.diag(-np.ones(2*N), -1))
    
    return Qobj(m)

def Hr(w):
    """
    This Hamiltonian will be used to model the cavity
    """
    m = sc.hbar * w *   create(21) * destroy(21)
    return Qobj(m)

def Hi(N, B, V):
    m = 2 * B * sc.elementary_charge * V * N * (create() + destroy())
    return Qobj(m)

def EJ_tanh_pulse(t, P0, R0, T):
    return (1-P0*(1+np.tanh(-R0+(t/T)*2*R0))/2.0)

def EJ_tanh_pulse_inverse_t(value, P0, R0, T):
    a = (2*((1-value)/P0) - 1)
    return (R0 + (0.5 * (np.log(1+a) - np.log(1-a)))) * (T/(2*R0))

def EJ_double_pulse(t, P0, R0, T1, T2):
    return EJ_tanh_pulse(t, P0, R0, T1) + (EJ_tanh_pulse(t - T2, -P0, R0, T1) - 1)

# Time depenedent part (drive) of Hamiltonian
def Hd_coeff(t,args):
    Percentage=args['Percentage']
    Range=args['Range']
    T1=args['T1']
    T2=args['T2']
    return EJ_double_pulse(t, Percentage, Range, T1, T2)

# Function for plottting the overlap between the original and the first excited state

def calculate_overlap(t, N, Ec, Ej, ng, T1, T2, T_max, P0, R0, Points, states, initial_state):
    ng_vec = np.linspace(-1, 1, 200) # ????

    # Calculate the eigenstates and eigenenergies of the original Hamiltonian
    energies = np.array([hamiltonian(Ec, Ej, N, ng).eigenenergies() for ng in ng_vec])
    evals, ekets = hamiltonian(Ec, Ej, N, ng).eigenstates()

    # Decompose the Hamiltonian into the charge part, Josephson part and the time-dependent part
    Htot = [Hc(Ec,Ej,N,ng), [Hj(Ec,Ej,N,ng), Hd_coeff]]

    # set initial state as the ground state of the time independent hamiltonian
    psi0=ekets[initial_state]

    # calculate the eigenstates of the time independent hamiltonian using the new Josephson energy determined by the EJ pulsing at time T
    evalsf, eketsf = hamiltonian(Ec, Ej*EJ_double_pulse(T_max, P0, R0, T1, T2), N, ng).eigenstates()

    # solve the Schrodinger equation using the time dependent Hamiltonian and the initial state to model the evolution of the state adiabatically or instantaneously (dependent on the pulse slope) 
    output = sesolve(Htot, psi0, t, args={'Percentage': P0, 'Range': R0,'T1':T1, 'T2':T2})
    
    overlap_list = []
    for state in states:
        """
        States determines what states we are interested in overlap to the initial state

        Points: number of points along the time axis
        """
        Ovrlp=np.zeros(Points) # shows the time evolution of the state

        # Calculate the corresponding overlap - integral of state A and state B over all time
        if state < 0:
            for n in range(0,Points):
                # overlap of the output to the initial state
                Ovrlp[n]=np.abs(psi0.dag().overlap(output.states[n]))**2
        else:
            for n in range(0,Points):
                # overlap of the output to the eigenstate (state) at time T of the time-independent Hamiltonian 
                Ovrlp[n]=np.abs(eketsf[state].dag().overlap(output.states[n]))**2

        overlap_list.append(Ovrlp)

    return overlap_list


if __name__ == "__main__":
    print("Running...")

    N = 10
    Ec = 1.0
    Ej = 50.0
    ng=0.5
    T_max = 300
    T1 = 100
    T2 = 350
    Points=10000
    P0=0.99
    R0 = 100 * np.pi
    cavity_resonance = 1.2
    g = 0.5


    tlist = np.linspace(0, T_max, Points)
    fig, ax = plt.subplots( nrows=2, ncols= 1,figsize=(15, 8))
    #Ovrlps1 = calculate_overlap(tlist, N, Ec, Ej, ng, T1, T2, T_max, P0, R0, Points, states = [-1, 0], initial_state=1)
    Ovrlps0 = calculate_overlap(tlist, N, Ec, Ej, ng, T1, T2, T_max, P0, R0, Points, cavity_resonance, g, states = [-1, 0], initial_state=0)

    sp = np.fft.rfft(Ovrlps0[0][tlist > T1*0.7])
    freq = np.fft.rfftfreq(len(tlist[tlist > T1*0.7]), tlist[1] - tlist[0])

    ax[0].plot(tlist, Ovrlps0[0])
    ax[1].plot(freq, np.abs(sp))
    plt.show()