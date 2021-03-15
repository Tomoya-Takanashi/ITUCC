from qiskit import Aer,aqua,execute,QuantumCircuit
from qiskit.visualization.utils import _validate_input_state
from math import pi,degrees,acos,sqrt,cos
import numpy
import cmath

def make_M_element(ele_num,qubit_num,k_ind,p_ind,single_comb,double_comb):
    p_pauli = []
    k_pauli = []
    
    L_amps = len(single_comb)
    L_ampd = len(double_comb)
    
    if k_ind < L_ampd:
        kOpr_type = 'D'
        C_kOpr = double_comb[k_ind]
        I_double_k = k_ind
    elif k_ind >= L_ampd:
        kOpr_type = 'S'
        C_kOpr = single_comb[k_ind - L_ampd]
        I_single_k = k_ind - L_ampd
 
    if p_ind < L_ampd:
        pOpr_type = 'D'
        C_pOpr = double_comb[p_ind]
        I_double_p = p_ind
    elif p_ind >= L_ampd:
        pOpr_type = 'S'
        C_pOpr = single_comb[p_ind - L_ampd]
        I_single_p = p_ind - L_ampd
    
    s_coef = 1/2 * 1j
    d_coef = 1/8 * 1j
    k_coef = []
    p_coef = []
    if kOpr_type == 'S':
        for n in range(2):
            temp = []
            if n == 0:
                for i in range(qubit_num):
                    if i < C_kOpr[0]:
                        temp.append('I')
                    elif i == C_kOpr[0]:
                        temp.append('X')
                    elif i > C_kOpr[0] and i < C_kOpr[1]:
                        temp.append('Z')
                    elif i == C_kOpr[1]:
                        temp.append('Y')
                    else:
                        temp.append('I')
                k_coef.append(-1 * s_coef)
            elif n == 1:
                for i in range(qubit_num):
                    if i < C_kOpr[0]:
                        temp.append('I')
                    elif i == C_kOpr[0]:
                        temp.append('Y')
                    elif i > C_kOpr[0] and i < C_kOpr[1]:
                        temp.append('Z')
                    elif i == C_kOpr[1]:
                        temp.append('X')
                    else:
                        temp.append('I')
                k_coef.append(s_coef)
            k_pauli.append(temp)
            temp = []
            
    if kOpr_type == 'D':
        for n in range(8):
            temp = []
            if n == 0:
                for i in range(qubit_num):
                    if i == C_kOpr[0][0]:
                        temp.append('Y')
                    elif C_kOpr[0][0] < i and i < C_kOpr[0][1]:
                        temp.append('Z')
                    elif i == C_kOpr[0][1]:
                        temp.append('Y')
                    elif i == C_kOpr[1][0]:
                        temp.append('X')
                    elif C_kOpr[1][0] < i and i < C_kOpr[1][1]:
                        temp.append('Z')
                    elif i == C_kOpr[1][1]: 
                        temp.append('Y')
                    else:
                        temp.append('I')
                k_coef.append(-1 * d_coef)
            elif n == 1:
                for i in range(qubit_num):
                    if i == C_kOpr[0][0]:
                        temp.append('Y')
                    elif C_kOpr[0][0] < i and i < C_kOpr[0][1]:
                        temp.append('Z')
                    elif i == C_kOpr[0][1]:
                        temp.append('Y')
                    elif i == C_kOpr[1][0]:
                        temp.append('Y')
                    elif C_kOpr[1][0] < i and i < C_kOpr[1][1]:
                        temp.append('Z')
                    elif i == C_kOpr[1][1]:
                        temp.append('X')
                    else:
                        temp.append('I')
                k_coef.append(-1 * d_coef)
            elif n == 2:
                for i in range(qubit_num):
                    if i == C_kOpr[0][0]:
                        temp.append('X')
                    elif C_kOpr[0][0] < i and i < C_kOpr[0][1]:
                        temp.append('Z')
                    elif i == C_kOpr[0][1]:
                        temp.append('Y')
                    elif i == C_kOpr[1][0]:
                        temp.append('X')
                    elif C_kOpr[1][0] < i and i < C_kOpr[1][1]:
                        temp.append('Z')
                    elif i == C_kOpr[1][1]:
                        temp.append('X')
                    else:
                        temp.append('I')
                k_coef.append(-1 * d_coef)
            elif n == 3:
                for i in range(qubit_num):
                    if i == C_kOpr[0][0]:
                        temp.append('Y')
                    elif C_kOpr[0][0] < i and i < C_kOpr[0][1]:
                        temp.append('Z')
                    elif i == C_kOpr[0][1]:
                        temp.append('X')
                    elif i == C_kOpr[1][0]:
                        temp.append('X')
                    elif C_kOpr[1][0] < i and i < C_kOpr[1][1]:
                        temp.append('Z')
                    elif i == C_kOpr[1][1]:
                        temp.append('X')
                    else:
                        temp.append('I')
                k_coef.append(-1 * d_coef)
            elif n == 4:
                for i in range(qubit_num):
                    if i == C_kOpr[0][0]:
                        temp.append('X')
                    elif C_kOpr[0][0] < i and i < C_kOpr[0][1]:
                        temp.append('Z')
                    elif i == C_kOpr[0][1]:
                        temp.append('X')
                    elif i == C_kOpr[1][0]:
                        temp.append('X')
                    elif C_kOpr[1][0] < i and i < C_kOpr[1][1]:
                        temp.append('Z')
                    elif i == C_kOpr[1][1]:
                        temp.append('Y')
                    else:
                        temp.append('I')
                k_coef.append(d_coef)
            elif n == 5:
                for i in range(qubit_num):
                    if i == C_kOpr[0][0]:
                        temp.append('X')
                    elif C_kOpr[0][0] < i and i < C_kOpr[0][1]:
                        temp.append('Z')
                    elif i == C_kOpr[0][1]:
                        temp.append('Y')
                    elif i == C_kOpr[1][0]:
                        temp.append('Y')
                    elif C_kOpr[1][0] < i and i < C_kOpr[1][1]:
                        temp.append('Z')
                    elif i == C_kOpr[1][1]:
                        temp.append('Y')
                    else:
                        temp.append('I')
                k_coef.append(d_coef)
            elif n == 6:
                for i in range(qubit_num):
                    if i == C_kOpr[0][0]:
                        temp.append('Y')
                    elif C_kOpr[0][0] < i and i < C_kOpr[0][1]:
                        temp.append('Z')
                    elif i == C_kOpr[0][1]:
                        temp.append('X')
                    elif i == C_kOpr[1][0]:
                        temp.append('Y')
                    elif C_kOpr[1][0] < i and i < C_kOpr[1][1]:
                        temp.append('Z')
                    elif i == C_kOpr[1][1]:
                        temp.append('Y')
                    else:
                        temp.append('I')
                k_coef.append(d_coef)
            elif n == 7:
                for i in range(qubit_num):
                    if i == C_kOpr[0][0]:
                        temp.append('X')
                    elif C_kOpr[0][0] < i and i < C_kOpr[0][1]:
                        temp.append('Z')
                    elif i == C_kOpr[0][1]:
                        temp.append('X')
                    elif i == C_kOpr[1][0]:
                        temp.append('Y')
                    elif C_kOpr[1][0] < i and i < C_kOpr[1][1]:
                        temp.append('Z')
                    elif i == C_kOpr[1][1]:
                        temp.append('X')
                    else:
                        temp.append('I')
                k_coef.append(d_coef)
            k_pauli.append(temp)
    
    if pOpr_type == 'S':
        for n in range(2):
            temp = []
            if n == 0:
                for i in range(qubit_num):
                    if i < C_pOpr[0]:
                        temp.append('I')
                    elif i == C_pOpr[0]:
                        temp.append('X')
                    elif i > C_pOpr[0] and i < C_pOpr[1]:
                        temp.append('Z')
                    elif i == C_pOpr[1]:
                        temp.append('Y')
                    else:
                        temp.append('I')
                p_coef.append(-1 * s_coef)
            elif n == 1:
                for i in range(qubit_num):
                    if i < C_pOpr[0]:
                        temp.append('I')
                    elif i == C_pOpr[0]:
                        temp.append('Y')
                    elif i > C_pOpr[0] and i < C_pOpr[1]:
                        temp.append('Z')
                    elif i == C_pOpr[1]:
                        temp.append('X')
                    else:
                        temp.append('I')
                p_coef.append(s_coef)
            p_pauli.append(temp)
            temp = []
            
    if pOpr_type == 'D':
        for n in range(8):
            temp = []
            if n == 0:
                for i in range(qubit_num):
                    if i == C_pOpr[0][0]:
                        temp.append('Y')
                    elif C_pOpr[0][0] < i and i < C_pOpr[0][1]:
                        temp.append('Z')
                    elif i == C_pOpr[0][1]:
                        temp.append('Y')
                    elif i == C_pOpr[1][0]:
                        temp.append('X')
                    elif C_pOpr[1][0] < i and i < C_pOpr[1][1]:
                        temp.append('Z')
                    elif i == C_pOpr[1][1]: 
                        temp.append('Y')
                    else:
                        temp.append('I')
                p_coef.append(-1 * d_coef)
            elif n == 1:
                for i in range(qubit_num):
                    if i == C_pOpr[0][0]:
                        temp.append('Y')
                    elif C_pOpr[0][0] < i and i < C_pOpr[0][1]:
                        temp.append('Z')
                    elif i == C_pOpr[0][1]:
                        temp.append('Y')
                    elif i == C_pOpr[1][0]:
                        temp.append('Y')
                    elif C_pOpr[1][0] < i and i < C_pOpr[1][1]:
                        temp.append('Z')
                    elif i == C_pOpr[1][1]:
                        temp.append('X')
                    else:
                        temp.append('I')
                p_coef.append(-1 * d_coef)
            elif n == 2:
                for i in range(qubit_num):
                    if i == C_pOpr[0][0]:
                        temp.append('X')
                    elif C_pOpr[0][0] < i and i < C_pOpr[0][1]:
                        temp.append('Z')
                    elif i == C_pOpr[0][1]:
                        temp.append('Y')
                    elif i == C_pOpr[1][0]:
                        temp.append('X')
                    elif C_pOpr[1][0] < i and i < C_pOpr[1][1]:
                        temp.append('Z')
                    elif i == C_pOpr[1][1]:
                        temp.append('X')
                    else:
                        temp.append('I')
                p_coef.append(-1 * d_coef)
            elif n == 3:
                for i in range(qubit_num):
                    if i == C_pOpr[0][0]:
                        temp.append('Y')
                    elif C_pOpr[0][0] < i and i < C_pOpr[0][1]:
                        temp.append('Z')
                    elif i == C_pOpr[0][1]:
                        temp.append('X')
                    elif i == C_pOpr[1][0]:
                        temp.append('X')
                    elif C_pOpr[1][0] < i and i < C_pOpr[1][1]:
                        temp.append('Z')
                    elif i == C_pOpr[1][1]:
                        temp.append('X')
                    else:
                        temp.append('I')
                p_coef.append(-1 * d_coef)
            elif n == 4:
                for i in range(qubit_num):
                    if i == C_pOpr[0][0]:
                        temp.append('X')
                    elif C_pOpr[0][0] < i and i < C_pOpr[0][1]:
                        temp.append('Z')
                    elif i == C_pOpr[0][1]:
                        temp.append('X')
                    elif i == C_pOpr[1][0]:
                        temp.append('X')
                    elif C_pOpr[1][0] < i and i < C_pOpr[1][1]:
                        temp.append('Z')
                    elif i == C_pOpr[1][1]:
                        temp.append('Y')
                    else:
                        temp.append('I')
                p_coef.append(d_coef)
            elif n == 5:
                for i in range(qubit_num):
                    if i == C_pOpr[0][0]:
                        temp.append('X')
                    elif C_pOpr[0][0] < i and i < C_pOpr[0][1]:
                        temp.append('Z')
                    elif i == C_pOpr[0][1]:
                        temp.append('Y')
                    elif i == C_pOpr[1][0]:
                        temp.append('Y')
                    elif C_pOpr[1][0] < i and i < C_pOpr[1][1]:
                        temp.append('Z')
                    elif i == C_pOpr[1][1]:
                        temp.append('Y')
                    else:
                        temp.append('I')
                p_coef.append(d_coef)
            elif n == 6:
                for i in range(qubit_num):
                    if i == C_pOpr[0][0]:
                        temp.append('Y')
                    elif C_pOpr[0][0] < i and i < C_pOpr[0][1]:
                        temp.append('Z')
                    elif i == C_pOpr[0][1]:
                        temp.append('X')
                    elif i == C_pOpr[1][0]:
                        temp.append('Y')
                    elif C_pOpr[1][0] < i and i < C_pOpr[1][1]:
                        temp.append('Z')
                    elif i == C_pOpr[1][1]:
                        temp.append('Y')
                    else:
                        temp.append('I')
                p_coef.append(d_coef)
            elif n == 7:
                for i in range(qubit_num):
                    if i == C_pOpr[0][0]:
                        temp.append('X')
                    elif C_pOpr[0][0] < i and i < C_pOpr[0][1]:
                        temp.append('Z')
                    elif i == C_pOpr[0][1]:
                        temp.append('X')
                    elif i == C_pOpr[1][0]:
                        temp.append('Y')
                    elif C_pOpr[1][0] < i and i < C_pOpr[1][1]:
                        temp.append('Z')
                    elif i == C_pOpr[1][1]:
                        temp.append('X')
                    else:
                        temp.append('I')
                p_coef.append(d_coef)
            p_pauli.append(temp)
    return k_coef,k_pauli,p_coef,p_pauli


def make_V_element(ele_num,qubit_num,k_ind,single_comb,double_comb):
    i_coef = 1
    k_pauli = []
    L_amps = len(single_comb)
    L_ampd = len(double_comb)

    if k_ind < L_ampd:
        kOpr_type = 'D'
        C_kOpr = double_comb[k_ind]
        I_double_k = k_ind
    elif k_ind >= L_ampd:
        kOpr_type = 'S'
        C_kOpr = single_comb[k_ind - L_ampd]
        I_single_k = k_ind - L_ampd
                    
    k_coef = []
    p_coef = []

    s_coef = -1/2 
    d_coef = -1/8 
    if kOpr_type == 'S':
        for n in range(2):
            temp = []
            if n == 0:
                for i in range(qubit_num):
                    if i < C_kOpr[0]:
                        temp.append('I')
                    elif i == C_kOpr[0]:
                        temp.append('X')
                    elif i > C_kOpr[0] and i < C_kOpr[1]:
                        temp.append('Z')
                    elif i == C_kOpr[1]:
                        temp.append('Y')
                    else:
                        temp.append('I')
                k_coef.append(-1 * s_coef)
            elif n == 1:
                for i in range(qubit_num):
                    if i < C_kOpr[0]:
                        temp.append('I')
                    elif i == C_kOpr[0]:
                        temp.append('Y')
                    elif i > C_kOpr[0] and i < C_kOpr[1]:
                        temp.append('Z')
                    elif i == C_kOpr[1]:
                        temp.append('X')
                    else:
                        temp.append('I')
                k_coef.append(s_coef)
            k_pauli.append(temp)
            temp = []

    if kOpr_type == 'D':
        for n in range(8):
            temp = []
            if n == 0:
                for i in range(qubit_num):
                    if i == C_kOpr[0][0]:
                        temp.append('Y')
                    elif C_kOpr[0][0] < i and i < C_kOpr[0][1]:
                        temp.append('Z')
                    elif i == C_kOpr[0][1]:
                        temp.append('Y')
                    elif i == C_kOpr[1][0]:
                        temp.append('X')
                    elif C_kOpr[1][0] < i and i < C_kOpr[1][1]:
                        temp.append('Z')
                    elif i == C_kOpr[1][1]: 
                        temp.append('Y')
                    else:
                        temp.append('I')
                k_coef.append(-1 * d_coef)
            elif n == 1:
                for i in range(qubit_num):
                    if i == C_kOpr[0][0]:
                        temp.append('Y')
                    elif C_kOpr[0][0] < i and i < C_kOpr[0][1]:
                        temp.append('Z')
                    elif i == C_kOpr[0][1]:
                        temp.append('Y')
                    elif i == C_kOpr[1][0]:
                        temp.append('Y')
                    elif C_kOpr[1][0] < i and i < C_kOpr[1][1]:
                        temp.append('Z')
                    elif i == C_kOpr[1][1]:
                        temp.append('X')
                    else:
                        temp.append('I')
                k_coef.append(-1 * d_coef)
            elif n == 2:
                for i in range(qubit_num):
                    if i == C_kOpr[0][0]:
                        temp.append('X')
                    elif C_kOpr[0][0] < i and i < C_kOpr[0][1]:
                        temp.append('Z')
                    elif i == C_kOpr[0][1]:
                        temp.append('Y')
                    elif i == C_kOpr[1][0]:
                        temp.append('X')
                    elif C_kOpr[1][0] < i and i < C_kOpr[1][1]:
                        temp.append('Z')
                    elif i == C_kOpr[1][1]:
                        temp.append('X')
                    else:
                        temp.append('I')
                k_coef.append(-1 * d_coef)
            elif n == 3:
                for i in range(qubit_num):
                    if i == C_kOpr[0][0]:
                        temp.append('Y')
                    elif C_kOpr[0][0] < i and i < C_kOpr[0][1]:
                        temp.append('Z')
                    elif i == C_kOpr[0][1]:
                        temp.append('X')
                    elif i == C_kOpr[1][0]:
                        temp.append('X')
                    elif C_kOpr[1][0] < i and i < C_kOpr[1][1]:
                        temp.append('Z')
                    elif i == C_kOpr[1][1]:
                        temp.append('X')
                    else:
                        temp.append('I')
                k_coef.append(-1 * d_coef)
            elif n == 4:
                for i in range(qubit_num):
                    if i == C_kOpr[0][0]:
                        temp.append('X')
                    elif C_kOpr[0][0] < i and i < C_kOpr[0][1]:
                        temp.append('Z')
                    elif i == C_kOpr[0][1]:
                        temp.append('X')
                    elif i == C_kOpr[1][0]:
                        temp.append('X')
                    elif C_kOpr[1][0] < i and i < C_kOpr[1][1]:
                        temp.append('Z')
                    elif i == C_kOpr[1][1]:
                        temp.append('Y')
                    else:
                        temp.append('I')
                k_coef.append(d_coef)
            elif n == 5:
                for i in range(qubit_num):
                    if i == C_kOpr[0][0]:
                        temp.append('X')
                    elif C_kOpr[0][0] < i and i < C_kOpr[0][1]:
                        temp.append('Z')
                    elif i == C_kOpr[0][1]:
                        temp.append('Y')
                    elif i == C_kOpr[1][0]:
                        temp.append('Y')
                    elif C_kOpr[1][0] < i and i < C_kOpr[1][1]:
                        temp.append('Z')
                    elif i == C_kOpr[1][1]:
                        temp.append('Y')
                    else:
                        temp.append('I')
                k_coef.append(d_coef)
            elif n == 6:
                for i in range(qubit_num):
                    if i == C_kOpr[0][0]:
                        temp.append('Y')
                    elif C_kOpr[0][0] < i and i < C_kOpr[0][1]:
                        temp.append('Z')
                    elif i == C_kOpr[0][1]:
                        temp.append('X')
                    elif i == C_kOpr[1][0]:
                        temp.append('Y')
                    elif C_kOpr[1][0] < i and i < C_kOpr[1][1]:
                        temp.append('Z')
                    elif i == C_kOpr[1][1]:
                        temp.append('Y')
                    else:
                        temp.append('I')
                k_coef.append(d_coef)
            elif n == 7:
                for i in range(qubit_num):
                    if i == C_kOpr[0][0]:
                        temp.append('X')
                    elif C_kOpr[0][0] < i and i < C_kOpr[0][1]:
                        temp.append('Z')
                    elif i == C_kOpr[0][1]:
                        temp.append('X')
                    elif i == C_kOpr[1][0]:
                        temp.append('Y')
                    elif C_kOpr[1][0] < i and i < C_kOpr[1][1]:
                        temp.append('Z')
                    elif i == C_kOpr[1][1]:
                        temp.append('X')
                    else:
                        temp.append('I')
                k_coef.append(d_coef)
            k_pauli.append(temp)
    return k_coef,k_pauli
