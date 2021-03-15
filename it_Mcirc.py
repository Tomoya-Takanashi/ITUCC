import sys
sys.path.append('../')
from openfermion.chem import MolecularData
from qiskit import Aer,aqua,execute,QuantumCircuit
from qiskit.visualization.utils import _validate_input_state
from math import pi,degrees,acos,sqrt,cos
from .rtcircuit import RTCIRCUIT
from .rtmol import RTMOL
from . import initial
from joblib import Parallel,delayed
import numpy
import cmath
import vqe

class MCIRC(RTCIRCUIT,RTMOL):

    def __init__(self,molecule):
        #super().__init__()
        super().__init__(molecule)
        #super(RTMOL,self).__init__(molecule)
        M_k_coef = []
        M_p_coef = []
        M_k_pauli = []
        M_p_pauli = []
        for i in range(len(self.vqe_parameter)):
            temp_k_coef = []
            temp_k_pauli = []
            temp_p_coef = []
            temp_p_pauli = []
            for j in range(len(self.vqe_parameter)):
                ans = initial.make_M_element(self.ele_num,self.qubit_num,i,j,self.s_comb,self.d_comb)
                temp_k_coef.append(ans[0])
                temp_k_pauli.append(ans[1])
                temp_p_coef.append(ans[2])
                temp_p_pauli.append(ans[3])
                M_k_coef.append(temp_k_coef)
                M_p_coef.append(temp_p_coef)
                M_k_pauli.append(temp_k_pauli)
                M_p_pauli.append(temp_p_pauli)
                
        self.M_k_coef = M_k_coef
        self.M_p_coef = M_p_coef
        self.M_k_pauli = M_k_pauli
        self.M_p_pauli = M_p_pauli

    def add_kpauli(self,uccsd,ind,m_k,m_p):
        #print(ind,m_k,m_p)
        uccsd.x(self.qubit_num)
        for n in range(self.qubit_num):
            if self.M_k_pauli[m_k][m_p][ind][n] == 'X':
                uccsd.cx(self.qubit_num,n)
            elif self.M_k_pauli[m_k][m_p][ind][n] == 'Y':
                uccsd.cy(self.qubit_num,n)
            elif self.M_k_pauli[m_k][m_p][ind][n] == 'Z':
                uccsd.cz(self.qubit_num,n)
        uccsd.x(self.qubit_num)
        return uccsd
    
    def add_ppauli(self,uccsd,ind,m_k,m_p):
        #print(ind,m_k,m_p)
        for n in range(self.qubit_num):
            if self.M_p_pauli[m_k][m_p][ind][n] == 'X':
                uccsd.cx(self.qubit_num,n)
            elif self.M_p_pauli[m_k][m_p][ind][n] == 'Y':
                uccsd.cy(self.qubit_num,n)
            elif self.M_p_pauli[m_k][m_p][ind][n] == 'Z':
                uccsd.cz(self.qubit_num,n)
        return uccsd

    def update_M(self,param,use_core=1,parallel=False):
        def make(m_k,m_p):
            return  self.M_element(m_k,m_p,param)
        
        if parallel:
            result = Parallel(n_jobs=use_core,backend="threading")([delayed(make)(m_k, m_p) 
                                                                    for m_k in range(self.L_param) for m_p in range(self.L_param)])
            M_matrix = numpy.array(result)
            M_matrix = M_matrix.reshape( self.L_param,  self.L_param )
            
        else:
            M_matrix = numpy.zeros(( self.L_param, self.L_param ))
            for m_k in range(self.L_param):
                for m_p in range(self.L_param):
                    M_matrix[m_k][m_p] = self.M_element(m_k,m_p,param)

        
        M_matrix[numpy.absolute(M_matrix) < 1e-10] = 0.
        return M_matrix

    def M_cal_coef(self,m_k,m_p,d_i,d_j):
        coef = self.M_k_coef[m_k][m_p][d_i].conjugate() * self.M_p_coef[m_k][m_p][d_j]
        return coef
    
    def M_element(self,m_k,m_p,phi):
        
        if m_k < self.L_d_param:
            kOpr_type = 'D'
            kOpr_comb = self.d_comb[m_k]
            kOpr_d_ind = m_k
        elif m_k >= self.L_d_param:
            kOpr_type = 'S'
            kOpr_comb = self.s_comb[m_k - self.L_d_param]
            kOpr_s_ind = m_k - self.L_d_param
            
        if m_p < self.L_d_param:
            pOpr_type = 'D'
            pOpr_comb = self.d_comb[m_p]
            pOpr_d_ind = m_p
        elif m_p >= self.L_d_param:
            pOpr_type = 'S'
            pOpr_comb = self.s_comb[m_p - self.L_d_param]
            pOpr_s_ind = m_p - self.L_d_param

        if m_k == m_p:
            if kOpr_type == 'S' and pOpr_type == 'S':
                exp_val = []
                for d_i  in range(2):                    
                    for  d_j in range(2):
                        if d_i == d_j:
                            exp_val.append(1* self.M_cal_coef(m_k,m_p,d_i,d_j)*2)
                        else:
                            uccsd = self.initial_state(d_i,d_j)
                            uccsd = self.make_SS_dig(uccsd,phi,m_k,m_p,d_i,d_j,kOpr_s_ind)
                            exp_val.append(self.M_cal_coef(m_k,m_p,d_i,d_j) * self.ancila_measure(uccsd)*2)

            elif kOpr_type == 'D' and pOpr_type == 'D':
                exp_val = []
                for d_i  in range(8):
                    for  d_j in range(8):
                        if d_i == d_j:
                            exp_val.append(1* self.M_cal_coef(m_k,m_p,d_i,d_j)*2)
                        else:
                            uccsd = self.initial_state(d_i,d_j)
                            uccsd = self.make_DD_dig(uccsd,phi,m_k,m_p,d_i,d_j,kOpr_d_ind)
                            exp_val.append(self.M_cal_coef(m_k,m_p,d_i,d_j) * self.ancila_measure(uccsd)*2)
                        
        else:
            if kOpr_type == 'S' and pOpr_type == 'S':
                exp_val = []
                for d_i  in range(2):
                    for  d_j in range(2):
                        uccsd = self.initial_state(d_i,d_j)
                        uccsd = self.make_SS(uccsd,phi,m_k,m_p,d_i,d_j,kOpr_s_ind,pOpr_s_ind)
                        exp_val.append(self.M_cal_coef(m_k,m_p,d_i,d_j) * self.ancila_measure(uccsd)*2)

            elif kOpr_type == 'S' and pOpr_type == 'D':
                exp_val = []
                for d_i  in range(2):
                    for  d_j in range(8):
                        uccsd = self.initial_state(d_i,d_j)
                        uccsd = self.make_SD(uccsd,phi,m_k,m_p,d_i,d_j,kOpr_s_ind,pOpr_d_ind)
                        exp_val.append(self.M_cal_coef(m_k,m_p,d_i,d_j) * self.ancila_measure(uccsd)*2)

            elif kOpr_type == 'D' and pOpr_type == 'S':
                exp_val = []
                for d_i  in range(8):
                    for  d_j in range(2):
                        uccsd = self.initial_state(d_i,d_j)
                        uccsd = self.make_DS(uccsd,phi,m_k,m_p,d_i,d_j,kOpr_d_ind,pOpr_s_ind)
                        exp_val.append(self.M_cal_coef(m_k,m_p,d_i,d_j) * self.ancila_measure(uccsd)*2)

            elif kOpr_type == 'D' and pOpr_type == 'D':
                exp_val = []
                for d_i  in range(8):
                    for  d_j in range(8):
                        uccsd = self.initial_state(d_i,d_j)
                        uccsd = self.make_DD(uccsd,phi,m_k,m_p,d_i,d_j,kOpr_d_ind,pOpr_d_ind)
                        exp_val.append(self.M_cal_coef(m_k,m_p,d_i,d_j) * self.ancila_measure(uccsd)*2)

        return sum(exp_val)

    def make_SS_dig(self,uccsd,phi,m_k,m_p,d_i,d_j,kOpr_s_ind):
        amp_s = 0
        amp_d = 0
        # double circuit
        for r1 in range(len(self.d_comb)):
            for r3 in range(8):
                uccsd = self.double_exp(r3,r1,uccsd,amp_s,amp_d,phi)
            amp_d = amp_d + 1

        # single circuit
        for r1 in range(len(self.s_comb)):
            if kOpr_s_ind == r1:
                for r3 in range(2):
                    uccsd = self.single_exp(r3,r1,uccsd,amp_s,amp_d,phi)
                    if r3 == d_i:
                        uccsd = self.add_kpauli(uccsd,d_i,m_k,m_p)                        
                    if r3 == d_j:
                        uccsd = self.add_ppauli(uccsd,d_j,m_k,m_p)
                amp_s = amp_s + 1
            else:
                for r3 in range(2):
                    uccsd = self.single_exp(r3,r1,uccsd,amp_s,amp_d,phi)
                amp_s = amp_s + 1
        return uccsd

    def make_DD_dig(self,uccsd,phi,m_k,m_p,d_i,d_j,kOpr_d_ind):
        amp_s = 0
        amp_d = 0
        for r1 in range(len(self.d_comb)):
            if kOpr_d_ind == r1:
                for r3 in range(8):
                    uccsd = self.double_exp(r3,r1,uccsd,amp_s,amp_d,phi)
                    if r3 == d_i:
                        uccsd = self.add_kpauli(uccsd,d_i,m_k,m_p)
                    if r3 == d_j:
                        uccsd = self.add_ppauli(uccsd,d_j,m_k,m_p)
                amp_d = amp_d + 1
            else:
                for r3 in range(8):
                    uccsd = self.double_exp(r3,r1,uccsd,amp_s,amp_d,phi)
                amp_d = amp_d + 1
                    
        for r1 in range(len(self.s_comb)):
            for r3 in range(2):
                uccsd = self.single_exp(r3,r1,uccsd,amp_s,amp_d,phi)
            amp_s = amp_s + 1
        return uccsd

    def make_SS(self,uccsd,phi,m_k,m_p,d_i,d_j,kOpr_s_ind,pOpr_s_ind):
        amp_s = 0
        amp_d = 0
        for r1 in range(len(self.d_comb)):
            for r3 in range(8):
                uccsd = self.double_exp(r3,r1,uccsd,amp_s,amp_d,phi)
            amp_d = amp_d + 1
                    
        for r1 in range(len(self.s_comb)):
            if kOpr_s_ind == r1 :
                for r3 in range(2):
                    uccsd = self.single_exp(r3,r1,uccsd,amp_s,amp_d,phi)
                    if r3 == d_i:
                        uccsd = self.add_kpauli(uccsd,d_i,m_k,m_p)
                amp_s = amp_s + 1

            elif pOpr_s_ind == r1:
                for r3 in range(2):
                    uccsd = self.single_exp(r3,r1,uccsd,amp_s,amp_d,phi)
                    if r3 == d_j:
                        uccsd = self.add_ppauli(uccsd,d_j,m_k,m_p)
                amp_s = amp_s + 1
            else:
                for r3 in range(2):
                    uccsd = self.single_exp(r3,r1,uccsd,amp_s,amp_d,phi)
                amp_s = amp_s + 1
        return uccsd
    
    def make_DD(self,uccsd,phi,m_k,m_p,d_i,d_j,kOpr_d_ind,pOpr_d_ind):
        amp_s = 0
        amp_d = 0
        for r1 in range(len(self.d_comb)):
            if kOpr_d_ind == r1:
                for r3 in range(8):
                    uccsd = self.double_exp(r3,r1,uccsd,amp_s,amp_d,phi)
                    if r3 == d_i:
                        uccsd = self.add_kpauli(uccsd,d_i,m_k,m_p)
                amp_d = amp_d + 1
            elif pOpr_d_ind == r1:
                for r3 in range(8):
                    uccsd = self.double_exp(r3,r1,uccsd,amp_s,amp_d,phi)
                    if r3 == d_j:
                        uccsd = self.add_ppauli(uccsd,d_j,m_k,m_p)
                amp_d = amp_d + 1
            else:
                for r3 in range(8):
                    uccsd = self.double_exp(r3,r1,uccsd,amp_s,amp_d,phi)
                amp_d = amp_d + 1
            
        for r1 in range(len(self.s_comb)):
            for r3 in range(2):
                uccsd = self.single_exp(r3,r1,uccsd,amp_s,amp_d,phi)
            amp_s = amp_s + 1

        return  uccsd

    def make_SD(self,uccsd,phi,m_k,m_p,d_i,d_j,kOpr_s_ind,pOpr_d_ind):
        amp_s = 0
        amp_d = 0
        for r1 in range(len(self.d_comb)):
            if pOpr_d_ind == r1:
                for r3 in range(8):
                    uccsd = self.double_exp(r3,r1,uccsd,amp_s,amp_d,phi)
                    if r3 == d_j:
                        uccsd = self.add_ppauli(uccsd,d_j,m_k,m_p)
                amp_d = amp_d + 1
            else:
                for r3 in range(8):
                    uccsd = self.double_exp(r3,r1,uccsd,amp_s,amp_d,phi)
                amp_d = amp_d + 1

        for r1 in range(len(self.s_comb)):
            if kOpr_s_ind == r1:
                for r3 in range(2):
                    uccsd = self.single_exp(r3,r1,uccsd,amp_s,amp_d,phi)
                    if r3 == d_i:
                        uccsd = self.add_kpauli(uccsd,d_i,m_k,m_p)
                amp_s = amp_s + 1
            else:
                for r3 in range(2):
                    uccsd = self.single_exp(r3,r1,uccsd,amp_s,amp_d,phi)
                amp_s = amp_s + 1
        return uccsd

    def make_DS(self,uccsd,phi,m_k,m_p,d_i,d_j,kOpr_d_ind,pOpr_s_ind):
        amp_s = 0
        amp_d = 0
        for r1 in range(len(self.d_comb)):
            if kOpr_d_ind == r1:
                for r3 in range(8):
                    uccsd = self.double_exp(r3,r1,uccsd,amp_s,amp_d,phi)
                    if r3 == d_i:
                        uccsd = self.add_kpauli(uccsd,d_i,m_k,m_p)
                amp_d = amp_d + 1
            else:
                for r3 in range(8):
                    uccsd = self.double_exp(r3,r1,uccsd,amp_s,amp_d,phi)
                amp_d = amp_d + 1
                
        for r1 in range(len(self.s_comb)):
            if pOpr_s_ind == r1:
                for r3 in range(2):
                    uccsd = self.single_exp(r3,r1,uccsd,amp_s,amp_d,phi)
                    if r3 == d_j:
                        uccsd = self.add_ppauli(uccsd,d_j,m_k,m_p)
                amp_s = amp_s + 1
            else:
                for r3 in range(2):
                    uccsd = self.single_exp(r3,r1,uccsd,amp_s,amp_d,phi)
                amp_s = amp_s + 1
                
        return uccsd
    
