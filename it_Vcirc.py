import sys
sys.path.append('../')
from openfermion.chem import MolecularData
from qiskit import Aer,aqua,execute,QuantumCircuit
from qiskit.visualization.utils import _validate_input_state
from math import pi,degrees,acos,sqrt,cos
from . import initial
from .rtcircuit import RTCIRCUIT
from .rtmol import RTMOL
from joblib import Parallel,delayed
import numpy
import cmath
import vqe

class VCIRC(RTMOL,RTCIRCUIT):

    def __init__(self,molecule):        
        super().__init__(molecule)
        V_k_coef = []
        V_k_pauli = []
        for i in range(len(self.vqe_parameter)):
            ans = initial.make_V_element(self.ele_num,self.qubit_num,i,self.s_comb,self.d_comb)
            V_k_coef.append(ans[0])
            V_k_pauli.append(ans[1])

        self.V_k_coef = V_k_coef
        self.V_k_pauli = V_k_pauli    

    def add_pauli(self,uccsd,ind,m_k):
        for n in range(self.qubit_num):
            if self.V_k_pauli[m_k][ind][n] == 'X':
                uccsd.cx(self.qubit_num,n)
            elif self.V_k_pauli[m_k][ind][n] == 'Y':
                uccsd.cy(self.qubit_num,n)
            elif self.V_k_pauli[m_k][ind][n] == 'Z':
                uccsd.cz(self.qubit_num,n)
        return uccsd

    def add_hamilgate(self,uccsd,h_opr):
        for r1 in range(len(h_opr)):
            if h_opr[r1][1] == 'X':
                uccsd.cx(self.qubit_num,h_opr[r1][0])
            elif h_opr[r1][1] == 'Y':
                uccsd.cy(self.qubit_num,h_opr[r1][0])
            elif h_opr[r1][1] == 'Z':
                uccsd.cz(self.qubit_num,h_opr[r1][0])
        return uccsd

    def update_V(self,param,rOpr,rNum,use_core=1,parallel=False):
        V_matrix = numpy.zeros(( self.L_param  ))
        if parallel:
            for m_k in range(self.L_param):
                V_matrix[m_k] = self.V_element(m_k,param,rOpr,rNum,ncore=use_core,paral=True)
            return V_matrix
        else:
            for m_k in range(self.L_param):
                V_matrix[m_k] = self.V_element(m_k,param,rOpr,rNum)
            return V_matrix

    def V_cal_coef(self,m_k,d_i,coef):
        ans = self.V_k_coef[m_k][d_i].real * coef * (-1)
        return ans

    def V_element(self,m_k,phi,rOpr,rNum,ncore=1,paral=False):
                
        if m_k < self.L_d_param:
            kOpr_type = 'D'
            kOpr_comb = self.d_comb[m_k]
            kOpr_d_ind = m_k
        elif m_k >= self.L_d_param:
            kOpr_type = 'S'
            kOpr_comb = self.s_comb[m_k - self.L_d_param]
            kOpr_s_ind = m_k - self.L_d_param
        
        if paral:
            if kOpr_type == 'S':

                def make_s_ele(d_i,d_j,m_k,param,s_ind,rOpr):
                    uccsd = self.initial_state(d_i,d_j)
                    uccsd = self.make_S(uccsd,param,m_k,d_i,s_ind,rOpr[d_j])
                    return self.V_cal_coef(m_k,d_i,rNum[d_j]) * self.ancila_measure(uccsd)*2

                exp_val = Parallel(n_jobs=ncore,backend="threading")([delayed(make_s_ele)(d_i,d_j,m_k,phi,kOpr_s_ind,rOpr) 
                                                                      for d_i in range(2) for d_j in range(len(rOpr))])
                

            elif kOpr_type == 'D':

                def make_d_ele(d_i,d_j,m_k,param,d_ind,rOpr):
                    uccsd = self.initial_state(d_i,d_j)
                    uccsd = self.make_D(uccsd,param,m_k,d_i,d_ind,rOpr[d_j])
                    return self.V_cal_coef(m_k,d_i,rNum[d_j]) * self.ancila_measure(uccsd)*2

                
                exp_val = Parallel(n_jobs=ncore,backend="threading")([delayed(make_d_ele)(d_i,d_j,m_k,phi,kOpr_d_ind,rOpr)
                                                                      for d_i in range(8) for d_j in range(len(rOpr))])
                
        else:
            if kOpr_type == 'S':
                exp_val = []
                for d_i in range(2):
                    for d_j in range(len(rOpr)):
                        uccsd = self.initial_state(d_i,d_j)
                        uccsd = self.make_S(uccsd,phi,m_k,d_i,kOpr_s_ind,rOpr[d_j])
                        exp_val.append(self.V_cal_coef(m_k,d_i,rNum[d_j]) * self.ancila_measure(uccsd)*2)

            elif kOpr_type == 'D':
                exp_val = []
                for d_i in range(8):
                    for d_j in range(len(rOpr)):
                        uccsd = self.initial_state(d_i,d_j)
                        uccsd = self.make_D(uccsd,phi,m_k,d_i,kOpr_d_ind,rOpr[d_j])
                        exp_val.append(self.V_cal_coef(m_k,d_i,rNum[d_j]) * self.ancila_measure(uccsd)*2)

        return sum(exp_val)

    def make_S(self,uccsd,phi,m_k,d_i,kOpr_s_ind,Opr):
        amp_s = 0
        amp_d = 0
        for r1 in range(len(self.d_comb)):
            for r3 in range(8):
                uccsd = self.double_exp(r3,r1,uccsd,amp_s,amp_d,phi)
            amp_d = amp_d + 1
            
        for r1 in range(len(self.s_comb)):
            if kOpr_s_ind == r1:
                for r3 in range(2):
                    uccsd = self.single_exp(r3,r1,uccsd,amp_s,amp_d,phi)
                    if r3 == d_i:
                        uccsd.x(self.qubit_num)
                        uccsd = self.add_pauli(uccsd,d_i,m_k)
                        uccsd.x(self.qubit_num)
                amp_s = amp_s + 1
            else:
                for r3 in range(2):
                    uccsd = self.single_exp(r3,r1,uccsd,amp_s,amp_d,phi)
                amp_s = amp_s + 1

        uccsd = self.add_hamilgate(uccsd,Opr)
        return uccsd

    def make_D(self,uccsd,phi,m_k,d_i,kOpr_d_ind,Opr):
        amp_s = 0
        amp_d = 0
        for r1 in range(len(self.d_comb)):
            if kOpr_d_ind == r1:
                for r3 in range(8):
                    uccsd = self.double_exp(r3,r1,uccsd,amp_s,amp_d,phi)
                    if r3 == d_i:
                        uccsd.x(self.qubit_num)
                        uccsd = self.add_pauli(uccsd,d_i,m_k)
                        uccsd.x(self.qubit_num)
                amp_d = amp_d + 1
            else:
                for r3 in range(8):
                    uccsd = self.double_exp(r3,r1,uccsd,amp_s,amp_d,phi)
                amp_d = amp_d + 1    
                
        for r1 in range(len(self.s_comb)):
            for r3 in range(2):
                uccsd = self.single_exp(r3,r1,uccsd,amp_s,amp_d,phi)
            amp_s = amp_s + 1
            
        uccsd = self.add_hamilgate(uccsd,Opr)
        return uccsd 
    
