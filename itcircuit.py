import sys
#print(sys.path)
from openfermion.chem import MolecularData
from qiskit import Aer,aqua,execute,QuantumCircuit
from qiskit.visualization.utils import _validate_input_state
from math import pi,degrees,acos,sqrt,cos
from .rtmol import RTMOL
import numpy
import cmath
import vqe

class RTCIRCUIT:
    """
    def __init__(self):        
        print('test')
        self.s_gate = [['H','Y'],['Y','H']]
        self.d_gate = [['Y','Y','H','Y'],['Y','Y','Y','H'],['H','Y','H','H'],['Y','H','H','H'],
                       ['H','H','H','Y'],['H','Y','Y','Y'],['Y','H','Y','Y'],['H','H','Y','H']]
        
        self.machine = Aer.get_backend('statevector_simulator')
    """

    def ancila_measure(self,uccsd):        
        uccsd.h(self.qubit_num)
        #uccsd.rx(-pi/2,qubit_num)
        result = execute(uccsd,self.machine).result()
        count = result.get_statevector(uccsd)
        rho = _validate_input_state(count)
        num_rho = int(numpy.log2(len(rho)))
        state = [bin(i)[2:].zfill(num_rho) for i in range(2**num_rho)]
        p0 = 0
        p1 = 0
        
        for r in range(len(count)):
            count[r] = abs(count[r])**2
            
        for r in range(len(count)):
            if state[r][0] == '0':
                p0 = p0 + count[r] 
            if state[r][0] == '1':
                p1 = p1 + count[r]
    
        exp_val =  p0 -1*p1
        uccsd = 0
        return exp_val

    def initial_state(self,d_i,d_j):
        #--- circuit compose ---#
        uccsd = 0
        uccsd = QuantumCircuit(self.qubit_num+1,self.qubit_num+1)
        #theta = cmath.phase(self.k_coef[d_i].conjugate() * self.p_coef[d_j]*i_coef)
        #coef = self.M_k_coef[i].conjugate() * self,p_coef[j]*i_coef
        #--- add h gate ---#
        uccsd.h(self.qubit_num)
        #uccsd.rz(theta,qubit_num)
        #--- initial state ---#
        for qs in range(self.ele_num):
            uccsd.x(qs)
        return uccsd

    def single_exp(self,r3,r1,uccsd,amp_s,amp_d,phi):

        if(self.s_gate[r3][0] == 'H'):
            uccsd.h(self.s_comb[r1][0])
        elif(self.s_gate[r3][0] == 'Y'):
            uccsd.rx(-pi/2,self.s_comb[r1][1])
            
        if(self.s_gate[r3][1] == 'H'):
            uccsd.h(self.s_comb[r1][0])
        elif(self.s_gate[r3][1] == 'Y'):
            uccsd.rx(-pi/2,self.s_comb[r1][1])

        loop_s = self.s_comb[r1][0]
        loop_num = self.s_comb[r1][1] - self.s_comb[r1][0]
        
        for r4 in range(loop_num):
            uccsd.cx(loop_s,loop_s+1)
            #scnot_count += 1
            loop_s = loop_s + 1

        if(r3 == 0):
            uccsd.rz(phi[amp_s+amp_d]*(-1),self.s_comb[r1][1])
        elif(r3 == 1):
            uccsd.rz(phi[amp_s+amp_d],self.s_comb[r1][1])

        loop_s = self.s_comb[r1][1]
        
        for r4 in range(loop_num):
            uccsd.cx(loop_s-1,loop_s)
            #scnot_count += 1
            loop_s = loop_s - 1

        if(self.s_gate[r3][0] == 'H'):
            uccsd.h(self.s_comb[r1][0])
        elif(self.s_gate[r3][0] == 'Y'):
            uccsd.rx(pi/2,self.s_comb[r1][1])

        if(self.s_gate[r3][1] == 'H'):
            uccsd.h(self.s_comb[r1][0])
        elif(self.s_gate[r3][1] == 'Y'):
            uccsd.rx(pi/2,self.s_comb[r1][1])

        return uccsd

    def double_exp(self,r3,r1,uccsd,amp_s,amp_d,phi):
        
        if(self.d_gate[r3][0] == 'H'):
            uccsd.h(self.d_comb[r1][0][0])
        elif(self.d_gate[r3][0] == 'Y'):
            uccsd.rx(-pi/2,self.d_comb[r1][0][0])

        if(self.d_gate[r3][1] == 'H'):
            uccsd.h(self.d_comb[r1][0][1])
        elif(self.d_gate[r3][1] == 'Y'):
            uccsd.rx(-pi/2,self.d_comb[r1][0][1])

        if(self.d_gate[r3][2] == 'H'):
            uccsd.h(self.d_comb[r1][1][0])
        elif(self.d_gate[r3][2] == 'Y'):
            uccsd.rx(-pi/2,self.d_comb[r1][1][0])

        if(self.d_gate[r3][3] == 'H'):
            uccsd.h(self.d_comb[r1][1][1])
        elif(self.d_gate[r3][3] == 'Y'):
            uccsd.rx(-pi/2,self.d_comb[r1][1][1])
        occ_cx_loop = self.d_comb[r1][0][1] - self.d_comb[r1][0][0]
        vir_cx_loop = self.d_comb[r1][1][1] - self.d_comb[r1][1][0]

        occ_loop = self.d_comb[r1][0][0]
        vir_loop = self.d_comb[r1][1][0]
        for r5 in range(occ_cx_loop):
            uccsd.cx(occ_loop,occ_loop+1)
            #dcnot_count += 1
            occ_loop = occ_loop + 1

        uccsd.cx(self.d_comb[r1][0][1],self.d_comb[r1][1][0])
        
        for r5 in range(vir_cx_loop):
            uccsd.cx(vir_loop,vir_loop+1)
            #dcnot_count += 1
            vir_loop = vir_loop + 1

        if(r3 == 0 or r3 == 1 or r3 == 2 or r3 == 3):
            uccsd.rz(phi[amp_s+amp_d]*(-1),self.d_comb[r1][1][1])
        elif(r3 == 4 or r3 == 5 or r3 == 6 or r3 == 7):
            uccsd.rz(phi[amp_s+amp_d],self.d_comb[r1][1][1])

        occ_loop = self.d_comb[r1][0][1]
        vir_loop = self.d_comb[r1][1][1]    
        for r5 in range(vir_cx_loop):
            uccsd.cx(vir_loop-1,vir_loop)
            #dcnot_count += 1
            vir_loop = vir_loop -1
            
        uccsd.cx(self.d_comb[r1][0][1],self.d_comb[r1][1][0])
        
        for r5 in range(occ_cx_loop):
            uccsd.cx(occ_loop-1,occ_loop)
            #dcnot_count += 1
            occ_loop = occ_loop - 1

        if(self.d_gate[r3][0] == 'H'):
            uccsd.h(self.d_comb[r1][0][0])
        elif(self.d_gate[r3][0] == 'Y'):
            uccsd.rx(pi/2,self.d_comb[r1][0][0])

        if(self.d_gate[r3][1] == 'H'):
            uccsd.h(self.d_comb[r1][0][1])
        elif(self.d_gate[r3][1] == 'Y'):
            uccsd.rx(pi/2,self.d_comb[r1][0][1])

        if(self.d_gate[r3][2] == 'H'):
            uccsd.h(self.d_comb[r1][1][0])
        elif(self.d_gate[r3][2] == 'Y'):
            uccsd.rx(pi/2,self.d_comb[r1][1][0])

        if(self.d_gate[r3][3] == 'H'):
            uccsd.h(self.d_comb[r1][1][1])
        elif(self.d_gate[r3][3] == 'Y'):
            uccsd.rx(pi/2,self.d_comb[r1][1][1])

        return uccsd

"""
basis = "sto-3g" #basis set
multiplicity = 1 #spin multiplicity
charge = 0   #total charge for the molecule
#geometry = [("H",(-dis,0,0)),("H",(0,0,0)),("H",(dis,0,0))]
geometry = [("H",(0.37,0,0)),("H",(-0.37,0,0))]
molecule = MolecularData(geometry, basis, multiplicity, charge)

#mol = RTMOL(molecule)
mol_circ = RTCIRCUIT(molecule)
"""
