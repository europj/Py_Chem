# Self-Consistent Field Main Script - Unrestricted
# Torin Stetina
# June 27th 2017

import sys
import time
import numpy as np
from numpy import genfromtxt
from numpy import linalg as LA
from scipy.linalg import eig, eigh, inv, expm
import scipy.io

# Internal imports
from mo_transform import ao2mo
from mp2 import*
from response import*
##sys.path.insert(0, '/path/to/application/app/folder')


###############################
##     F U N C T I O N S     ##
###############################

#--------------------------------------
def getIntegrals(molecule):
  ''' Reads integrals from dat/mat files '''

  # Vnn = Nuclear repulsion energy value
  # Vne = Electron-Nuclear attraction matrix
  # T   = Kinetic energy matrix
  # S   = Overlap matrix
  # ERI = Electron-Electron repulsion tensor

  direc = 'test_systems/' + molecule
  Vnn = genfromtxt('./' + direc + '/Vnn.dat',dtype=None)
  Vne = genfromtxt('./' + direc + '/Vne.dat',dtype=None)
  T   = genfromtxt('./' + direc + '/T.dat',dtype=None)
  S   = genfromtxt('./' + direc + '/S.dat',dtype=None)
  ERI = scipy.io.loadmat( direc + '/ERI.mat', squeeze_me=False) 
  return Vnn, Vne, T, S, ERI['ERI']
#--------------------------------------

#--------------------------------------
def calcAlphaBeta(Nelec, mult):
  ''' Defines the number of alpha/beta electrons '''

  # Nelec = # of electrons in system
  # mult  = multiplicity of system

  if Nelec % 2 == 0:
    Na, Nb = Nelec//2, Nelec//2
    Na += (mult-1)/2
    Nb -= (mult-1)/2
  elif Nelec % 2 != 0:
    Na, Nb = Nelec//2, Nelec//2
    Na += (mult-1)/2 + 1
    Nb -= (mult-1)/2
  return Na, Nb
#--------------------------------------

#--------------------------------------
def buildFock(dim,P_a,P_b,h,ERI):
  ''' N^4 loop through basis fuctions to build the Fock matrix
      and calculate the Hartree energy '''

  E0 = 0
  Vee_a = np.zeros((dim,dim))
  Vee_b = np.zeros((dim,dim))
  for m in range(0,dim):
    for n in range(0,dim):
      for k in range(0,dim):
        for l in range(0,dim):
          Vee_a[m,n] += (P_a[k,l] + P_b[k,l]) * ERI[m,n,l,k] \
                                  - P_a[k,l]  * ERI[m,k,l,n]
          Vee_b[m,n] += (P_a[k,l] + P_b[k,l]) * ERI[m,n,l,k] \
                                  - P_b[k,l]  * ERI[m,k,l,n]

      E0 += 0.5 *((P_a[m,n] + P_b[m,n])*h[m,n] \
         +  P_a[m,n]*(h[m,n] + Vee_a[m,n]) \
         +  P_b[m,n]*(h[m,n] + Vee_b[m,n]))
  E0 += Vnn
  F_a = h + Vee_a
  F_b = h + Vee_b

  return E0, F_a, F_b
#--------------------------------------

#--------------------------------------
def deltaP(P,P_old):  
  ''' Calculate change in density matrix  '''
  return max(abs(P.flatten()-P_old.flatten()))
#--------------------------------------

#--------------------------------------
def is_pos_def(x):
  ''' Check if a matrix is positive definite '''
  return np.all(np.linalg.eigvals(x) > 0)
#--------------------------------------

#--------------------------------------
def getOVfvector(F, Nelec, dim):
  ''' Get occupied-virtual block of Fock matrix,
      then convert into vector for QC eigenvalue problem '''

  F_ov = np.zeros((Nelec, dim-Nelec))
  F_ov = F[:Nelec, Nelec:]
  f = np.zeros(((Nelec)*(dim-Nelec)))
  ia = -1
  for i in range(0,Nelec):
    for a in range(0, dim - (Nelec)):
      ia += 1
      f[ia] = F_ov[i,a]
  
  return f
#--------------------------------------

#--------------------------------------
def update_MOs(C_a_old,C_b_old,alp,D_a,D_b,Na,Nb):
  ''' Rotate given molecular orbitals '''

  # form LT of rotation matrix
  K_a = np.zeros((dim, dim))
  K_b = np.zeros((dim, dim))
  ia = 0
  for i in range(0, Na):
    for a in range(Na,dim): 
      K_a[a,i] =  D_a[ia]
      ia += 1
  ia = 0
  for i in range(0, Nb):
    for a in range(Nb,dim): 
      K_b[a,i] =  D_b[ia]
      ia += 1

  # Make UT part
  K_a = (K_a - K_a.T)
  K_b = (K_b - K_b.T)

  # form the unitary transformation matrix
  Ua  = expm(alp*K_a)
  Ub  = expm(alp*K_b)

  # rotate MOs and check if the energy is lower
  C_a = np.dot(C_a_old , Ua)
  C_b = np.dot(C_b_old , Ub)

  return C_a, C_b, Ua, Ub
#--------------------------------------

#--------------------------------------
def NR_step(h,F_a_mo,F_b_mo,f_a,f_b,P_a,P_b,C_a,
            C_b,eps_a,eps_b,Na,Nb,dim,ERI,E0):
  ''' Take a Newton-Raphson step '''

  print "\nDoing Newton-Raphson Step"

  # Build the Hessian
  eriMO    = ao2mo(ERI, [C_a, C_b], False)
  eps_a    = np.diag(F_a_mo)
  eps_b    = np.diag(F_b_mo)
  A, B     = responseAB_UHF(eriMO, [eps_a,eps_b],[Na,Nb])

  # Solve Ax = b
  f_ab     = np.append(f_a,f_b)
  D        = np.linalg.solve(A+B,-f_ab)
  D_a, D_b = D[:Na*(dim-Na)], D[Na*(dim-Na):]
  C_a_old = C_a
  C_b_old = C_b
  old_E   = E0

  # Do backwards line search to find an appropriate step
  E_list, found_step = [], False
  alpha_list = np.arange(1,0,-0.1)
  for alp in alpha_list:

    # rotate the MOs
    C_a, C_b, Ua, Ub = update_MOs(C_a_old,C_b_old,alp,D_a,D_b,Na,Nb)

    # check the current energy
    P_a = np.dot(C_a[:,0:Na],np.transpose(C_a[:,0:Na]))
    P_b = np.dot(C_b[:,0:Nb],np.transpose(C_b[:,0:Nb]))
    E0, F_a, F_b = buildFock(dim,P_a,P_b,h,ERI)
    E_list.append(E0)

  # if we have a lower energy, let's use it
  if np.min(E_list) < old_E:
    found_step = True
    index = E_list.index(np.min(E_list))
    C_a, C_b, Ua, Ub = update_MOs(C_a_old,C_b_old,alpha_list[index],D_a,D_b,Na,Nb)
    P_a = np.dot(C_a[:,0:Na],np.transpose(C_a[:,0:Na]))
    P_b = np.dot(C_b[:,0:Nb],np.transpose(C_b[:,0:Nb]))
    E0, F_a, F_b = buildFock(dim,P_a,P_b,h,ERI)
    print "alpha = %s, Energy = %s" % (alpha_list[index], np.min(E_list))

  if not found_step:
    print "Didn't find a lower energy step"
    sys.exit()

  return F_a, F_b, C_a, C_b, E0
#--------------------------------------

#--------------------------------------
def scf(C_a,C_b,P_a,P_b,F_a,F_b,Na,Nb):
  ''' Perform an SCF switching between standard Roothaan-Hall
      and Newton-Raphson steps '''

  Edelta, Pdelta, count, E0  = 1.0, 1.0, 0, 0
  # Start main SCF loop
  while Edelta > Econver and Pdelta > Pconver and count < 1000:
    count += 1
   
    # Save information from previous iteration
    old_E = E0
    P_old  = P_a + P_b   
   
    # Build Fock Matrix and Hartree energy
    E0, F_a, F_b = buildFock(dim,P_a,P_b,h,ERI)
  
    # Fock mo basis transform
    F_a_mo = np.dot(C_a.T, np.dot(F_a, C_a))
    F_b_mo = np.dot(C_b.T, np.dot(F_b, C_b))  
  
    # ---  Quadratic Convergence  ---
  
    # Bypass first loop
    if count < 5 or not doNR: 
  
      # Standard Roothaan-Hall
      F_a_oao = np.dot(X.T,np.dot(F_a, X))
      F_b_oao = np.dot(X.T,np.dot(F_b, X))
      eps_a, C_a_oao = eigh(F_a_oao)
      eps_b, C_b_oao = eigh(F_b_oao)
      C_a = np.dot(X, C_a_oao)
      C_b = np.dot(X, C_b_oao)
     
    else:
  
      # evaluate the gradient
      f_a   = getOVfvector(F_a_mo, Na, dim)
      f_b   = getOVfvector(F_b_mo, Nb, dim)
#     deriv = max(LA.norm(f_a),LA.norm(f_b))

      if doNR:
        # Do Newton-Raphson step
        F_a, F_b, C_a, C_b, E0 = NR_step(h,F_a_mo,F_b_mo,f_a,f_b,P_a,P_b,
                                         C_a,C_b,eps_a,eps_b,Na,Nb,dim,ERI,E0)
       
    print 'Step = %s\n\tEnergy = %.12f' % (count-1, E0)
  
    # Update Density Matrix
    # P_old = P_a + P_b   
    P_a = np.dot(C_a[:,0:Na],np.transpose(C_a[:,0:Na]))
    P_b = np.dot(C_b[:,0:Nb],np.transpose(C_b[:,0:Nb]))
    
    # Evaluate convergence
    Pdelta = deltaP((P_a + P_b), P_old)
    Edelta = np.abs(E0 - old_E)

  # Final diagnolization to keep F_mo diagonal 
  F_a_oao = np.dot(X.T,np.dot(F_a, X))
  F_b_oao = np.dot(X.T,np.dot(F_b, X))
  eps_a, C_a_oao = eigh(F_a_oao)
  eps_b, C_b_oao = eigh(F_b_oao)
  C_a = np.dot(X, C_a_oao)
  C_b = np.dot(X, C_b_oao)

  return E0, count, C_a, C_b, P_a, P_b, F_a, F_b
#--------------------------------------

#--------------------------------------
def check_curvature(C_a,C_b,F_a,F_b,ERI,Na,Nb):
  ''' Check whether the SCF solution is stable '''

  print "\nChecking curvature"

  eriMO      = ao2mo(ERI, [C_a, C_b], False)
  eps_a      = np.diag(np.dot(C_a.T, np.dot(F_a, C_a)) )
  eps_b      = np.diag(np.dot(C_b.T, np.dot(F_b, C_b)) )
  A, B       = responseAB_UHF(eriMO, [eps_a,eps_b],[Na,Nb])
  M          = np.bmat([[A, B],[B, A]])
  eps_M, v_M = eigh(M) 

  if eps_M[0] < 0:
    print "Lowest eigenvalue = ", eps_M[0]
    return False, v_M[:,0]
  else:
    return True, v_M[:,0]
#--------------------------------------

#--------------------------------------
def rotate_MOs(X,C_a,C_b,P_a,P_b,Na,Nb,E0):
  ''' Rotate MOs based on the Hessian eigenvector '''

  C_a_old  = C_a
  C_b_old  = C_b
  old_E    = E0
  D_a, D_b = X[:Na*(dim-Na)], X[Na*(dim-Na):]

  #TODO: need to determine this stepsize better
  alp = np.pi/4
  # rotate the MOs
  C_a, C_b, Ua, Ub = update_MOs(C_a_old,C_b_old,alp,D_a,D_b,Na,Nb)

  # check the current energy
  P_a = np.dot(C_a[:,0:Na],np.transpose(C_a[:,0:Na]))
  P_b = np.dot(C_b[:,0:Nb],np.transpose(C_b[:,0:Nb]))
  E0, F_a, F_b = buildFock(dim,P_a,P_b,h,ERI)

  return C_a, C_b, P_a, P_b, F_a, F_b
#--------------------------------------

#--------------------------------------
def print_SCF_results(name,basis,mult,E0,count):

  print '\n~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ '
  print '                   R e s u l t s            \n'
  print 'Molecule: ' + name 
  print 'Basis: ' + basis
  print 'Multiplicity: ' + str(mult)
  print 'E(SCF) = ' + str(E0) + ' a.u.'
  print '~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ '
#--------------------------------------

###############################
##       M   A   I   N       ##
###############################

start_time = time.time()

#### TEST SYSTEMS #### 

## Test Molecules
#mol, Nelec, name, basis, mult = 'H2_STO3G', 2, 'H2', 'STO-3G', 1
#mol, Nelec, name, basis, mult = 'HeHplus_STO3G', 2, 'HeH+', 'STO-3G', 1
#mol, Nelec, name, basis, mult = 'CO_STO3G', 14, 'CO', 'STO-3G', 1 
mol, Nelec, name, basis, mult = 'H2O_STO3G', 10, 'Water', 'STO-3G', 3
#mol, Nelec, name, basis, mult = 'Methanol_STO3G', 18, 'Methanol', 'STO-3G', 1
#mol, Nelec, name, basis, mult = 'Li_STO3G', 3, 'Lithium', 'STO-3G', 2
#mol, Nelec, name, basis, mult = 'O2_STO3G', 16, 'Oxygen', 'STO-3G', 3

######################

# Get integrals from files
Vnn, Vne, T, S, ERI = getIntegrals(mol)

# Number of basis functions dim, Nalpha, Nbeta
dim = len(S)
Na, Nb = calcAlphaBeta(Nelec, mult)

# Build Core Hamiltonian
h = T + Vne

# Set up initial Fock with core guess, and density at 0
eps_h, C_h = eigh(h)
F_a, F_b = h, h
P_a, P_b = np.zeros((dim,dim)), np.zeros((dim,dim))
C_a, C_b = np.zeros((dim,dim)), np.zeros((dim,dim))

# Form transformation matrix
s, Y = eigh(S)
s = np.diag(s**(-0.5))
X = np.dot(Y, np.dot(s, Y.T))

# Initialize variables
Pconver = 1.0e-8
Econver = 1.0e-8
P_old  = P_a + P_b
doNR   = True
doQN   = False

# Do stable optimization
stable, icheck, Maxcheck = False, 0, 3
while not stable and icheck < Maxcheck:

  # Do SCF
  E0, count, C_a, C_b, P_a, P_b, F_a, F_b = scf(C_a,C_b,P_a,P_b,F_a,F_b,Na,Nb)
  print_SCF_results(name,basis,mult,E0,count)

  # Check stability and rotate orbitals if necessary
  stable, Hvec = check_curvature(C_a,C_b,F_a,F_b,ERI,Na,Nb)
  if not stable:
    print "WAVE FUNCTION NOT STABLE\n"
    C_a, C_b, P_a, P_b, F_a, F_b = rotate_MOs(Hvec,C_a,C_b,P_a,P_b,Na,Nb,E0)
  else:
    print "STABLE WAVE FUNCTION FOUND\n"

  icheck += 1

## Get Spin Expectation Value
## <S^2> = 1/4 * [(Tr[Pmz*S])^2 + 2*Tr[Pmz*S*Pmz*S]]
#Pmz = P_a - P_b
#spin_expect = 1/4.0 * ((np.trace(np.dot(Pmz,S)))**2 + 2*np.trace(np.dot(Pmz,S).dot(Pmz).dot(S)))
#
#elapsed_time = time.time() - start_time

#### Print results ###
#print ''
#print '~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ '
#print '                   R e s u l t s            \n'
#print 'Molecule: ' + name 
#print 'Basis: ' + basis
#print 'Multiplicity: ' + str(mult)
#print 'E(SCF) = ' + str(E0) + ' a.u.'
#print '<S'+u'\xb2'+ '> =', spin_expect
#print 'SCF iterations: ' + str(count)
#print 'Elapsed time: ' + str(elapsed_time) + ' sec\n'
##print 'MO Coeffs (alpha) = \n' + np.array_str(C_a)
##print 'Fock Matrix (alpha) = \n' + np.array_str(F_a)
##print 'Fock Matrix (beta) = \n' + np.array_str(F_b)
##print 'Density Matrix (alpha) = \n' + np.array_str(P_a)
##print 'Density Matrix (beta) = \n' + np.array_str(P_b)
##print 'Orbital Energies (alpha) = \n' + str(eps_a) + '\n'
##print 'Orbital Energies (beta) = \n' + str(eps_b) + '\n'
#print '~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ '
#print ''
#
## Build the Hessian
##eriMO    = ao2mo(ERI, [C_a, C_b], False)
##eps_a    = np.diag(F_a_mo)
##eps_b    = np.diag(F_b_mo)
##A, B     = responseAB_UHF(eriMO, [eps_a,eps_b],[Na,Nb])
##M        = np.bmat([[A, B],[B, A]])
##eps_M, v_M = eigh(M) 
##print "Lowest eigenvalue = ", eps_M[0]
#
#
## Convert AO to MO orbital basis
##print '-------------------------'
##eriMO_u = ao2mo(ERI, [C_a, C_b], False)
##mp2(eriMO, eps, Nelec)
##print responseAB_UHF(eriMO, [eps_a, eps_b], Nelec)
##TDHF(eriMO_u, [eps_a, eps_b], [Na, Nb], False)
##print '-------------------------'
#
