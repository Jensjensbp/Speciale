# find steady state

import time
import numpy as np

from EconModel import jit

from consav.grids import equilogspace
from consav.markov import tauchen, find_ergodic
from consav.misc import elapsed
from scipy import optimize

import household_problem

def prepare_hh_ss(model):
    """ prepare the household block for finding the steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############

    par.a_grid[:] = equilogspace(0.0,par.a_max,par.Na)
    
    par.beta_shares[:] = np.array([par.HtM_share,1-par.HtM_share-par.PIH_share,par.PIH_share])
    
    ###########################
    # 2. initial distribution #
    ###########################
    
    for i_fix in range(par.Nfix):
        ss.Dbeg[i_fix,:,:] = 0.0      
        ss.Dbeg[i_fix,0,0] = par.beta_shares[i_fix]
    
    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    model.set_hh_initial_guess()
    
def find_ss(model,do_print=False):

    par = model.par
    ss = model.ss 
    
    ###### PRE HOUSEHOLD PROBLEM ######
    
    # Step 1:    
    ss.shock_TFP = 1.0
    ss.shock_beta = 1.0
    ss.Pi = 1.0
    
    ss.w = par.w_ss
    ss.delta = par.delta_ss
    ss.lambda_u_s = par.lambda_u_s_ss
    ss.qB = par.qB_share_ss*ss.w
    ss.RealR_ex_post = ss.RealR = ss.R = par.R_ss**(1/12)
    
    # Step 2:
    ss.q = 1/(ss.RealR-par.delta_q)
    ss.B = ss.qB/ss.q    
    
    # Step 3:
    ss.UI_ratio_high = par.UI_ratio_high
    ss.UI_duration = par.UI_duration
    ss.tau = par.tau       
    
    ###### HOUSEHOLD PROBLEM ######
    
    # Step 1 - 6 in obj defined below
    optimize.brentq(obj, -0.05, 0.1, args=(model,do_print,), xtol = 1e-8, rtol = 1e-8)
    
    # Misc: final evaluation
    ss.L = par.L[np.argmin(np.abs(par.diff))]
                
    model.solve_hh_ss()
    model.simulate_hh_ss()
    
    A_hh_fix = [np.sum(ss.a[i_fix]*ss.D[i_fix])/np.sum(ss.D[i_fix]) for i_fix in range(par.Nfix)]
        
    A_hh_no_PIH = par.HtM_share*A_hh_fix[0] + (1-par.HtM_share)*A_hh_fix[1]
    dA_hh_dPIH_dmid = A_hh_fix[2]-A_hh_fix[1]
    
    par.PIH_share = (ss.qB-A_hh_no_PIH)/dA_hh_dPIH_dmid
    
    par.beta_shares[:] = np.array([par.HtM_share,1-par.HtM_share-par.PIH_share,par.PIH_share])
    
    for i_fix in range(par.Nfix):
        ss.Dbeg[i_fix,:,:] = 0.0      
        ss.Dbeg[i_fix,0,0] = par.beta_shares[i_fix]    
    
    model.simulate_hh_ss()        
    
    # Misc: Reset parameters for next time
    par.L = np.zeros(30)
    par.diff = np.zeros(30) + 100.0
        
    par.it = 0
    
    ###### POST HOUSEHOLD PROBLEM ######
    
    # Step 1: 
    ss.S = ss.S_hh = np.sum(ss.Dbeg*ss.s)
    ss.ut = (1-ss.u)*ss.delta + ss.u
    
    ss.lambda_u = 1 - ss.u/ss.ut
    
    # Step 2:
    ss.theta = par.theta_ss
    par.A = ss.lambda_u_s/((ss.theta)**(1-par.alpha))
    ss.lambda_v = par.A*ss.theta**(-par.alpha)
    
    ss.vt = ss.S*ss.theta
    ss.v = (1-ss.lambda_v)*ss.vt
    
    ss.entry = ss.vt - (1-ss.delta)*ss.v
    
    # Step 3:
    ss.px = (par.epsilon_p-1)/par.epsilon_p
    ss.M = ss.px*ss.shock_TFP-ss.w
    par.p = par.p_fac*ss.delta

    Vj_Upsilon = (ss.delta/par.p)**(-1/par.psi)

    _nom = par.p*Vj_Upsilon**(-1)
    if np.abs(par.psi-1.0) < 1e-8:
        _nom *= np.log(Vj_Upsilon)
    else:
        _nom *= par.psi/(par.psi-1)*(1-Vj_Upsilon**(1-par.psi))

    _denom = (1-par.p*Vj_Upsilon**(-par.psi))

    mu_Vj = _nom/_denom

    ss.Vj = ss.M/(1+par.beta_firm*mu_Vj-par.beta_firm*(1-ss.delta))
        
    par.Upsilon = ss.Vj/Vj_Upsilon
    ss.mu = mu_Vj*ss.Vj

    _fac = 1-par.beta_firm*(1-ss.lambda_v)*(1-ss.delta)
        
    ss.Vv = par.kappa_0
    
    par.kappa = ss.lambda_v*ss.Vj - _fac*ss.Vv


def obj(L_guess, model, do_print):
        
    par = model.par
    ss = model.ss
    
    # Step 1:
    ss.L = L_guess
    
    #Step 2:
    model.solve_hh_ss()
    model.simulate_hh_ss()
    
    # Step 3:
    A_hh_fix = [np.sum(ss.a[i_fix]*ss.D[i_fix])/np.sum(ss.D[i_fix]) for i_fix in range(par.Nfix)]
        
    A_hh_no_PIH = par.HtM_share*A_hh_fix[0] + (1-par.HtM_share)*A_hh_fix[1]
    dA_hh_dPIH_dmid = A_hh_fix[2]-A_hh_fix[1]

    par.PIH_share = (ss.qB-A_hh_no_PIH)/dA_hh_dPIH_dmid
    
    par.beta_shares[:] = np.array([par.HtM_share,1-par.HtM_share-par.PIH_share,par.PIH_share])
    
    for i_fix in range(par.Nfix):
        ss.Dbeg[i_fix,:,:] = 0.0      
        ss.Dbeg[i_fix,0,0] = par.beta_shares[i_fix]    
    
    # step 4:
    model.simulate_hh_ss()
    
    # step 5:
    ss.u = ss.U_POL_hh
    ss.guess_U_UI = ss.U_UI_hh
    
    # step 6
    L_endo = ss.qB + ss.tau*ss.w*(1-ss.u) - (1+par.delta_q*ss.q)*ss.B - ss.UI_ratio_high*ss.w*(1-ss.tau)*ss.U_UI_hh - par.UI_ratio_low*ss.w*(1-ss.tau)*(ss.u - ss.U_UI_hh)   
    
    if do_print: 
        print(f'Guess on L of: {L_guess} causes diff L_guess - L_endo of: {ss.L - L_endo}')
    
    # Misc: save difference to secure lowest possibel discrepancy
    par.L[par.it] = ss.L
    
    par.diff[par.it] = ss.L - L_endo 
    
    par.it += 1
        
    return ss.L - L_endo