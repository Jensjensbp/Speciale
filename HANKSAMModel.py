import pickle
from copy import deepcopy
import time
import numpy as np
import statsmodels.api as sm

import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from EconModel import EconModelClass
from GEModelTools import GEModelClass
from consav.misc import elapsed

import household_problem
import steady_state
import blocks

# from root_finding import brentq

from scipy import optimize


class HANKSAMModelClass(EconModelClass,GEModelClass):    

    #########
    # setup #
    #########

    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = ['par','ini','ss','path','sim']
        
        # b. household
        self.grids_hh = ['a']
        self.pols_hh = ['a','s','V']
        self.inputs_hh = ['shock_beta','w','L','RealR_ex_post','delta','lambda_u_s','tau','UI_ratio_high','UI_duration']
        self.inputs_hh_z = ['delta','lambda_u_s']
        self.outputs_hh = ['a','c','s','s_cond','u_UI','V','u_pol']
        self.intertemps_hh = ['vbeg_a','vbeg']

        # c. GE
        self.shocks = ['shock_TFP','shock_beta']
        self.unknowns = ['px','Vj','Vv','Pi','vt','ut','S','guess_U_UI']
        self.targets = ['errors_Vj','errors_Vv','errors_Pi','errors_assets','errors_vt','errors_ut','errors_search', 'errors_U_UI']

        # functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards
        self.blocks = [
            'blocks.production',
            'blocks.labor_market',
            'blocks.entry',
            'blocks.price_setters',
            'blocks.central_bank',
            'blocks.government',
            'hh',
            'blocks.market_clearing']

        # misc
        self.other_attrs = ['data','moms','datamoms']

    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nfix = 3 # Number of households types

        # a. macros
        par.UI_system = 'fixed'
        par.financing = 'constant debt'
        
        par.wage_setting = 'fixed'
        par.free_entry = False
        par.exo_sep = False

        par.only_SAM = False
        par.exo_search = False

        # b. model parameters

        # preferences
        par.beta_grid = np.array([0.00,0.96**(1/12),0.98**(1/12)]) #Grid over betas
        
        par.HtM_share = 0.1456499378120454 # low beta share
        par.PIH_share = 0.15 # high beta share
        
        par.beta_firm = par.beta_grid[-1] # firm discount factor
        par.beta_HtM = 0.90 # cut-off for hands-to-mouth behavior

        par.sigma = 2.0 # CRRA coefficient
        par.nu = 2.0
        par.varphi = 0.6

        # matching and bargaining
        par.delta_ss = 0.02748598211864653 # separation rate (from data)
        par.lambda_u_s_ss = 0.3820788784109759 # effective job-finding rate
        par.A = np.nan # matching efficiency, determined endogenously
        par.theta_ss = 0.60 # tightness in ss
        par.alpha = 0.60 # matching elasticity

        # intermediary goods firms
        par.w_ss = 0.70 # wage in steady state
        par.kappa = np.nan # flow vacancy cost, determined endogenously
        par.kappa_0 = 0.1 # fixed vacancy cost
        par.psi = 1.0 # separation elasticity
        par.xi = 0.05 # entry elasticity
        par.p_fac = 1.20 # factor for maximum increase in separation rate
        par.p = np.nan # maximum separation rate, determined endogenously
        par.Upsilon = np.nan # Vj at maximum separation rate, determined endogenously

        # final goods firms
        par.epsilon_p = 6.0 # price elasticity      
        par.phi = 600.0 # Rotemberg cost
        
        # monetary policy
        par.rho_R = 0.0 # inertia
        par.delta_pi = 1.5 # inflation aggressiveness
        par.R_ss = 1.02 # Yearly interest rate

        # government
        par.qB_share_ss = 0.8 # government bonds (share of wage)
        
        par.Nu = 13 # number of u states
        par.UI_duration = np.nan # UI duration
        par.UI_ratio_high = np.nan # high UI ratio (rel. to w) *before* exhausation (in steady state)
        par.UI_ratio_low = np.nan # low UI ratio (rel. to w) *after* exhausation
        par.tau = 0.1363
        
        par.ela_phi_rath = 0.2

        par.delta_q = 1-1/60 # maturity of government bonds

        # b. shocks
        par.rho_shock_TFP = 0.965 # persitence
        par.jump_shock_TFP = -0.007 # jump

        par.rho_shock_beta = 0.965 # persistence
        par.jump_shock_beta = 0.01 # jump

        # c. household problem
        par.Na = 100 # number of asset grid points
        par.a_max = 50 # max level of assets
        
        # d. calibration targets 
        par.lambda_u_ss = 0.3056686310719316
        par.C_drop_ss_target = - 6.0

        # d. misc
        par.T = 500 # length of path        
        
        par.max_iter_solve = 100_000 # maximum number of iterations when solving
        par.max_iter_simulate = 100_000 # maximum number of iterations when simulating
        par.max_iter_broyden = 50 # maximum number of iteration when solving eq. system
        
        par.tol_solve = 1e-12 # tolerance when solving
        par.tol_simulate = 1e-12 # tolerance when simulating
        par.tol_broyden = 1e-8 # tolerance when solving eq. system
        par.tol_R = 1e-12 # tolerance when finding RealR and S_hh for ss
        par.tol_calib = 1e-5 # tolerance when calibrating (C_drop and var_u)
        
        par.py_hh = False
        par.py_blocks = False
        par.full_z_trans = True
        
        # For root-finding:
        par.L = np.zeros(30)
        par.diff = np.zeros(30) + 100.0
        
        par.it = 0        
        
    def allocate(self):
        """ allocate model """
        
        par = self.par

        par.beta_shares = np.zeros(par.Nfix)

        par.Nz = (par.Nu+1)
        par.z_grid = np.ones(par.Nz)
        
        self.allocate_GE() # should always be called here

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss

    ##############
    # more setup #
    ##############


    def set_macros(self,free_entry=None,wage_setting=None):
        """ set macros """

        par = self.par

        # baseline
        unknowns = ['px','Vj','Vv','Pi','ut','vt','U_UI_hh_guess']
        targets = ['errors_Vj','errors_Vv','errors_Pi','errors_assets','errors_ut','errors_vt','errors_U_UI']

        if not free_entry is None: par.free_entry = free_entry
        if not wage_setting is None: par.wage_setting = wage_setting 

        # a. free entry
        if par.free_entry:
            unknowns = ['px','entry','Vj','Pi','ut','vt','U_UI_hh_guess']

        # b. wage setting
        if par.wage_setting == 'fixed':
            pass
        elif par.wage_setting == 'rule':
            unknowns += ['w']
            targets += ['errors_WageRule']
        else:
            raise NotImplementedError

        self.update_aggregate_settings(unknowns=unknowns,targets=targets)

    ###############
    # calibration #
    ###############

    def obj_calib_C_drop(self,s):
        """ objective when calibrating w """

        par = self.par
        ss = self.ss

        self.par.HtM_share = s        
        
        self.find_ss()
        
        #average consumption of employed
        Ce = (np.sum(ss.c[0,0,:]*ss.D[0,0,:]) + np.sum(ss.c[1,0,:]*ss.D[1,0,:])+ np.sum(ss.c[2,0,:]*ss.D[2,0,:]) )/ ( np.sum(ss.D[0,0,:]) + np.sum(ss.D[1,0,:])+ np.sum(ss.D[2,0,:]) ) 
            
        # average consumption of unemployed in first period. 
        Cu = (np.sum(ss.c[0,1,:]*ss.D[0,1,:]) + np.sum(ss.c[1,1,:]*ss.D[1,1,:])+ np.sum(ss.c[2,1,:]*ss.D[2,1,:]) )/ ( np.sum(ss.D[0,1,:]) + np.sum(ss.D[1,1,:])+ np.sum(ss.D[2,1,:]) ) 
        
        diff = (Cu/Ce-1)*100 - par.C_drop_ss_target
        
        print(f'guess on HtM_share of {s} causes diff drop - drop_target of {diff}')
        
        return diff

    def calibrate_to_C_drop(self,s_min=0.01,s_max=0.3,do_print=False):
        """ calibrate beta_low_share to fit chosen consumption drop"""

        t0 = time.time()

        par = self.par
        
        optimize.brentq(self.obj_calib_C_drop,s_min,s_max,xtol=par.tol_calib,rtol=par.tol_calib)
        
        if do_print: print(f'calibration done in {elapsed(t0)}')
        
    def obj_calib_lambda_u(self,leff):
        """ objective when calibrating w """

        par = self.par
        ss = self.ss

        self.par.lambda_u_s_ss = leff         
        
        self.find_ss()
        
        diff = ss.lambda_u - par.lambda_u_ss
        
        print(f'guess on lambda_eff of {leff} causes diff drop - drop_target of {diff}')
        
        return diff

    def calibrate_to_lambda_u(self,lambda_u_min=0.2,lambda_u_max=0.4,do_print=False):
        """ calibrate beta_low_share to fit chosen consumption drop"""

        t0 = time.time()

        par = self.par
        
        optimize.brentq(self.obj_calib_lambda_u,lambda_u_min,lambda_u_max,xtol=par.tol_calib,rtol=par.tol_calib)
        
        if do_print: print(f'calibration done in {elapsed(t0)}')