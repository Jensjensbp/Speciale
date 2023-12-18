import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit
def solve_hh_backwards(par,z_trans,
    shock_beta,w,L,RealR_ex_post,delta,lambda_u_s,tau,UI_ratio_high,UI_duration,
    vbeg_a_plus,vbeg_a,vbeg_plus,vbeg,a,c,s_cond,s,u_UI,V,u_pol,ss=False):
    """ solve backwards with vbeg_a_plus from previous iteration """

    v_a = np.zeros((par.Nfix,par.Nz,par.Na))

    # a. consumption-saving
    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):
      
            # i. income
            if i_z == 0:
                u_UI_ = 0.0
                y = w*(1-tau) + L
                u_pol[i_fix,i_z,:] = 0
                u_UI_ = 0.0
            else:
                u_UI_ = np.fmax(np.fmin(UI_duration-(i_z-1),1.0),0.0)
                y = (u_UI_*UI_ratio_high + (1-u_UI_)*par.UI_ratio_low)*w*(1-tau) + L
                u_pol[i_fix,i_z,:] = 1
            
            u_UI[i_fix,i_z,:] = u_UI_

            # ii. EGM
            vbeg_plus_interp = np.zeros(par.Na)
            m = RealR_ex_post*par.a_grid + y
                        
            if par.beta_grid[i_fix] < par.beta_HtM: # HtM
       
                a[i_fix,i_z,:] = 0.0
                c[i_fix,i_z,:] = m
                V[i_fix,i_z,:] = c[i_fix,i_z]**(1.0-par.sigma) / (1.0-par.sigma)
 
            elif ss:
 
                c[i_fix,i_z,:] = 0.9*m
                a[i_fix,i_z,:] = m-c[i_fix,i_z,:]
                V[i_fix,i_z,:] = c[i_fix,i_z]**(1.0-par.sigma) / (1.0-par.sigma)
               
            else:
                
                # o. EGM
                vbeg_plus_disc = vbeg_plus[i_fix,i_z]
                
                c_endo = (shock_beta*par.beta_grid[i_fix]*vbeg_a_plus[i_fix,i_z])**(-1/par.sigma)
                m_endo = c_endo + par.a_grid
           
                # oo. interpolation to fixed grid
                interp_1d_vec(m_endo,par.a_grid,m,a[i_fix,i_z])
                interp_1d_vec(m_endo,vbeg_plus_disc,m,vbeg_plus_interp)
               
                # ooo. enforce borrowing constraint
                a[i_fix,i_z,:] = np.fmax(a[i_fix,i_z,:],0.0)
 
                # oooo. implied consumption
                c[i_fix,i_z] = m - a[i_fix,i_z]
                V[i_fix,i_z,:] = c[i_fix,i_z]**(1.0-par.sigma) / (1.0-par.sigma) + shock_beta*par.beta_grid[i_fix]*vbeg_plus_interp
           
            # iii. envelope
            v_a[i_fix,i_z] = RealR_ex_post*c[i_fix,i_z]**(-par.sigma)            
    
    # b. searching
    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):

            if par.exo_search:
                
                s_cond[i_fix,i_z,:] = 1.0

            else:

                if i_z == 0:
                    s_cond[i_fix,i_z] = ((1/par.varphi)*lambda_u_s*np.fmax((V[i_fix,0]-V[i_fix,i_z+1]),0))**(1/par.nu)
                elif 0 < i_z < par.Nz - 1:
                    s_cond[i_fix,i_z] = ((1/par.varphi)*lambda_u_s*np.fmax((V[i_fix,0]-V[i_fix,i_z+1]),0))**(1/par.nu)
                else:
                    s_cond[i_fix,i_z] = ((1/par.varphi)*lambda_u_s*np.fmax((V[i_fix,0]-V[i_fix,i_z]),0))**(1/par.nu)                
                
                # bounding
                s_cond[i_fix,i_z,:] = np.fmin(s_cond[i_fix,i_z,:],0.99*lambda_u_s**-1)
                s_cond[i_fix,i_z,:] = np.fmax(s_cond[i_fix,i_z,:],0.01*lambda_u_s**-1) 
                
            if i_z == 0:
                s[i_fix,i_z] = delta*s_cond[i_fix,i_z]
            else:
                s[i_fix,i_z] = s_cond[i_fix,i_z]
                
    # c. transition            
    fill_z_trans(par,z_trans,delta,s_cond,lambda_u_s)            
        
    # d. beginning of period
    for i_fix in range(par.Nfix):
        for i_z_lag in range(par.Nz):

            # i. search
            u_search = -par.varphi * (s_cond[i_fix,i_z_lag])**(1+par.nu) / (1+par.nu)
            if i_z_lag == 0:
                vbeg[i_fix,i_z_lag] = delta*u_search
            else:
                vbeg[i_fix,i_z_lag] = u_search
            
            # ii. consumption-saving
            vbeg_a[i_fix,i_z_lag] = 0.0
            
            for i_z in range(par.Nz):
                vbeg[i_fix,i_z_lag] += z_trans[i_fix,:,i_z_lag,i_z]*V[i_fix,i_z] 
                vbeg_a[i_fix,i_z_lag] += z_trans[i_fix,:,i_z_lag,i_z]*v_a[i_fix,i_z] 
                    
################
# fill_z_trans #
################

@nb.njit(fastmath=True)
def fill_z_trans(par,z_trans,delta,s_cond,lambda_u_s):
    """ transition matrix for z """

    for i_fix in nb.prange(par.Nfix):
        for i_z in nb.prange(par.Nz):
            for i_a in nb.prange(par.Na):
                for i_z_plus in nb.prange(par.Nz):

                    if i_z == 0:
                            
                        if i_z_plus == 0:
                            u_trans = (1.0-delta) + delta*(s_cond[i_fix,i_z,i_a]*lambda_u_s)
                        elif i_z_plus == 1:
                            u_trans = delta*(1.0-s_cond[i_fix,i_z,i_a]*lambda_u_s)
                        else:
                            u_trans = 0.0
                                                                 
                    else:

                        if i_z_plus == 0:
                            u_trans = s_cond[i_fix,i_z,i_a]*lambda_u_s                            
                        elif (i_z_plus == i_z+1) or (i_z_plus == i_z == par.Nu):
                            u_trans = 1.0-s_cond[i_fix,i_z,i_a]*lambda_u_s                            
                        else:
                            u_trans = 0.0
                            
                    z_trans[i_fix,i_a,i_z,i_z_plus] = u_trans