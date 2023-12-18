import numpy as np
import numba as nb

from GEModelTools import lag, lead

@nb.njit
def delta_func(Vj,par,ss):    
    """ separations """

    return par.p*(np.fmax(Vj/par.Upsilon,1))**(-par.psi)

@nb.njit
def mu_func(Vj,par):
    """ continuation costs """

    if np.abs(par.psi-1.0) < 1e-8:
        _fac = np.log(np.fmax(Vj/par.Upsilon,1.0))
    else:
        _fac = par.psi/(par.psi-1)*(1.0-(np.fmax(Vj/par.Upsilon,1.0))**(1-par.psi))

    _nom = par.p*par.Upsilon
    _denom = 1.0-par.p*(np.fmax(Vj/par.Upsilon,1.0))**(-par.psi)

    return _fac*_nom/_denom   

@nb.njit
def production(par,ini,ss,shock_TFP,delta,w,px,Vj,Vv,entry,mu,M,errors_Vj):
    w[:] = ss.w

    M[:] = shock_TFP*px-w
        
    delta[:] = delta_func(Vj,par,ss)
    mu[:] = mu_func(Vj,par)

    Vj_plus = lead(Vj,ss.Vj)
    delta_plus = lead(delta,ss.delta)
    mu_plus = lead(mu,ss.mu)

    cont_Vj = (1-delta_plus)*par.beta_firm*Vj_plus - par.beta_firm*mu_plus
    errors_Vj[:] = Vj-(M+cont_Vj)

    entry[:] = ss.entry*(np.fmax(Vv[:]/ss.Vv,0.0))**par.xi

@nb.njit
def labor_market(par,ini,ss,vt,S,theta,delta,lambda_v,entry,lambda_u,lambda_u_s,v,u,ut,errors_vt,errors_ut):

    theta[:] = vt / S

    lambda_v[:] = par.A*theta**(-par.alpha)
    lambda_u_s[:] = par.A*theta**(1-par.alpha)
                    
    u[:] = 1-(S*lambda_u_s+(1-ut))
    v[:] = (1-lambda_v)*vt

    u_lag = lag(ini.u,u)
    v_lag = lag(ini.v,v)

    errors_vt[:] = vt - ((1-ss.delta)*v_lag + entry)
    errors_ut[:] = ut - (u_lag + delta*(1-u_lag))
    
    lambda_u[:] = 1 - u/ut
    
@nb.njit
def entry(par,ini,ss,lambda_v,Vj,Vv,errors_Vv):

    LHS = Vv
    Vv_plus = lead(Vv,ss.Vv)
        
    RHS = -par.kappa + lambda_v*Vj + (1-lambda_v)*(1-ss.delta)*par.beta_firm*Vv_plus

    errors_Vv[:] = LHS-RHS

@nb.njit
def price_setters(par,ini,ss,shock_TFP,u,px,Pi,errors_Pi):
    
    LHS = 1-par.epsilon_p + par.epsilon_p*px

    Pi_plus = lead(Pi,ss.Pi)        
    shock_TFP_plus = lead(shock_TFP,ss.shock_TFP)
    u_plus = lead(u,ss.u)

    RHS = par.phi*(Pi-ss.Pi)*Pi - par.beta_firm*par.phi*((Pi_plus-ss.Pi)*Pi_plus*(shock_TFP_plus*(1-u_plus))/(shock_TFP*(1-u)))
    errors_Pi[:] = LHS-RHS

@nb.njit
def central_bank(par,ini,ss,Pi,R,RealR,q,RealR_ex_post):

    for t in range(par.T):
        R_lag = ss.R if t == 0 else R[t-1]
        R[t] = ss.R*(R_lag/ss.R)**(par.rho_R)*(Pi[t]/ss.Pi)**(par.delta_pi*(1-par.rho_R))
            
        if t < par.T-1:
            RealR[t] = R[t]/Pi[t+1]
        else:
            RealR[t] = R[t]/ss.Pi

    # iv. arbitrage
    for k in range(par.T):
        t = par.T-1-k
        q_plus = q[t+1] if t < par.T-1 else ini.q
        q[t] = (1+par.delta_q*q_plus)/RealR[t]
                
    q_lag = lag(ini.q,q)
        
    RealR_ex_post[:] = (1+par.delta_q*q)/q_lag

@nb.njit
def government(par,ini,ss,w,u,q,tau,B,qB,UI_ratio_high,guess_U_UI,UI_duration,L):
    
    if par.UI_system == 'fixed':
        UI_ratio_high[:] = par.UI_ratio_high
        UI_duration[:] = par.UI_duration
            
    elif par.UI_system == 'varying':
        UI_ratio_high[:] = ss.UI_ratio_high*(u/ss.u)**par.ela_phi_rath
        UI_duration[:] = par.UI_duration
    else:
        raise ValueError(f'UI system not in allowed set')
            
    tau[:] = ini.tau
                
    if par.financing == 'constant debt':
        B[:] = ini.B    
        L[:] = q*B + tau*w*(1.0-u) - (1.0+par.delta_q*q)*B - UI_ratio_high*w*(1.0-tau)*guess_U_UI - par.UI_ratio_low*w*(1-tau) * (u - guess_U_UI)   
        
    elif par.financing == 'deficit financing':
        for t in range(par.T):
            B_lag = B[t-1] if t > 0 else ini.B

            if t < 40:
                L[t] = ini.L
                B[t] = ((1+par.delta_q*q[t])*B_lag + UI_ratio_high[t]*w[t]*(1-tau[t])*guess_U_UI[t] + par.UI_ratio_low*w[t]*(1-tau[t])*(u[t] - guess_U_UI[t]) +  L[t] - tau[t]*w[t]*(1-u[t]))/q[t]
            elif 39 < t < 50:
                B[t] = B[39] - (t-39)*(B[39] - ini.B)/10
                L[t] = q[t]*B[t] + tau[t]*w[t]*(1-u[t]) - (1+par.delta_q*q[t])*B_lag - UI_ratio_high[t]*w[t]*(1-tau[t])*guess_U_UI[t] - par.UI_ratio_low*w[t]*(1-tau[t])*(u[t] - guess_U_UI[t])
            else:
                B[t] = ini.B
                L[t] = q[t]*B[t] + tau[t]*w[t]*(1-u[t]) - (1+par.delta_q*q[t])*B_lag - UI_ratio_high[t]*w[t]*(1-tau[t])*guess_U_UI[t] - par.UI_ratio_low*w[t]*(1-tau[t])*(u[t] - guess_U_UI[t]) 
                
    else:
        raise ValueError(f'Financing not in allowed set')
        
    qB[:] = q*B


@nb.njit
def market_clearing(par,ini,ss,qB,A_hh,U_POL_hh,u,guess_U_UI,U_UI_hh,errors_assets,errors_search,errors_U_UI):
    
    errors_assets[:] = qB - A_hh
    errors_search[:] = U_POL_hh - u
    
    # ii. share on unemployment benefits
    errors_U_UI[:] = guess_U_UI - U_UI_hh