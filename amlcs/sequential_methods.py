import os
import numpy as np
import sys
import scipy.sparse as spa
import pandas as pd
import time
from netCDF4 import Dataset
from commons_utils import compute_modified_Cholesky_decomposition
import torch
from Rev_SDE import REVERSE_SDE


##########################################################################################
##########################################################################################
##########################################################################################
# General class - sequential ensemble data assimilation (Original, Unchanged)
##########################################################################################
##########################################################################################
##########################################################################################
class ensemble_DA:

    XB = None; 
    XA = None;
    XB_map = None;
    XA_map = None;
    Nens = None;
    nm = None;
    gr = None;
    XB_s = [];
    Ys = None;
    infla = None;
    
    def __init__(self, nm, infla, Nens):
        self.Nens = Nens;
        self.nm = nm;
        self.infla = infla;
        self.test = 1;
        #self.load_background_ensemble();
        
    

    def load_background_ensemble(self):
        nm = self.nm;
        nm.load_ensemble();
        self.XB = nm.X.copy();
    
    
    def get_ensemble_variable(self, v):
        Nens = self.Nens;
        v_info = v[0];
        v_reso = v[1];
        var_index = v_info[0];
        var_level = v_info[1];
        lat, lon = v_reso[0], v_reso[1];
        n = lat*lon;
        XB_v = np.zeros((n,Nens));
        nm = self.nm;
        for e in range(0, Nens):
            if 'PSG' in nm.var_names[var_index]:
                xb_e_v = self.XB[e][var_index][:,:].reshape((n,));
            else:
                xb_e_v = self.XB[e][var_index][var_level,:,:].reshape((n,));
            XB_v[:,e] = xb_e_v;
        return XB_v;
    
    
    def get_ensemble_block(self, msk_cor):
        n_vars = len(msk_cor);
        XB_block = np.empty(shape=(0,self.Nens));
        if self.test == 1: print('* ENDJ - Working on block {0}'.format(msk_cor));
        for v in msk_cor:
            XB_v = self.get_ensemble_variable(v);
            XB_block = np.concatenate((XB_block, XB_v), axis=0);
        return XB_block;
        
    
    def prepare_background(self):
        pass;
        
    def prepare_analysis(self, ob, k, args = None):
        pass;
    
    def perform_assimilation(self, ob):
        pass;
    
    def map_vector_states(self):
        self.XA = [];
        Nens = self.Nens;
        for e in range(0, Nens):
            xa_e = self.nm.map_vector_state(self.XA_map, e);
            nc_f = f'{self.nm.ensemble_0}ens_{e}/ensemble_member.nc';
            self.nm.map_state_netcdf(xa_e, nc_f); 
            self.XA.append(xa_e);
    
    def perform_forecast(self):
        self.nm.forecast_ensemble(self.Nens);
        
    def check_time_store(self, k, list_k):
        if k in list_k:
            nm = self.nm;
            xb_k = nm.compute_snapshot_mean(self.XB);
            fn_xb = f'{nm.time_snapshots}xb{k}.nc';
            xa_k = nm.compute_snapshot_mean(self.XA);
            fn_xa = f'{nm.time_snapshots}xa{k}.nc';
            nm.map_state_netcdf(xb_k, fn_xb);
            nm.map_state_netcdf(xa_k, fn_xa);
            
    
    def covariance_inflation(self, XA):  
        xa = XA.mean(axis=1).reshape(-1,1);
        DX = XA-xa;
        XA = xa + self.infla * DX;  
        return XA;
        
    def clear_all(self):
        self.nm.clear_all_folders();

##########################################################################################
# EnKF_MC_obs (Original, Unchanged)
##########################################################################################
class EnKF_MC_obs(ensemble_DA):
            
    
    def prepare_background(self):
        mask_cor = self.nm.mask_cor;
        var_names = self.nm.var_names;
        Nens = self.Nens;
        self.XB_map = [];
        gr = self.nm.gs;
        for msk_cor, pre_info in zip(mask_cor, gr.lpr):
            XB_block = self.get_ensemble_block(msk_cor);
            xb_block = XB_block.mean(axis=1).reshape(-1,1);
            DX_block = XB_block-xb_block;
            Binv_sqrt_block = compute_modified_Cholesky_decomposition(DX_block, pre_info, thr=0.15);

            self.XB_map.append({'XB_b':XB_block, 'xb_b':xb_block, 'Binv_s_b':Binv_sqrt_block});
    
    def prepare_analysis(self, ob, k, args = None):
        self.Ys = [];
        Nens = self.Nens;
        mask_cor = self.nm.mask_cor;
        N_blocks = len(mask_cor);
        for block in range(0, N_blocks):
            Ys_block = ob.get_perturbed_observations(block, k, Nens);  
            self.Ys.append(Ys_block); 
    
    def perform_assimilation(self, ob):
        self.XA_map = [];
        for XB_info, Ys_block, R_info, H_block in zip(self.XB_map, self.Ys, ob.obs_R_sparse, ob.obs_H_sparse):
            XB_block = XB_info['XB_b'];
            R_block  = R_info['R'];
            Binv_sqrt_block = XB_info['Binv_s_b'];
            XA_block = self.perform_assimilation_block(XB_block, Binv_sqrt_block, H_block, R_block, Ys_block);
            XA_block = self.covariance_inflation(XA_block);
            self.XA_map.append(XA_block);
        self.map_vector_states(); #Update ensemble folders
    
    def perform_assimilation_block(self, XB, Binv_sqrt, H, R, Ys):
    
        Ds = Ys - H @ XB;
        
        H_spar = H.toarray();
        
        P = spa.linalg.spsolve_triangular(Binv_sqrt, H_spar.T, lower=False, check_finite=False);
        
        Inno = R + P.T @ P;

        Q_temp = P @ spa.linalg.spsolve(Inno, Ds);
        
        DXa = spa.linalg.spsolve_triangular(Binv_sqrt.T, Q_temp, lower=True, check_finite=False);

        XA = XB + DXa;  
        
        return XA;
        
        
##########################################################################################
# LETKF (Original, Unchanged)
##########################################################################################
class LETKF(ensemble_DA):
            
    
    def prepare_background(self):
        mask_cor = self.nm.mask_cor;
        var_names = self.nm.var_names;
        Nens = self.Nens;
        self.XB_map = [];
        gr = self.nm.gs;
        for msk_cor, lbo_info in zip(mask_cor, gr.lbo):
            XB_block = self.get_ensemble_block(msk_cor);
            xb_block = XB_block.mean(axis=1).reshape(-1,1);
            DX_block = XB_block-xb_block;
            self.XB_map.append({'XB_b':XB_block, 'xb_b':xb_block, 'DX_b':DX_block, 'lbo_b':lbo_info});


    def prepare_analysis(self, ob, k, args = None):
        self.y = [];
        Nens = self.Nens;
        mask_cor = self.nm.mask_cor;
        N_blocks = len(mask_cor);
        for block in range(0, N_blocks):
            self.y.append(ob.y_obs[k][block]); 
        self.nm.gs.compute_local_boxobs(ob.obs_H_sparse);
    
    def perform_assimilation(self, ob):
        self.XA_map = [];
        for XB_info, y_block, R_info, H_block, lobs_block in zip(self.XB_map, self.y, ob.obs_R_sparse, ob.obs_H_sparse, self.nm.gs.lbo_obs):
            XB_block = XB_info['XB_b'];
            xb_block = XB_info['xb_b'];
            DX_block = XB_info['DX_b'];
            lbo_block = XB_info['lbo_b'];
            Ri_block  = R_info['Ri'];
            XA_block = self.perform_assimilation_block(XB_block, xb_block, DX_block, H_block, Ri_block, y_block, lbo_block, lobs_block);
            self.XA_map.append(XA_block);
        self.map_vector_states(); #Update ensemble folders
        
    
    def perform_assimilation_local_box(self, XB, xb, DX, H, Ri, y):
    
        n, Nens = XB.shape;
        d = y - H @ xb;
        Yb = H @ DX;
        
        Pa_Nens = (Nens-1)*np.eye(Nens) + Yb.T @ ( Ri @ Yb );
        Q_temp = Yb.T @ (Ri @ d);
        
        U, S, _ = np.linalg.svd(Pa_Nens, full_matrices=False);
        
        Pa_sqrt = U @ ( np.diag(np.sqrt(Nens/S)) @ U.T );
        Pa_invs = U @ ( np.diag(1/S) @ U.T );
        
        wa = Pa_invs @ Q_temp;
        
        xa = xb + DX @ wa;
        
        XA = xa + DX @ Pa_sqrt;
        
        return XA;
        
        
    
    def perform_assimilation_block(self, XB, xb, DX, H, Ri, y, lbo_info, lobs):

        
        n, Nens = XB.shape;
        
        lbo, nlbo = lbo_info; #local boxes information (indexes, number of components in all boxes)

        y_model = H.T @ y;
        
        
        Ri_space = H.T @ (Ri @ H);
        Ri_space = Ri_space.toarray();
        
        XA = np.zeros((n, Nens));
        
        
        for i in range(0, n):
            #local box for model component i
            lbo_i = np.array(lbo[i]).astype('int32');
            gp_i, = np.where(lbo_i==i);
            xb_i = xb[lbo_i];
            XB_i = XB[lbo_i];
            #local observation operator
            H_ind = np.array(lobs[i]).astype('int32'); #local observed components
            m_i = H_ind.size;
            if m_i>0:
                n_i = xb_i.size;
                I = np.arange(0, m_i);
                J = H_ind;
                H_i = spa.coo_matrix((np.ones(m_i),(I,J)), shape=(m_i, n_i));
                DX_i = DX[lbo_i];
                y_i  = y_model[lbo_i[H_ind]];
                Ri_i = np.diag(Ri_space[lbo_i[H_ind], lbo_i[H_ind]]).reshape((m_i, m_i)); #local data error covariance matrix
                XA_i = self.perform_assimilation_local_box(XB_i, xb_i, DX_i, H_i, Ri_i, y_i);
            else:
                XA_i = XB_i;
                
            XA[i, :] = XA_i[gp_i, :]; 
        
        return XA;

##########################################################################################
# LEnKF (Original, Unchanged)
##########################################################################################
class LEnKF(ensemble_DA):
            
    
    def prepare_background(self):
        mask_cor = self.nm.mask_cor;
        var_names = self.nm.var_names;
        Nens = self.Nens;
        self.XB_map = [];
        gr = self.nm.gs;
        for msk_cor, lbo_info in zip(mask_cor, gr.lbo):
            XB_block = self.get_ensemble_block(msk_cor);
            xb_block = XB_block.mean(axis=1).reshape(-1,1);
            DX_block = XB_block-xb_block;
            self.XB_map.append({'XB_b':XB_block, 'DX_b':DX_block, 'lbo_b':lbo_info});


    def prepare_analysis(self, ob, k, args = None):
        self.Ys = [];
        Nens = self.Nens;
        mask_cor = self.nm.mask_cor;
        N_blocks = len(mask_cor);
        for block in range(0, N_blocks):
            Ys_block = ob.get_perturbed_observations(block, k, Nens);  
            self.Ys.append(Ys_block);
        self.nm.gs.compute_local_boxobs(ob.obs_H_sparse); 
    
    def perform_assimilation(self, ob):
        self.XA_map = [];
        for XB_info, Ys_block, R_info, H_block, lobs_block in zip(self.XB_map, self.Ys, ob.obs_R_sparse, ob.obs_H_sparse, self.nm.gs.lbo_obs):
            XB_block = XB_info['XB_b'];
            lbo_block = XB_info['lbo_b'];
            Ri_block  = R_info['Ri'];
            XA_block = self.perform_assimilation_block(XB_block, H_block, Ri_block, Ys_block, lbo_block, lobs_block);
            self.XA_map.append(XA_block);
        self.map_vector_states(); #Update ensemble folders
        
    
    def perform_assimilation_local_box(self, XB, H, R, Ys):
    
        Ds = Ys - H @ XB;
        
        Pb = np.cov(XB);
        
        XA = XB + Pb @ ( H.T @ np.linalg.solve(R + H @ (Pb @ H.T), Ds) );
        
        return XA;
        
        
    
    def perform_assimilation_block(self, XB, H, Ri, Ys, lbo_info, lobs):

        
        n, Nens = XB.shape;
        
        lbo, nlbo = lbo_info; #local boxes information (indexes, number of components in all boxes)

        Ys_model = H.T @ Ys;
        
        
        Ri_space = H.T @ (Ri @ H);
        Ri_space = Ri_space.toarray();
        
        XA = np.zeros((n, Nens));
        
        
        for i in range(0, n):
            #local box for model component i
            lbo_i = np.array(lbo[i]).astype('int32');
            gp_i, = np.where(lbo_i==i);
            XB_i = XB[lbo_i];
            #local observation operator
            H_ind = np.array(lobs[i]).astype('int32'); #local observed components
            m_i = H_ind.size;
            if m_i>0:
                n_i,_ = XB_i.shape;
                I = np.arange(0, m_i);
                J = H_ind;
                H_i = spa.coo_matrix((np.ones(m_i),(I,J)), shape=(m_i, n_i));
                Ys_i  = Ys_model[lbo_i[H_ind]];
                Ri_i = np.diag(Ri_space[lbo_i[H_ind], lbo_i[H_ind]]).reshape((m_i, m_i)); #local data error covariance matrix
                XA_i = self.perform_assimilation_local_box(XB_i, H_i, Ri_i, Ys_i);
            else:
                XA_i = XB_i;
                
            XA[i, :] = XA_i[gp_i, :]; 
        
        return XA;

##########################################################################################
##########################################################################################
##########################################################################################
# NEW CLASS: ReverseSDE
# Integration of the Ensemble Score Filter based on the provided Rev_SDE.py code.
##########################################################################################
##########################################################################################
##########################################################################################
class ReverseSDE(ensemble_DA):
    
    def __init__(self, nm, infla, Nens, pseudo_time_step=100, scalefact=0.1, likelihood_weight=1.0):
        """
        Initializes the ReverseSDE data assimilation method.

        Args:
            pseudo_time_step (int): The number of time steps for the reverse SDE solver.
            scalefact (float): A scaling factor used in the observation operator.
        """
        super().__init__(nm, infla, Nens)
        self.pseudo_time_step = pseudo_time_step
        self.scalefact = scalefact
        self.obs_info = []
        self.likelihood_weight = likelihood_weight


    def prepare_background(self):
        """
        Prepares the background ensemble for each block.
        """
        mask_cor = self.nm.mask_cor
        self.XB_map = []
        for msk_cor in mask_cor:
            XB_block = self.get_ensemble_block(msk_cor)
            self.XB_map.append({'XB_b': XB_block})

    def prepare_analysis(self, ob, k, args=None):
        """
        Prepares the observation data needed for the assimilation for each block.
        """
        self.obs_info = []
        mask_cor = self.nm.mask_cor
        N_blocks = len(mask_cor)

        for block in range(N_blocks):
            y_block = ob.y_obs[k][block]
            R_block = ob.obs_R_sparse[block]['R']
            H_block = ob.obs_H_sparse[block]
            
            # Extract observation error standard deviation from the covariance matrix
            sigma_block = np.sqrt(R_block.diagonal())
            
            # Determine the indices of the state variables that are observed
            # H is sparse, non-zero columns indicate observed states.
            indxob_block = np.unique(H_block.tocoo().col)
            
            y_dim = len(y_block)
            
            # NOTE: Assuming all observations are linear as H is a linear operator.
            # The REVERSE_SDE class can handle non-linear obs, but the framework's H is linear.
            # Therefore, all observation indices are considered linear.
            indx_indxob_linear = np.arange(y_dim)

            self.obs_info.append({
                'y': y_block,
                'sigma': sigma_block,
                'y_dim': y_dim,
                'indxob': indxob_block,
                'indx_indxob_linear': indx_indxob_linear
            })
            
    def perform_assimilation(self, ob):
        """
        Performs the data assimilation for each block using the Reverse SDE method.
        """
        self.XA_map = []
        for XB_info, obs_block_info in zip(self.XB_map, self.obs_info):
            XB_block = XB_info['XB_b']
            
            # The original standard deviation of the prior ensemble is needed for inflation
            initial_std = np.std(XB_block, axis=1) # <--- CORRECTED LINE

            # Transpose XB_block to match the expected shape (ensemble_size, state_dim)
            prior_ensemble = XB_block.T
            
            # Instantiate the REVERSE_SDE solver for the current block
            sde_solver = REVERSE_SDE(
                pseudo_time_step=self.pseudo_time_step,
                prior_ensemble=prior_ensemble,
                ensemble_size=self.Nens,
                obs=obs_block_info['y'].flatten(),
                sigma=obs_block_info['sigma'],
                y_dim=obs_block_info['y_dim'],
                scalefact=self.scalefact,
                indxob=obs_block_info['indxob'],
                indx_indxob_linear=obs_block_info['indx_indxob_linear'],
                initial_std=initial_std,
                likelihood_weight=self.likelihood_weight
            )
            
            # Run the assimilation
            xa_block_transposed = sde_solver.reverse_SDE()
            
            # Transpose the result back to the framework's convention (state_dim, ensemble_size)
            XA_block = xa_block_transposed.T

            # The REVERSE_SDE class has its own inflation, so we don't call self.covariance_inflation
            self.XA_map.append(XA_block)
            
        self.map_vector_states() # Update ensemble folders


##########################################################################################
##########################################################################################
##########################################################################################
# General factory - sequential ensemble data assimilation
##########################################################################################
##########################################################################################
##########################################################################################
class sequential_method:
      
    method_name = None;
      
    def __init__(self, method_name):
        self.method_name = method_name;
      
    def get_instance(self, nm, infla, Nens, p_time_step=100, likelihood_weight=1.0):
        if self.method_name=='EnKF_MC_obs': return EnKF_MC_obs(nm, infla, Nens);
        if self.method_name=='LETKF': return LETKF(nm, infla, Nens);
        if self.method_name=='LEnKF': return LEnKF(nm, infla, Nens);
        # Register the new filter with the factory
        if self.method_name=='ReverseSDE': 
            return ReverseSDE(nm, infla, Nens, pseudo_time_step=p_time_step, likelihood_weight=likelihood_weight)
