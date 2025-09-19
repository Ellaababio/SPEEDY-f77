import os
import numpy as np
import sys
import scipy.sparse as spa
import pandas as pd
import time
from netCDF4 import Dataset
from commons_utils import compute_modified_Cholesky_decomposition
import torch


##########################################################################################
##########################################################################################
##########################################################################################
# General class - sequential ensemble data assimilation
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
##########################################################################################
##########################################################################################
# Observation space version of:
# Nino-Ruiz, E. D., Sandu, A., & Deng, X. (2018). An ensemble Kalman filter implementation based on modified Cholesky decomposition for inverse covariance matrix estimation. SIAM Journal on Scientific Computing, 40(2), A867-A886.
# To be published
##########################################################################################
##########################################################################################
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
          
          #Q_temp = spa.linalg.spsolve_triangular(Binv_sqrt,Q_temp,lower=False);
          
          DXa = spa.linalg.spsolve_triangular(Binv_sqrt.T, Q_temp, lower=True, check_finite=False);

          XA = XB + DXa;  
          
          return XA;
      
         
##########################################################################################
##########################################################################################
##########################################################################################
# Miyoshi, T., & Yamane, S. (2007). Local ensemble transform Kalman filtering with an AGCM at a T159/L48 resolution. Monthly Weather Review, 135(11), 3841-3861.
##########################################################################################
##########################################################################################
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
        
        #print(f'Ri {Ri.shape}');
        #print(f'Yb {Yb.shape}');
        #print(f'd {d.shape}');
        #print(f'H {H.shape}');
        #print(f'xb {xb.shape}');
        #print(f'y {y.shape}');
        
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
              #print(f'H_ind {H_ind}');
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
                 #perform_assimilation_local_box(self, XB, xb, DX, H, Ri, y)
              else:
                 XA_i = XB_i;
                 
              XA[i, :] = XA_i[gp_i, :]; 
          
          return XA;
          #XB, xb, DX, H, Ri, y):
          



##########################################################################################
##########################################################################################
##########################################################################################
# Ott, E., Hunt, B. R., Szunyogh, I., Zimin, A. V., Kostelich, E. J., Corazza, M., ... & Yorke, J. A. (2004). A local ensemble Kalman filter for atmospheric data assimilation. Tellus A: Dynamic Meteorology and Oceanography, 56(5), 415-428.
##########################################################################################
##########################################################################################
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
              #print(f'H_ind {H_ind}');
              m_i = H_ind.size;
              if m_i>0:
                 n_i,_ = XB_i.shape;
                 I = np.arange(0, m_i);
                 J = H_ind;
                 H_i = spa.coo_matrix((np.ones(m_i),(I,J)), shape=(m_i, n_i));
                 Ys_i  = Ys_model[lbo_i[H_ind]];
                 Ri_i = np.diag(Ri_space[lbo_i[H_ind], lbo_i[H_ind]]).reshape((m_i, m_i)); #local data error covariance matrix
                 XA_i = self.perform_assimilation_local_box(XB_i, H_i, Ri_i, Ys_i);
                 #perform_assimilation_local_box(self, XB, xb, DX, H, Ri, y)
              else:
                 XA_i = XB_i;
                 
              XA[i, :] = XA_i[gp_i, :]; 
          
          return XA;
          #XB, xb, DX, H, Ri, y):   

##########################################################################################
##########################################################################################
##########################################################################################
# Reverse-SDE Score Filter (global/observation-space variant)
# Adapted to the ensemble_DA class interface used in this repo.
#
# Notes/assumptions:
# - Treats observations as *linear direct selections* (H rows each pick one state component),
#   matching how other methods build H. Nonlinear obs can be added where marked TODOs live.
# - Operates block-by-block, consistent with EnKF_MC_obs / LETKF / LEnKF structure.
# - No torch dependency; pure NumPy.
##########################################################################################
##########################################################################################
##########################################################################################
class ReverseSDE(ensemble_DA):

    def __init__(self, nm, infla, Nens,
                 pseudo_time_steps: int = 200,
                 eps_alpha: float = 0.05,
                 scalefact: float = 1.0,
                 rng_seed: int = 42):
        super().__init__(nm, infla, Nens)
        self.p_time_step = int(pseudo_time_steps)
        self.eps_alpha = float(eps_alpha)
        self.scalefact = float(scalefact)
        self.rng = np.random.RandomState(int(rng_seed))

        # Per-block working storage
        self.XB_map = []      # background info (like other methods)
        self.obs_map = []     # per-block obs mappings prepared in prepare_analysis
        self.XA_map = None

    # ===== Reverse-SDE schedule pieces (mirrors the original formulation) =====
    def _cond_alpha(self, t):
        # alpha_t
        return 1.0 - (1.0 - self.eps_alpha) * t

    def _cond_sigma_sq(self, t):
        # sigma_t^2
        return t

    def _f(self, t):
        # f = d(log alpha)/dt
        alpha_t = self._cond_alpha(t)
        return -(1.0 - self.eps_alpha) / alpha_t

    def _g(self, t):
        # g satisfies: d(sigma^2)/dt - 2 f sigma^2 = g^2
        d_sigma_sq_dt = 1.0
        g2 = d_sigma_sq_dt - 2.0 * self._f(t) * self._cond_sigma_sq(t)
        return np.sqrt(g2)

    def _g_tau(self, t):
        # likelihood tempering (tau) as in the original code
        return 1.0 - t

    # ====== ensemble_DA hooks ======
    def prepare_background(self):
        """
        Build per-block background structures (just like other methods),
        and cache statistics needed later (means/stds).
        """
        self.XB_map = []
        gr = self.nm.gs
        for msk_cor, _unused in zip(self.nm.mask_cor, getattr(gr, "lpr", gr.lbo)):
            XB_block = self.get_ensemble_block(msk_cor)       # shape: (n, Nens)
            xb_block = XB_block.mean(axis=1, keepdims=True)   # (n,1)
            DX_block = XB_block - xb_block                    # (n, Nens)
            # Per-dimension spread (used for post inflation to restore initial std)
            std_init = DX_block.std(axis=1)                   # (n,)
            self.XB_map.append({
                "XB_b": XB_block,
                "xb_b": xb_block,
                "DX_b": DX_block,
                "std_init": std_init
            })
        # nothing returned; stored on self

    def prepare_analysis(self, ob, k, args=None):
        """
        Build observed indices per block from sparse H and get 1-D y and sigma that
        match the H-row order. Works with dense or sparse R/Ri.
        """
        import numpy as _np
        import scipy.sparse as _spa

        self.obs_map = []

        for block, XB_info in enumerate(self.XB_map):
            # Sparse H (m x n)
            H_block = ob.obs_H_sparse[block]
            # R info may contain 'R' (cov) and/or 'Ri' (precision)
            R_info = ob.obs_R_sparse[block]
            # Observations for this time step/block (could be (m,), (m,1), etc.)
            y_block = _np.asarray(ob.y_obs[k][block])

            # --- derive ordered state-column indices each H row selects ---
            Hc = H_block.tocoo()
            order = _np.argsort(Hc.row)
            idx_observed = Hc.col[order].astype(_np.int64)  # (m,)
            m = idx_observed.size

            # --- y must be 1-D (m,) and aligned to H rows ---
            y_block = y_block.reshape(-1)[:m]  # force 1-D, trim just in case

            # --- robust diagonal extraction from R or Ri ---
            def _diag_from(x):
                if x is None:
                    return None
                if _spa.issparse(x):
                    return x.diagonal()
                x = _np.asarray(x)
                if x.ndim == 1:
                    return x
                if x.ndim == 2:
                    return _np.diag(x)
                raise ValueError("R/Ri must be 1D or 2D (or sparse)")
            diag_R  = _diag_from(R_info.get("R",  None))
            diag_Ri = _diag_from(R_info.get("Ri", None))

            eps = _np.finfo(float).eps
            if diag_R is not None:
                sigma = _np.sqrt(_np.maximum(eps, diag_R)).reshape(-1)[:m]
            elif diag_Ri is not None:
                sigma = 1.0 / _np.sqrt(_np.maximum(eps, diag_Ri)).reshape(-1)[:m]
            else:
                raise ValueError("Neither R nor Ri found in ob.obs_R_sparse[block].")

            # Save per-block obs info (all 1-D arrays)
            self.obs_map.append({
                "idx_observed": idx_observed,
                "y": y_block.astype(float),
                "sigma": sigma.astype(float),
            })


    def perform_assimilation(self, ob):
        """
        Reverse-SDE block update (Torch) with proper publication of self.XA.
        Requires:
        - self.XB_map from prepare_background()
        - self.obs_map from prepare_analysis()
        Produces:
        - self.XA_map: list of (n_block, Nens) arrays for all blocks
        - self.XA via self.map_vector_states()
        """
        import numpy as _np
        import torch as _torch

        # Fresh output container for all blocks
        self.XA_map = []

        # Torch RNG/device
        if not hasattr(self, "device"):
            self.device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        _torch.manual_seed(getattr(self, "rng_seed", 42))

        # Pseudo-time setup
        psteps = int(getattr(self, "p_time_step", 200))
        dt = 1.0 / float(psteps)
        hist_len = 100
        tol = 1.0e-4
        sf = float(getattr(self, "scalefact", 1.0))

        def _cond_alpha(tt, eps_alpha):
            return 1.0 - (1.0 - eps_alpha) * tt

        def _cond_sigma_sq(tt):
            return tt

        def _f(tt, eps_alpha):
            alpha_t = 1.0 - (1.0 - eps_alpha) * tt
            return -(1.0 - eps_alpha) / alpha_t

        def _g(tt, eps_alpha):
            d_sigma_sq_dt = 1.0
            g2 = d_sigma_sq_dt - 2.0 * _f(tt, eps_alpha) * _cond_sigma_sq(tt)
            return float(_np.sqrt(g2))

        def _g_tau(tt):
            return 1.0 - tt

        eps_alpha = float(getattr(self, "eps_alpha", 0.05))
        device = self.device

        for block, (XB_info, OM) in enumerate(zip(self.XB_map, self.obs_map)):
            # --- Background block ---
            XB = XB_info["XB_b"]            # shape: (n_block, Nens)
            init_std = XB_info["std_init"]  # shape: (n_block,)
            n_block, Nens = XB.shape

            # Observed indices inside this block
            idx_ob = OM["idx_observed"]     # (m,)
            m = idx_ob.size if idx_ob is not None else 0

            # If no observations map to this block, carry background forward with inflation
            if m == 0:
                XA_block = self.covariance_inflation(XB)  # (n_block, Nens)
                self.XA_map.append(XA_block)
                continue

            # Prior ensemble as (Nens, n_block)
            prior_ens = XB.T.copy()

            # Observations (1-D arrays aligned to H rows)
            y = OM["y"].reshape(-1)         # (m,)
            sigma = OM["sigma"].reshape(-1) # (m,)

            # ----- Normalize observed subspace using prior stats -----
            X0_obs = prior_ens[:, idx_ob]               # (Nens, m)
            mean_X0 = X0_obs.mean(axis=0)               # (m,)
            std_X0  = X0_obs.std(axis=0) + 1e-12        # (m,)
            X0_obs_n = (X0_obs - mean_X0) / std_X0      # (Nens, m)

            # Linear obs model: y ≈ sf * x_obs
            y_n = ((y - sf * mean_X0) / std_X0).reshape(-1)          # (m,)
            sigma_n = ((sigma / std_X0) * sf).reshape(-1)            # (m,)

            # ----- Torch tensors (ensure 1-D for broadcasting with xt: (Nens, m)) -----
            X0_obs_n_t = _torch.from_numpy(X0_obs_n).to(device=device, dtype=_torch.float32)  # (Nens, m)
            y_n_t      = _torch.from_numpy(y_n).to(device=device, dtype=_torch.float32)       # (m,)
            sigma_n_t  = _torch.from_numpy(sigma_n).to(device=device, dtype=_torch.float32)   # (m,)

            # Initialize pseudo-time state in observed subspace
            xt = _torch.randn((Nens, m), device=device, dtype=_torch.float32)

            xt_means_hist = _torch.zeros((hist_len, m), device=device, dtype=_torch.float32)
            t = 1.0

            # ----- Reverse SDE loop (observed subspace only) -----
            for i in range(psteps):
                alpha_t = _cond_alpha(t, eps_alpha)
                sigma2_t = _cond_sigma_sq(t)
                g = _g(t, eps_alpha)
                f = _f(t, eps_alpha)
                tau = _g_tau(t)

                # prior term: (xt - alpha_t * X0_obs_n) / sigma2_t
                prior_term = (xt - alpha_t * X0_obs_n_t) / sigma2_t  # (Nens, m)

                # linear likelihood score:  -(sf*xt - y_n)/sigma_n^2 * sf
                like_score = -(sf * xt - y_n_t) / (sigma_n_t ** 2) * sf  # (Nens, m)

                drift = - (f * xt + (g ** 2) * (prior_term - tau * like_score))
                noise = _torch.sqrt(_torch.tensor(dt, device=device)) * g * _torch.randn_like(xt)
                xt_next = xt + dt * drift + noise

                # early stop heuristic
                if i > 0.5 * psteps:
                    mu = xt_next.mean(dim=0)
                    prev_mu = xt_means_hist.mean(dim=0)
                    if _torch.all(_torch.abs(mu - prev_mu) < tol):
                        xt = xt_next
                        break
                    xt_means_hist[i % hist_len, :] = mu

                # hard stop guard near the end
                if i > 0.9 * psteps:
                    xt = xt_next
                    break

                xt = xt_next
                t = max(0.0, t - dt)

            # ----- De-normalize observed subspace and stitch back -----
            xt_np = xt.detach().cpu().numpy()            # (Nens, m)
            x_obs_ana = mean_X0 + xt_np * std_X0         # (Nens, m)

            x_full = prior_ens.copy()                    # (Nens, n_block)
            x_full[:, idx_ob] = x_obs_ana                # observed dims updated

            # restore per-dimension spread to initial std (discentralize)
            mu = x_full.mean(axis=0)                     # (n_block,)
            sd = x_full.std(axis=0) + 1e-12              # (n_block,)
            Xstd = (x_full - mu) / sd
            x_full = Xstd * init_std + mu

            # to (n_block, Nens), then apply global inflation like other methods
            XA_block = x_full.T
            XA_block = self.covariance_inflation(XA_block)

            # Append this block’s analysis
            self.XA_map.append(XA_block)

        # Sanity: one output per correlation block
        assert len(self.XA_map) == len(self.nm.mask_cor), \
            f"ReverseSDE: XA_map has {len(self.XA_map)} blocks, expected {len(self.nm.mask_cor)}."

        # Publish analysis ensemble (sets self.XA and writes member NetCDFs)
        self.map_vector_states()

        # Final guard to avoid downstream None errors
        assert self.XA is not None and len(self.XA) == self.Nens, \
            "ReverseSDE: self.XA not built after map_vector_states()."




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
      
      def get_instance(self, nm, infla, Nens):
          if self.method_name=='EnKF_MC_obs': return EnKF_MC_obs(nm, infla, Nens);
          if self.method_name=='LETKF': return LETKF(nm, infla, Nens);
          if self.method_name=='LEnKF': return LEnKF(nm, infla, Nens);
          if self.method_name == 'ReverseSDE':return ReverseSDE(nm, infla, Nens)
          



    
    
    
    
        
    
