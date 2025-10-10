import os
import numpy as np
import sys
import scipy.sparse as spa
import pandas as pd
import time
from netCDF4 import Dataset
from commons_utils import compute_modified_Cholesky_decomposition
import torch
from netCDF4 import Dataset as NC
# ===== BEGIN: generic grid dump helpers =====


import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset as NC

VAR_ORDER = ['UG0','VG0','TG0','TRG0','PSG0','UG1','VG1','TG1','TRG1','PSG1']

def _extract_grid_from_mean_state(mean_state, var_name, level):
    """Return a 2D (nlat, nlon) array slice from a model mean_state structure."""
    vi = VAR_ORDER.index(var_name)
    arr = mean_state[vi]
    # TRG may have a tracer dim (1, lev, nlat, nlon)
    if var_name.startswith("TRG") and arr.ndim == 4:
        arr = arr[0, ...]
    # PSG is 2D (nlat, nlon)
    if var_name.startswith("PSG"):
        return np.asarray(arr, dtype=float)
    return np.asarray(arr[level, :, :], dtype=float)

def write_grid_values_csv(var_name, level, grid_bkg, grid_ana, grid_truth,
                          grid_noda=None, out_csv_path=None):
    """
    Write raw grid values to a CSV with flattened columns.
    Columns (NoDA available):  <var>_bkg_lev<k>, <var>_ana_lev<k>, <var>_truth_lev<k>, <var>_noda_lev<k>
    Columns (NoDA missing):   <var>_bkg_lev<k>, <var>_ana_lev<k>, <var>_truth_lev<k>
    """
    cols = {
        f"{var_name}_bkg_lev{level}":   grid_bkg.ravel(),
        f"{var_name}_ana_lev{level}":   grid_ana.ravel(),
        f"{var_name}_truth_lev{level}": grid_truth.ravel(),
    }
    if grid_noda is not None:
        cols[f"{var_name}_noda_lev{level}"] = grid_noda.ravel()

    df = pd.DataFrame(cols)
    if out_csv_path is None:
        out_csv_path = f"{var_name}_lev{level}_values.csv"
    df.to_csv(out_csv_path, index=False)
    return out_csv_path


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
        self._cycle_idx = 0  # incremented once per perform_assimilation() call
        # Per-block working storage
        self.XB_map = []      # background info (like other methods)
        self.obs_map = []     # per-block obs mappings prepared in prepare_analysis
        self.XA_map = None
    # ===== BEGIN: generic grid dump helpers =====


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

    # ====== ensemble_DA hooks ======
    def perform_assimilation(self, ob):
        """
        Reverse-SDE block update (Torch) with hardened publication of self.XA:
        - Preflight logs of block counts; auto-fallback if XB/obs maps are empty.
        - Iterates by block index (not zip) to avoid silent truncation.
        - Per-10-step likelihood-score debug.
        - One-time dump of block_map.json for external analyzers.
        - Graceful fallback (inflated background) for any problematic/missing block.
        """
        import os
        import json
        import numpy as _np
        import torch as _torch

        # -------- CONFIG --------
        self._cycle_idx = getattr(self, "_cycle_idx", 0) + 1
        DEBUG_EVERY = 20
        SAVE_GAUSS_BLOCKS = False
        GAUSS_DIRNAME = "gauss_checks"
        # ------------------------

        # Ensure device / RNG
        if not hasattr(self, "device"):
            self.device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        _torch.manual_seed(getattr(self, "rng_seed", 42))
        device = self.device

        # SDE schedule params
        psteps = int(getattr(self, "p_time_step", 200))
        dt = 1.0 / float(psteps)
        hist_len = 100
        tol = 1.0e-4
        sf = float(getattr(self, "scalefact", 1.0))
        eps_alpha = float(getattr(self, "eps_alpha", 0.05))

        # Helper: readable block label (like your current prints)
        def _block_label(block_idx):
            try:
                items = self.nm.mask_cor[block_idx]
            except Exception:
                return f"block={block_idx} (mask_cor unavailable)"
            if not items:
                return f"block={block_idx} (empty)"
            var_names, levels = [], set()
            for it in items:
                (v_idx, lev) = it[0]
                var_names.append(self.nm.var_names[v_idx])
                levels.add(lev)
            lev_str = f"lev={list(levels)[0]}" if len(levels) == 1 else "levs={" + ",".join(str(L) for L in sorted(levels)) + "}"
            vlist = ",".join(var_names[:5]) + ("..." if len(var_names) > 5 else "")
            return f"block={block_idx} [{lev_str}] vars={vlist}"

        # Prepare output dir for gaussianity blocks (if enabled)
        if SAVE_GAUSS_BLOCKS:
            try:
                os.makedirs(os.path.join(self.nm.path, GAUSS_DIRNAME), exist_ok=True)
            except Exception as e:
                print(f"[ReverseSDE] WARNING: could not create gauss dir: {e}")

        # Preflight: ensure maps exist
        xb_len = len(getattr(self, "XB_map", []) or [])
        om_len = len(getattr(self, "obs_map", []) or [])
        mc_len = len(self.nm.mask_cor)

        print(f"[ReverseSDE] preflight: XB_map={xb_len}, obs_map={om_len}, mask_cor={mc_len}")
        '''
        # One-time dump of block_map.json (only if maps are present and we haven't dumped yet)
        if xb_len > 0 and not getattr(self, "_block_map_dumped", False):
            try:
                blocks = []
                for b_idx, items in enumerate(self.nm.mask_cor):
                    vars_set, levs_set = set(), set()
                    for it in items:
                        (v_idx, lev) = it[0]
                        vars_set.add(self.nm.var_names[v_idx])
                        levs_set.add(int(lev))
                    blocks.append({
                        "block_idx": b_idx,
                        "vars": sorted(vars_set),
                        "levels": sorted(levs_set)
                    })
                out = {"var_names": list(self.nm.var_names), "blocks": blocks}
                out_path = os.path.join(self.nm.path, "block_map.json")
                with open(out_path, "w") as f:
                    json.dump(out, f, indent=2)
                print(f"[ReverseSDE] wrote block_map.json -> {out_path}")
            except Exception as e:
                print(f"[ReverseSDE] WARNING: failed to write block_map.json: {e}")
            finally:
                self._block_map_dumped = True  # never attempt again this process
        '''
        self.XA_map = []

        # AUTO-FALLBACK if XB_map missing (no background prepared)
        if xb_len == 0:
            print("[ReverseSDE] WARNING: XB_map is empty. Using inflated background as analysis for all blocks.")
            for block_idx in range(mc_len):
                XB_block = self.get_ensemble_block(self.nm.mask_cor[block_idx])  # (n_block, Nens)
                XA_block = self.covariance_inflation(XB_block)
                self.XA_map.append(XA_block)
                if SAVE_GAUSS_BLOCKS:
                    try:
                        _np.save(os.path.join(self.nm.path, GAUSS_DIRNAME, f"XB_block_{block_idx}.npy"), XB_block)
                    except Exception as e:
                        print(f"[ReverseSDE][block={block_idx}] WARNING: save failed: {e}")
            # Publish and return
            self.map_vector_states()
            return

        # Main loop (index-based to avoid zip truncation)
        for block_idx in range(mc_len):
            label = _block_label(block_idx)

            # Background info for this block
            try:
                XB_info = self.XB_map[block_idx]
            except Exception as e:
                print(f"[ReverseSDE][{label}] ERROR: XB_map missing block {block_idx}: {e}. Using fallback.")
                XB_block = self.get_ensemble_block(self.nm.mask_cor[block_idx])
                XA_block = self.covariance_inflation(XB_block)
                self.XA_map.append(XA_block)
                continue

            XB = XB_info["XB_b"]            # (n_block, Nens)
            init_std = XB_info["std_init"]  # (n_block,)
            n_block, Nens = XB.shape

            # Get observation map (may be None)
            OM = self.obs_map[block_idx] if block_idx < om_len else None

            # Default fallback: inflated background
            XA_block_fallback = self.covariance_inflation(XB)

            # No obs case
            if (OM is None) or ("idx_observed" not in OM) or (OM["idx_observed"] is None) or (OM["idx_observed"].size == 0):
                self.XA_map.append(XA_block_fallback)
                continue

            # Observed indices and vectors
            idx_ob = OM["idx_observed"]
            y = OM["y"].reshape(-1)
            sigma = OM["sigma"].reshape(-1)
            m = idx_ob.size

            # Prior ensemble in (Nens, n_block)
            prior_ens = XB.T.copy()

            # Normalize observed subspace
            X0_obs = prior_ens[:, idx_ob]                 # (Nens, m)
            mean_X0 = X0_obs.mean(axis=0)                 # (m,)
            std_X0  = X0_obs.std(axis=0) + 1e-12          # (m,)
            X0_obs_n = (X0_obs - mean_X0) / std_X0        # (Nens, m)

            y_n = ((y - sf * mean_X0) / std_X0).reshape(-1)       # (m,)
            sigma_n = ((sigma / std_X0) * sf).reshape(-1)         # (m,)

            # Torch tensors
            X0_obs_n_t = _torch.from_numpy(X0_obs_n).to(device=device, dtype=_torch.float32)
            y_n_t      = _torch.from_numpy(y_n).to(device=device, dtype=_torch.float32)
            sigma_n_t  = _torch.from_numpy(sigma_n).to(device=device, dtype=_torch.float32)

            # Initialize reverse SDE state
            xt = _torch.randn((Nens, m), device=device, dtype=_torch.float32)
            xt_means_hist = _torch.zeros((hist_len, m), device=device, dtype=_torch.float32)
            t = 1.0

            print(f"[ReverseSDE][{label}] starting reverse-SDE (m={m}, Nens={Nens})")

            block_numeric_ok = True

            for i in range(1, psteps+1):
                if i == 1:  # only print once per block
                    try:
                        std_val   = std_X0.mean().item()
                        sigma_val = sigma.mean().item()
                        sigma_n_val = sigma_n.mean().item()
                        print(f"[ReverseSDE][{label}] init stats: "
                            f"ensemble std={std_val:.3e}, obs err σ={sigma_val:.3e}, normalized σ_n={sigma_n_val:.3e}")
                    except Exception as e:
                        print(f"[ReverseSDE][{label}] could not compute init stats: {e}")
                alpha_t = self._cond_alpha(t)
                sigma2_t  = self._cond_sigma_sq(t)
                f       = self._f(t)
                g       = self._g(t)
                tau     = self._g_tau(t)

                prior_term = (xt - alpha_t * X0_obs_n_t) / sigma2_t
                like_score  = -((sf * xt) - y_n_t) / (sigma_n_t ** 2) * sf  # no tempering

                # tempered likelihood score that actually enters the drift
                like_tau = tau * like_score

                # likelihood contribution to the drift magnitude (ignore the separate prior pull):
                # drift = -( f*xt + g^2 * (prior_term - like_tau) )
                # so the likelihood "pull" part is  + g^2 * like_tau
                pull = (g ** 2) * like_tau

                if  DEBUG_EVERY > 0 and (i % DEBUG_EVERY == 0 or i == 1):
                    with _torch.no_grad():
                        abs_base = _torch.abs(like_score)
                        abs_tau  = _torch.abs(like_tau)
                        abs_pull = _torch.abs(pull)

                        # finite mask
                        m = _torch.isfinite(abs_pull) & _torch.isfinite(abs_tau) & _torch.isfinite(abs_base)
                        if m.any():
                            print(
                                f"[ReverseSDE][{label}] step={i:03d} "
                                f"|like_score| mean={abs_base[m].mean().item():.4e} max={abs_base[m].max().item():.4e}  "
                                f"|like_tau| mean={abs_tau[m].mean().item():.4e} max={abs_tau[m].max().item():.4e}  "
                                #f"|pull| mean={abs_pull[m].mean().item():.4e} max={abs_pull[m].max().item():.4e}  "
                                #f"tau={tau:.3f} g={g:.3e}"
                            )
                        else:
                            print(f"[ReverseSDE][{label}] step={i:03d} diagnostics non-finite")
                # --- end diagnostics ---

                drift = - (f * xt + (g ** 2) * (prior_term - tau * like_score))
                noise = _torch.sqrt(_torch.tensor(dt, device=device, dtype=xt.dtype)) * g * _torch.randn_like(xt)
                xt_next = xt + dt * drift + noise

                if not _torch.isfinite(xt_next).all():
                    print(f"[ReverseSDE][{label}] State became non-finite at step {i}. Using fallback for this block.")
                    block_numeric_ok = False
                    break
                '''
                # early stop heuristic
                if i > 0.5 * psteps:
                    mu = xt_next.mean(dim=0)
                    prev_mu = xt_means_hist.mean(dim=0)
                    if _torch.all(_torch.abs(mu - prev_mu) < tol):
                        xt = xt_next
                        break
                    xt_means_hist[i % hist_len, :] = mu

                # hard stop near the end
                if i > 0.9 * psteps:
                    xt = xt_next
                    break
                '''
                xt = xt_next
                t = max(0.0, t - dt)
            # Produce XA_block (either fallback or from xt)
            if not block_numeric_ok:
                self.XA_map.append(XA_block_fallback)
                continue

            xt_np = xt.detach().cpu().numpy()                # (Nens, m)
            x_obs_ana = mean_X0 + xt_np * std_X0             # (Nens, m)
            x_full = prior_ens.copy()                        # (Nens, n_block)
            x_full[:, idx_ob] = x_obs_ana

            mu = x_full.mean(axis=0)                         # (n_block,)
            sd = x_full.std(axis=0) + 1e-12
            Xstd = (x_full - mu) / sd
            x_full = Xstd * init_std + mu

            XA_block = self.covariance_inflation(x_full.T)   # (n_block, Nens)
            self.XA_map.append(XA_block)

            # (optional) save clean XB for gaussianity if enabled
            if SAVE_GAUSS_BLOCKS:
                try:
                    path_out = os.path.join(self.nm.path, GAUSS_DIRNAME, f"XB_block_{block_idx}.npy")
                    _np.save(path_out, XB)
                    print(f"[ReverseSDE][{label}] saved BACKGROUND block for gaussianity: {path_out}")
                except Exception as e:
                    print(f"[ReverseSDE][{label}] WARNING: save failed: {e}")

        # Final sanity: if XA_map is short, pad with inflated background per block
        if len(self.XA_map) < mc_len:
            print(f"[ReverseSDE] WARNING: XA_map has {len(self.XA_map)} of {mc_len} blocks. Padding with fallbacks.")
            for block_idx in range(len(self.XA_map), mc_len):
                XB_block = self.get_ensemble_block(self.nm.mask_cor[block_idx])
                XA_block = self.covariance_inflation(XB_block)
                self.XA_map.append(XA_block)
        # ---------- generic grid dump call (TG1 level 7 by default) ----------
        # --- grid dump (safe version) ---
        # --- grid dump (per-cycle) ---
        # ---- write raw grid values for one var/level (e.g., TG1 @ lev 7) ----
        k = getattr(self, "_cycle_idx", 0)  # per-cycle counter you added earlier
        var_name = "TG1"
        level    = 7  # surface

        try:
            k = getattr(self, "_cycle_idx", 0)  # your per-cycle counter

            # Require both ensembles to exist
            if getattr(self, "XB", None) is None or getattr(self, "XA", None) is None:
                raise RuntimeError(f"XB or XA is None at cycle {k}; skipping grid dump.")

            # Ensemble means from your model wrapper
            xb_mean = self.nm.compute_snapshot_mean(self.XB)
            xa_mean = self.nm.compute_snapshot_mean(self.XA)

            var_name = "TG1"
            level    = 7  # surface

            grid_bkg = _extract_grid_from_mean_state(xb_mean, var_name, level)
            grid_ana = _extract_grid_from_mean_state(xa_mean, var_name, level)

            # Load truth for this cycle: snapshots/reference_solution_{k}.nc
            truth_path = os.path.join(self.nm.path, "snapshots", f"reference_solution_{k}.nc")
            with NC(truth_path, "r") as ds_ref:
                arr = ds_ref[var_name][:]
                if var_name.startswith("TRG") and arr.ndim == 4:
                    arr = arr[0, ...]
                grid_truth = (arr if var_name.startswith("PSG") else arr[level, :, :]).astype(float)

            # Optional: load NoDA grid for this cycle: free_run/free_run_{k}.nc
            grid_noda = None
            noda_path = os.path.join(self.nm.path, "free_run", f"free_run_{k}.nc")
            if os.path.exists(noda_path):
                with NC(noda_path, "r") as ds_noda:
                    arr = ds_noda[var_name][:]
                    if var_name.startswith("TRG") and arr.ndim == 4:
                        arr = arr[0, ...]
                    grid_noda = (arr if var_name.startswith("PSG") else arr[level, :, :]).astype(float)

            out_dir = getattr(self.nm, "path", ".")
            out_csv = os.path.join(out_dir, f"{var_name}_lev{level}_values_cycle{k}.csv")
            write_grid_values_csv(var_name, level, grid_bkg, grid_ana, grid_truth, grid_noda, out_csv)
        except Exception as e:
            print(f"[grid-values] {var_name}@lev{level} cycle={locals().get('k','?')} skipped: {e}")
# --------------------------------------------------------------------

        self.map_vector_states()






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
          



    
    
    
    
        
    
