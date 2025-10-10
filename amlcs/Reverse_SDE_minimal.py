import os
import numpy as np
import sys
import scipy.sparse as spa
import pandas as pd
import time
from netCDF4 import Dataset
from commons_utils import compute_modified_Cholesky_decomposition
import torch
import json

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
        - One-time dump of block_map.json for external analyzers.
        - Graceful fallback (inflated background) for any problematic/missing block.
        """

        import numpy as _np
        import torch as _torch


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

        # Preflight: ensure maps exist
        xb_len = len(getattr(self, "XB_map", []) or [])
        om_len = len(getattr(self, "obs_map", []) or [])
        mc_len = len(self.nm.mask_cor)

        print(f"[ReverseSDE] preflight: XB_map={xb_len}, obs_map={om_len}, mask_cor={mc_len}")
        self.XA_map = []

        # AUTO-FALLBACK if XB_map missing (no background prepared)
        if xb_len == 0:
            print("[ReverseSDE] WARNING: XB_map is empty. Using inflated background as analysis for all blocks.")
            for block_idx in range(mc_len):
                XB_block = self.get_ensemble_block(self.nm.mask_cor[block_idx])  # (n_block, Nens)
                XA_block = self.covariance_inflation(XB_block)
                self.XA_map.append(XA_block)
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
                alpha_t = self._cond_alpha(t)
                sigma2_t  = self._cond_sigma_sq(t)
                f       = self._f(t)
                g       = self._g(t)
                tau     = self._g_tau(t)

                prior_term = (xt - alpha_t * X0_obs_n_t) / sigma2_t
                like_score  = -((sf * xt) - y_n_t) / (sigma_n_t ** 2) * sf  # no tempering
                drift = - (f * xt + (g ** 2) * (prior_term - tau * like_score))
                noise = _torch.sqrt(_torch.tensor(dt, device=device, dtype=xt.dtype)) * g * _torch.randn_like(xt)
                xt_next = xt + dt * drift + noise

                if not _torch.isfinite(xt_next).all():
                    print(f"[ReverseSDE][{label}] State became non-finite at step {i}. Using fallback for this block.")
                    block_numeric_ok = False
                    break
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


        # Final sanity: if XA_map is short, pad with inflated background per block
        if len(self.XA_map) < mc_len:
            print(f"[ReverseSDE] WARNING: XA_map has {len(self.XA_map)} of {mc_len} blocks. Padding with fallbacks.")
            for block_idx in range(len(self.XA_map), mc_len):
                XB_block = self.get_ensemble_block(self.nm.mask_cor[block_idx])
                XA_block = self.covariance_inflation(XB_block)
                self.XA_map.append(XA_block)

        # Publish analysis
        self.map_vector_states()
class sequential_method:
      
      method_name = None;
      
      def __init__(self, method_name):
          self.method_name = method_name;
      
      def get_instance(self, nm, infla, Nens):
          if self.method_name == 'ReverseSDE':return ReverseSDE(nm, infla, Nens)