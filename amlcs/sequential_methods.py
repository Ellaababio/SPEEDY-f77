import os
import numpy as np
import sys
import scipy.sparse as spa
import pandas as pd
import time
import json
from netCDF4 import Dataset
from commons_utils import compute_modified_Cholesky_decomposition
import torch
from netCDF4 import Dataset as NC


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
        
    

    
    def get_block_indices(self, block_idx):
        """
        Identify indices of variables within a block.
        Returns a dict with boolean masks or index arrays for U, V, WDG, WSG.
        Also returns the 'core' state mask (excluding WDG/WSG).
        """
        mask_cor = self.nm.mask_cor[block_idx]
        var_names = self.nm.var_names
        
        # Calculate total size of block and start/end indices for each var
        idx_map = {} # Initialize empty to capture all variables
        
        # Ensure we specifically track these for WDG/WSG logic later if needed, 
        # though dynamic addition covers them if they exist in the block.
        # But let's initialize known empty lists for WDG1/WSG1 just in case 
        # the consumer expects them to exist even if empty? 
        # Looking at consumer code: `wdg_indices = idx_info.get('WDG1', [])` - usage uses .get(), so safe.
        
        all_indices = []
        current_idx = 0
        
        # We need to map linear indices in XB_block back to variables
        # XB_block stack variables vertically.
        # But wait, get_ensemble_block flattens them?
        # get_ensemble_block: concatenate((XB_block, XB_v), axis=0)
        # So variables are stacked sequentially.
        
        for v in mask_cor:
            v_info = v[0]
            var_idx = v_info[0]
            var_name = var_names[var_idx]
            
            # Dimensions
            v_reso = v[1] # lat, lon
            n_points = v_reso[0] * v_reso[1]
            
            indices = list(range(current_idx, current_idx + n_points))
            
            if var_name not in idx_map:
                idx_map[var_name] = []
            idx_map[var_name].extend(indices)
                
            current_idx += n_points
            
        # Create masks/arrays
        res = {}
        for key in idx_map:
            res[key] = np.array(idx_map[key], dtype=int)
            
        # Core State Mask (Exclude WDG1, WSG1)
        # Assuming we want to exclude them from the Update/Diffusion State
        # If they are not in the block, these lists are empty.
        total_size = current_idx
        is_virtual = np.zeros(total_size, dtype=bool)
        if 'WDG1' in res and len(res['WDG1']) > 0: is_virtual[res['WDG1']] = True
        if 'WSG1' in res and len(res['WSG1']) > 0: is_virtual[res['WSG1']] = True
        
        res['core_mask'] = ~is_virtual
        res['is_wdg_mask'] = is_virtual # Or specific masks
        
        return res

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

    # ------------------ unified NetCDF writer (moved from EnKF to base) ------------------
    def _write_unified_nc_block(self, block, H_block=None, R_block=None, XB_block=None, XA_block=None, cycle_k=None, obs_info=None):
        """
        HPC-safe NetCDF writer (single .nc per cycle).
        Handles both EnKF (H/R) and ReverseSDE (obs_info) inputs.
        """
        import numpy as np
        from netCDF4 import Dataset

        # Resolve cycle index k
        k = cycle_k
        if k is None:
            if hasattr(self, '_cycle_k'): k = self._cycle_k
            elif hasattr(self, 'current_cycle_k'): k = self.current_cycle_k
        
        if k is None:
            print("[write_unified_nc_block] Warning: cycle_k could not be determined. Skipping write.")
            return

        # ensemble means
        xb_mean_full = XB_block.mean(axis=1)
        xa_mean_full = XA_block.mean(axis=1)

        # Helper to extract obs details
        obs_idx_block = np.array([], dtype=int)
        obs_vals = np.array([])
        sigma_vec = np.array([])
        
        # METHOD A: EnKF style (H sparse matrix provided)
        if H_block is not None and R_block is not None:
            Hc = H_block.tocoo()
            order = np.argsort(Hc.row)
            obs_idx_block = Hc.col[order].astype(int)

            # unperturbed obs
            # check self.Y_unp existence
            if hasattr(self, 'Y_unp') and len(self.Y_unp) > block:
                y_unp = self.Y_unp[block].reshape(-1)
                obs_vals = y_unp[: obs_idx_block.size]
            
            # sigma
            try:
                R_diag = R_block.diagonal()
            except:
                R_diag = np.array(R_block.todense()).diagonal()
            sigma_vec = np.sqrt(np.asarray(R_diag)).reshape(-1)[: obs_idx_block.size]

        # METHOD B: ReverseSDE style (obs_info dict provided)
        elif obs_info is not None:
             # obs_info = {'idx_observed': ..., 'y': ..., 'sigma': ...}
             obs_idx_block = obs_info.get('idx_observed', np.array([]))
             obs_vals = obs_info.get('y', np.array([]))
             sigma_vec = obs_info.get('sigma', np.array([]))
        
        # load truth/noDA
        ref_nc = os.path.join(self.nm.snapshots, f"reference_solution_{k}.nc")
        fru_nc = os.path.join(self.nm.free_run,    f"free_run_{k}.nc")
        X_ref = self.nm.load_netcdf_file(ref_nc)
        X_nod = self.nm.load_netcdf_file(fru_nc)

        # output file
        out_dir = self.nm.path
        os.makedirs(out_dir, exist_ok=True)
        nc_path = os.path.join(out_dir, f"unified_cycle{k}.nc")

        # open NC file
        if os.path.exists(nc_path):
            ds = Dataset(nc_path, "a")
        else:
            lat, lon = self.nm.gs.get_resolution(self.nm.res)
            ds = Dataset(nc_path, "w", format="NETCDF4")
            ds.createDimension("lat", lat)
            ds.createDimension("lon", lon)
            ds.cycle = int(k)

        # helper: safe var creation
        def _get_or_create_var(prefix, var_name, lev_tag, dtype="f8"):
            vname = f"{prefix}_{var_name}_{lev_tag}"
            if vname in ds.variables:
                return ds.variables[vname]
            return ds.createVariable(vname, dtype, ("lat", "lon"))

        # iterate block slices
        offset = 0
        for (v_info, res) in self.nm.mask_cor[block]:
            v_idx, lev = v_info
            lat, lon = res
            n = lat * lon
            start, end = offset, offset + n
            offset = end

            var_name = self.nm.var_names[v_idx]
            lev_tag = f"lev{lev}"

            # background / analysis means
            xb = xb_mean_full[start:end].astype(float).reshape(lat, lon)
            xa = xa_mean_full[start:end].astype(float).reshape(lat, lon)

            # truth / noda
            if "PSG" in var_name:
                tr2d = X_ref[v_idx][:, :]
                nd2d = X_nod[v_idx][:, :]
            else:
                tr2d = X_ref[v_idx][lev, :, :]
                nd2d = X_nod[v_idx][lev, :, :]
            tr = tr2d.astype(float)
            nd = nd2d.astype(float)

            # obs fields
            obs = np.full(n, np.nan)
            sig = np.full(n, np.nan)
            iso = np.zeros(n, dtype="i4")

            sel = (obs_idx_block >= start) & (obs_idx_block < end)
            if sel.any():
                local = obs_idx_block[sel] - start
                # Safety for extracted values
                if sel.sum() <= len(obs_vals) and sel.sum() <= len(sigma_vec):
                     obs[local] = obs_vals[sel] if len(obs_vals) > 0 else np.nan
                     sig[local] = sigma_vec[sel] if len(sigma_vec) > 0 else np.nan
                     iso[local] = 1

            obs_2d = obs.reshape(lat, lon)
            sig_2d = sig.reshape(lat, lon)
            iso_2d = iso.reshape(lat, lon)
            idx_2d = np.arange(n, dtype="i4").reshape(lat, lon)

            # write vars
            _get_or_create_var("idx",     var_name, lev_tag, "i4")[:, :] = idx_2d
            _get_or_create_var("xb_mean", var_name, lev_tag)[:, :] = xb
            _get_or_create_var("xa_mean", var_name, lev_tag)[:, :] = xa
            _get_or_create_var("truth",   var_name, lev_tag)[:, :] = tr
            _get_or_create_var("noda",    var_name, lev_tag)[:, :] = nd
            _get_or_create_var("obs",     var_name, lev_tag)[:, :] = obs_2d
            _get_or_create_var("sigma",   var_name, lev_tag)[:, :] = sig_2d
            _get_or_create_var("is_obs",  var_name, lev_tag, "i4")[:, :] = iso_2d

        ds.close()

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
    def __init__(self, nm, infla, Nens, nonlinear_obs=False, scalefact=1.0, wind_err=None, nonlinear_operator_type='arctan'):
        super().__init__(nm, infla, Nens)
        self.nonlinear_obs = bool(nonlinear_obs)
        self.scalefact = float(scalefact)
        self.wind_err = wind_err if wind_err is not None else {}
        self.nonlinear_operator_type = str(nonlinear_operator_type)
            
    
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
        # store unperturbed obs for each block (same source ReverseSDE used)
        self.Y_unp = [ob.y_obs[k][block] for block in range(0, N_blocks)]
        self._cycle_k = k
    
    def perform_assimilation(self, ob):
        self.XA_map = [];
        for block, (XB_info, Ys_block, R_info, H_block) in enumerate(zip(self.XB_map, self.Ys, ob.obs_R_sparse, ob.obs_H_sparse)):
            XB_block = XB_info['XB_b'];
            R_block  = R_info['R'];
            Binv_sqrt_block = XB_info['Binv_s_b'];
            XA_block = self.perform_assimilation_block(XB_block, Binv_sqrt_block, H_block, R_block, Ys_block, block, ob);
            XA_block = self.covariance_inflation(XA_block);
            self.XA_map.append(XA_block);
            # write unified Netcdf files for this block/cycle
            self._write_unified_nc_block(block, H_block, R_block, XB_block, XA_block)
        self.map_vector_states(); #Update ensemble folders
    
    def perform_assimilation_block(self, XB, Binv_sqrt, H, R, Ys, block_idx, ob=None):
      
          # 1. Compute linear predicted observations
          Hb_X = H @ XB
          
          # --- Wind Assimilation Logic ---
          is_wind_update = False
          wind_mode = None
          new_Ys, new_Hb_X, new_H_rows, new_R_diag = [], [], [], []
          new_is_wdg, new_sigma = [], []
          
          if ob is not None and hasattr(ob, 'wind_obs') and len(self.nm.mask_cor[block_idx]) > 0:
               # Identify Block Variable
               v_inf = self.nm.mask_cor[block_idx][0][0]
               var_name = self.nm.var_names[v_inf[0]]
               level = v_inf[1]
               
               if var_name == 'UG1': 
                   wind_mode = 'u'; partner_idx = block_idx + 1 
               elif var_name == 'VG1': 
                   wind_mode = 'v'; partner_idx = block_idx - 1
               
               # If Wind Block, fetch and process obs
               if wind_mode:
                   # Fetch Partner State (Background)
                   if 0 <= partner_idx < len(self.XB_map):
                       XB_partner = self.XB_map[partner_idx]['XB_b']

                       # Indices in this block
                       idx_info = self.get_block_indices(block_idx)
                       my_indices = idx_info.get(var_name, [])
                       
                       # Helper to process Wind Obs Type
                       def process_wind_type(type_key, calc_func, grad_func, sigma_val, is_circular):
                           if type_key in ob.wind_obs and level in ob.wind_obs[type_key]:
                               # Check cycle availability
                               k = getattr(self, '_cycle_k', 0)
                               if k < len(ob.y_wind_obs):
                                   d_dict = ob.y_wind_obs[k].get(type_key, {}).get(level)
                                   if d_dict is not None:
                                       obs_indices = ob.wind_obs[type_key][level]['stations']
                                       obs_vals = d_dict.flatten()
                                       
                                       # Match Obs to Grid
                                       mask_in_block = np.isin(obs_indices, my_indices)
                                       if not np.any(mask_in_block): return
                                       
                                       valid_obs_idx = np.where(mask_in_block)[0]
                                       
                                       for i in valid_obs_idx:
                                           grid_idx = obs_indices[i]
                                           local_idx = np.where(my_indices == grid_idx)[0][0]
                                           
                                           # States
                                           val_my = XB[local_idx, :]
                                           val_partner = XB_partner[local_idx, :]
                                           u_vec = val_my if wind_mode == 'u' else val_partner
                                           v_vec = val_partner if wind_mode == 'u' else val_my
                                           
                                           # Predict
                                           pred = calc_func(u_vec, v_vec)
                                           
                                           # Grading (Linearization)
                                           u_m, v_m = u_vec.mean(), v_vec.mean()
                                           grad = grad_func(u_m, v_m, wind_mode)
                                           
                                           # Append
                                           new_Ys.append(obs_vals[i])
                                           new_Hb_X.append(pred)
                                           new_R_diag.append(sigma_val**2) # Variance
                                           # Track if this row is WDG for circular diff later
                                           new_is_wdg.append(is_circular)
                                           new_sigma.append(sigma_val)
                                           
                                           # H row: sparse entry at local_idx
                                           new_H_rows.append((local_idx, grad))

                       # Define Physics
                       def calc_wdg(u,v): return np.arctan2(u,v)
                       def calc_wsg(u,v): return np.sqrt(u**2 + v**2)
                       
                       def grad_wdg(u,v,mode):
                           s2 = max(u**2+v**2, 1e-6)
                           return v/s2 if mode=='u' else -u/s2
                       
                       def grad_wsg(u,v,mode):
                           sm = max(np.sqrt(u**2+v**2), 1e-6)
                           return u/sm if mode=='u' else v/sm

                       # Process with configured sigmas
                       # Default to tuned values if not in config
                       sigma_wdg = self.wind_err.get('WDG1', 0.2)
                       sigma_wsg = self.wind_err.get('WSG1', 1.0)
                       
                       process_wind_type('WDG1', calc_wdg, grad_wdg, sigma_wdg, True)
                       process_wind_type('WSG1', calc_wsg, grad_wsg, sigma_wsg, False)

          # Augment Matrices if new obs found
          n_std = Ys.shape[0] # Boundary between standard and wind
          if new_Ys:
              # Perturb Wind Obs to match Ensemble Size (Nens)
              n_new = len(new_Ys)
              y_wind_mean = np.array(new_Ys).reshape(-1, 1) # (n_new, 1)
              
              # Generate perturbations vector
              sigmas = np.array(new_sigma).reshape(-1, 1)
              perturbations = np.random.normal(0, sigmas, size=(n_new, self.Nens))
              
              # Add noise to observations
              Ys_wind = y_wind_mean + perturbations
              
              # Ys
              Ys = np.vstack([Ys, Ys_wind])
              
              # Hb_X
              Hb_X_wind = np.array(new_Hb_X) # (n_new, Nens)
              Hb_X = np.vstack([Hb_X, Hb_X_wind])
              
              # R
              R_ext = spa.diags(new_R_diag)
              R = spa.block_diag((R, R_ext))
              
              # H
              h_data = [x[1] for x in new_H_rows]
              h_rows = np.arange(len(new_H_rows))
              h_cols = [x[0] for x in new_H_rows]
              H_ext = spa.csr_matrix((h_data, (h_rows, h_cols)), shape=(len(new_Ys), XB.shape[0]))
              H = spa.vstack((H, H_ext))

          
          # 2. Handle Nonlinear/Normalization Logic
          H_spar = H.toarray()
          Ds = Ys - Hb_X # Default innovation
          
          if self.nonlinear_obs:
              sf = self.scalefact
              
              # Split Standard vs Wind (Wind already handled with its own nonlinear operators)
              Hb_std = Hb_X[:n_std, :]
              Ys_std = Ys[:n_std, :]
              
              # Apply nonlinear forward operator on ensemble predictions
              # Observations are already in h-space (generated in observation.py)
              if self.nonlinear_operator_type == 'arctan_sq':
                  Hb_final_std = np.arctan((sf * Hb_std) ** 2)
              else:
                  # Default: arctan(sf * Hx)
                  Hb_final_std = np.arctan(sf * Hb_std)
              
              Ds_std = Ys_std - Hb_final_std
              
              # Re-assemble Ds with Wind Part
              if new_Ys:
                  # Wind Innovations
                  Ds_wind = Ys[n_std:, :] - Hb_X[n_std:, :]
                  
                  # Apply Circular Innovation Correction for WDG
                  is_wdg_arr = np.array(new_is_wdg, dtype=bool)
                  if np.any(is_wdg_arr):
                        # Force diff to [-pi, pi]
                        diffs = Ds_wind[is_wdg_arr, :]
                        diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
                        Ds_wind[is_wdg_arr, :] = diffs
                  
                  Ds = np.vstack([Ds_std, Ds_wind])
              else:
                  Ds = Ds_std
          else:
              # Linear Case - Still need circular correction
              if new_Ys:
                  is_wdg_arr = np.array(new_is_wdg, dtype=bool)
                  if np.any(is_wdg_arr):
                      # We need to offset indices by n_std
                      Ds_wind = Ds[n_std:, :]
                      wdg_subset = Ds_wind[is_wdg_arr, :]
                      wdg_subset = (wdg_subset + np.pi) % (2 * np.pi) - np.pi
                      Ds[n_std:, :][is_wdg_arr, :] = wdg_subset

          
          # 3. Standard EnKF Update
          P = spa.linalg.spsolve_triangular(Binv_sqrt, H_spar.T, lower=False);
          Inno = R + P.T @ P;
          Q_temp = P @ spa.linalg.spsolve(Inno, Ds);
          DXa = spa.linalg.spsolve_triangular(Binv_sqrt.T, Q_temp, lower=True);
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
            
    def __init__(self, nm, infla, Nens, nonlinear_obs=False, scalefact=1.0, wind_nonlinear_operator=False, wind_err=None, normalize_nonlinear=True, nonlinear_operator_type='arctan'):
        super().__init__(nm, infla, Nens)
        self.nonlinear_obs = bool(nonlinear_obs)
        self.scalefact = float(scalefact)
        self.wind_nonlinear_operator = wind_nonlinear_operator
        self.wind_err = wind_err if wind_err is not None else {}
        self.normalize_nonlinear = bool(normalize_nonlinear)
        self.nonlinear_operator_type = str(nonlinear_operator_type)
        
    
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
        self.current_k = k
        self._cycle_k = k
        self.ob = ob
        Nens = self.Nens;
        mask_cor = self.nm.mask_cor;
        N_blocks = len(mask_cor);
        for block in range(0, N_blocks):
            self.y.append(ob.y_obs[k][block]); 
            
        self.Y_unp = [ob.y_obs[k][block] for block in range(0, N_blocks)]
        self.nm.gs.compute_local_boxobs(ob.obs_H_sparse);
    
    def perform_assimilation(self, ob):
        self.XA_map = [];
        for block_idx, (XB_info, y_block, R_info, H_block, lobs_block) in enumerate(zip(self.XB_map, self.y, ob.obs_R_sparse, ob.obs_H_sparse, self.nm.gs.lbo_obs)):
            XB_block = XB_info['XB_b'];
            xb_block = XB_info['xb_b'];
            DX_block = XB_info['DX_b'];
            lbo_block = XB_info['lbo_b'];
            Ri_block  = R_info['Ri'];
            R_block   = R_info['R']
            mu_vec    = R_info.get('mu_vec')
            std_vec   = R_info.get('std_vec')
            XA_block = self.perform_assimilation_block(XB_block, xb_block, DX_block, H_block, Ri_block, y_block, lbo_block, lobs_block, block_idx=block_idx, mu_vec=mu_vec, std_vec=std_vec);
            self.XA_map.append(XA_block);
            
            # write unified Netcdf files for this block/cycle
            self._write_unified_nc_block(block_idx, H_block, R_block, XB_block, XA_block)
            
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
        
        
    
    def perform_assimilation_block(self, XB, xb, DX, H, Ri, y, lbo_info, lobs, block_idx=None, mu_vec=None, std_vec=None):

          
          n, Nens = XB.shape;
          
          lbo, nlbo = lbo_info; #local boxes information (indexes, number of components in all boxes)

          y_model = H.T @ y;
          mu_model = H.T @ mu_vec if mu_vec is not None else None
          std_model = H.T @ std_vec if std_vec is not None else None
          
          
          Ri_space = H.T @ (Ri @ H);
          Ri_space = Ri_space.toarray();
          
          XA = np.zeros((n, Nens));
          
          # Nonlinear Wind Setup
          ob = getattr(self, 'ob', None)
          k_step = getattr(self, 'current_k', 0)
          
          block_var_name = None
          block_level = None
          is_wind_update = False
          wind_mode = None
          
          # Identify if this block is UG1 or VG1
          if block_idx is not None and len(self.nm.mask_cor) > block_idx and len(self.nm.mask_cor[block_idx]) > 0:
               v_inf, _ = self.nm.mask_cor[block_idx][0]
               block_var_idx, block_level = v_inf
               block_var_name = self.nm.var_names[block_var_idx]
          
          if getattr(self, 'wind_nonlinear_operator', False):
               if block_var_name == 'UG1': wind_mode = 'u'
               elif block_var_name == 'VG1': wind_mode = 'v'
               
               if wind_mode and ob is not None:
                    # Partner block logic
                    partner_block_idx = block_idx + 1 if wind_mode == 'u' else block_idx - 1
                    if 0 <= partner_block_idx < len(self.XB_map):
                         partner_info = self.XB_map[partner_block_idx]
                         partner_XB = partner_info['XB_b'] # Full block for partner
                         is_wind_update = True
          
          for i in range(0, n):
              #local box for model component i
              lbo_i = np.array(lbo[i]).astype('int32');
              gp_i, = np.where(lbo_i==i);
              xb_i = xb[lbo_i];
              XB_i = XB[lbo_i];
              DX_i = DX[lbo_i];
              
              #local observation operator (Linear)
              H_ind = np.array(lobs[i]).astype('int32'); #local observed components
              m_i = H_ind.size;
              
              n_i = xb_i.size;
              
              has_standard_obs = (m_i > 0)
              y_i_lin, H_i_lin, Ri_i_lin = None, None, None
              
              if has_standard_obs:
                 I = np.arange(0, m_i);
                 J = H_ind;
                 H_i_lin = spa.coo_matrix((np.ones(m_i),(I,J)), shape=(m_i, n_i));
                 y_i_lin = y_model[lbo_i[H_ind]];
                 Ri_i_lin = np.diag(Ri_space[lbo_i[H_ind], lbo_i[H_ind]]).reshape((m_i, m_i)); #local data error covariance matrix

                 if self.nonlinear_obs:
                     sf = self.scalefact
                     # Apply nonlinear forward operator h(x) = arctan(sf * Hx) on ensemble
                     Hb_i_ens = H_i_lin @ XB_i  # (m_i, Nens)
                     
                     if self.normalize_nonlinear and mu_model is not None and std_model is not None:
                         mu_i = mu_model[lbo_i[H_ind]].reshape((m_i, 1))
                         std_i = std_model[lbo_i[H_ind]].reshape((m_i, 1))
                         Hb_i_ens = (Hb_i_ens - mu_i) / std_i
                         
                     if self.nonlinear_operator_type == 'arctan_sq':
                         Hb_i_ens_nl = np.arctan((sf * Hb_i_ens) ** 2)
                     else:
                         Hb_i_ens_nl = np.arctan(sf * Hb_i_ens)
                     Hb_i_ens_nl_mean = np.mean(Hb_i_ens_nl, axis=1, keepdims=True)

              
              # Determine if we should dynamically add nonlinear wind obs
              wdg_vals, wsg_vals = [], []
              wdg_sig, wsg_sig = [], []
              wdg_pred, wsg_pred = [], []
              
              if is_wind_update:
                  partner_xb_i = partner_info['xb_b'][lbo_i]
                  partner_DX_i = partner_info['DX_b'][lbo_i]
                  
                  # u and v background states at the gridpoints (n_i, 1) and (n_i, Nens)
                  if wind_mode == 'u':
                      u_mean, v_mean = xb_i, partner_xb_i
                      u_pert, v_pert = DX_i, partner_DX_i
                  else:
                      u_mean, v_mean = partner_xb_i, xb_i
                      u_pert, v_pert = partner_DX_i, DX_i
                      
                  u_ens = u_mean + u_pert
                  v_ens = v_mean + v_pert
                  
                  # Find which local points (0 to n_i-1) actually have wind stations
                  def get_wind_local(wname):
                      local_obs_vals = []
                      local_obs_sig = []
                      local_pred_ens = []
                      
                      if wname in ob.wind_obs and block_level in ob.wind_obs[wname]:
                          meta = ob.wind_obs[wname][block_level]
                          if k_step < len(ob.y_wind_obs) and wname in ob.y_wind_obs[k_step]:
                              data = ob.y_wind_obs[k_step][wname].get(block_level)
                              if data is not None:
                                  glob_stations = np.array(meta['stations']) # Stations globally
                                  data_flat = data.flatten()
                                  err_std = self.wind_err.get(wname, 1.0)
                                  
                                  # Match lbo_i against glob_stations
                                  for local_idx, global_point in enumerate(lbo_i):
                                      pos = np.where(glob_stations == global_point)[0]
                                      if pos.size > 0:
                                          val = data_flat[pos[0]]
                                          local_obs_vals.append(val)
                                          local_obs_sig.append(err_std)
                                          
                                          # Ensemble predictions at this local point
                                          u_e = u_ens[local_idx, :]
                                          v_e = v_ens[local_idx, :]
                                          if wname == 'WDG1':
                                              pred = np.arctan2(u_e, v_e) # (user wants arctan2(u,v))
                                          else: # WSG1
                                              pred = np.sqrt(u_e**2 + v_e**2)
                                              
                                          local_pred_ens.append(pred)
                                          
                      return np.array(local_obs_vals), np.array(local_obs_sig), np.array(local_pred_ens)
                  
                  wdg_vals, wdg_sig, wdg_pred = get_wind_local('WDG1')
                  wsg_vals, wsg_sig, wsg_pred = get_wind_local('WSG1')

              nonlinear_m = len(wdg_vals) + len(wsg_vals)
              
              if has_standard_obs or nonlinear_m > 0:
                  
                  total_m = m_i + nonlinear_m
                  
                  # 1. Standard obs part
                  if has_standard_obs:
                      if self.nonlinear_obs:
                          # Nonlinear forward operator: h(x) = arctan(sf * Hx)
                          # Observations already in h-space from observation.py
                          Yb_lin = Hb_i_ens_nl - Hb_i_ens_nl_mean  # (m_i, Nens)
                          d_lin = y_i_lin - Hb_i_ens_nl_mean  # (m_i, 1)
                      else:
                          Yb_lin = H_i_lin @ DX_i  # (m_i, Nens)
                          d_lin = y_i_lin - H_i_lin @ xb_i # (m_i, 1)
                  else:
                      Yb_lin = np.empty((0, Nens))
                      d_lin = np.empty((0, 1))
                      Ri_i_lin = np.empty((0, 0))
                      
                  # 2. Nonlinear part
                  Yb_non = []
                  d_non = []
                  Ri_non_diag = []
                  
                  def add_nonlinear_obs(y_real, y_pred_ens, sig, wname_current):
                      for k in range(len(y_real)):
                          y_pred_mean = np.mean(y_pred_ens[k, :])
                          yb_row = y_pred_ens[k, :] - y_pred_mean
                          
                          if 'WDG' in wname_current:
                              yb_row = (yb_row + np.pi) % (2 * np.pi) - np.pi
                          
                          Yb_non.append(yb_row)
                          
                          perturbed_y = y_real[k] + sig[k] * np.random.randn()
                          innov = perturbed_y - y_pred_mean
                          if 'WDG' in wname_current:
                              innov = (innov + np.pi) % (2 * np.pi) - np.pi
                          
                          d_non.append([innov])
                          Ri_non_diag.append(1.0 / (sig[k]**2))

                  if len(wdg_vals) > 0:
                      add_nonlinear_obs(wdg_vals, wdg_pred, wdg_sig, 'WDG')
                  
                  if len(wsg_vals) > 0:
                      add_nonlinear_obs(wsg_vals, wsg_pred, wsg_sig, 'WSG')
                      
                  if nonlinear_m > 0:
                      Yb_non_arr = np.array(Yb_non)
                      d_non_arr = np.array(d_non)
                      Ri_non_arr = np.diag(Ri_non_diag)
                  else:
                      Yb_non_arr = np.empty((0, Nens))
                      d_non_arr = np.empty((0, 1))
                      Ri_non_arr = np.empty((0, 0))
                      
                  # 3. Combine Linear and Nonlinear
                  Yb_total = np.vstack([Yb_lin, Yb_non_arr]) if has_standard_obs and nonlinear_m > 0 else (Yb_lin if has_standard_obs else Yb_non_arr)
                  d_total = np.vstack([d_lin, d_non_arr]) if has_standard_obs and nonlinear_m > 0 else (d_lin if has_standard_obs else d_non_arr)
                  
                  Ri_total = np.zeros((total_m, total_m))
                  if has_standard_obs: Ri_total[:m_i, :m_i] = Ri_i_lin
                  if nonlinear_m > 0: Ri_total[m_i:, m_i:] = Ri_non_arr
                  
                  # 4. Local Letkf SVD update (inlining perform_assimilation_local_box logic)
                  Pa_Nens = (Nens-1)*np.eye(Nens) + Yb_total.T @ ( Ri_total @ Yb_total );
                  Q_temp = Yb_total.T @ (Ri_total @ d_total);
                  
                  U, S, _ = np.linalg.svd(Pa_Nens, full_matrices=False);
                  
                  Pa_sqrt = U @ ( np.diag(np.sqrt(Nens/S)) @ U.T );
                  Pa_invs = U @ ( np.diag(1/S) @ U.T );
                  
                  wa = Pa_invs @ Q_temp;
                  xa_i = xb_i + DX_i @ wa;
                  XA_i = xa_i + DX_i @ Pa_sqrt;

              else:
                 XA_i = XB_i;
                 
              XA[i, :] = XA_i[gp_i, :]; 
          
          return XA;
          



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
                 pseudo_time_steps: int = 1000,
                 eps_alpha: float = 0.05, # keep between 0 and 1
                 scalefact: float = 1.0,
                 eps_beta: float = 0.025, # keep between 0 and 0.5
                 nonlinear_obs: bool = False,
                 normalize: bool = True,
                 drift_type: str = "old",
                 enable_early_stopping: bool = True,
                 score_clip: float = 10000.0,
                 state_clip: float = 20.0,
                 rng_seed: int = 42,
                 track_gridpoint_locs: list = None,
                 wind_err=None,
                 nonlinear_operator_type: str = 'arctan'):
        super().__init__(nm, infla, Nens)
        self.p_time_step = int(pseudo_time_steps)
        self.eps_alpha = float(eps_alpha)
        self.scalefact = float(scalefact)
        self.eps_beta = float(eps_beta)
        self.nonlinear_obs = bool(nonlinear_obs)
        self.normalize = bool(normalize)
        self.nonlinear_operator_type = str(nonlinear_operator_type)
        self.drift_type = drift_type
        self.enable_early_stopping = bool(enable_early_stopping)
        self.enable_early_stopping = bool(enable_early_stopping)
        self.score_clip = float(score_clip) if score_clip is not None else None
        self.state_clip = float(state_clip) if state_clip is not None else None
        self.rng_seed = int(rng_seed)
        self.track_gridpoint_locs = track_gridpoint_locs if track_gridpoint_locs is not None else [(8, 31), (24, 36)]
        self.rng = np.random.RandomState(self.rng_seed)
        self.wind_err = wind_err if wind_err is not None else {}
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
        return self.eps_beta + (1.0 - self.eps_beta) * t


    def _f(self, t):
        # f = d(log alpha)/dt
        alpha_t = self._cond_alpha(t)
        return -(1.0 - self.eps_alpha) / alpha_t

    def _g(self, t):
        # d(sigma^2)/dt - 2 f sigma^2 = g^2
        d_sigma_sq_dt = (1.0 - self.eps_beta)        # derivative of the schedule above
        g2 = d_sigma_sq_dt - 2.0 * self._f(t) * self._cond_sigma_sq(t)
        return np.sqrt(max(0.0, g2))

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
        self.current_cycle_k = int(k)  # remember k for file names

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
                "mu_vec": R_info.get("mu_vec"),
                "std_vec": R_info.get("std_vec"),
            })
        # --- inside prepare_analysis(), right before that debug print ---
        # ensure idx_observed is an ndarray for all blocks
        for b in self.obs_map:
            if not isinstance(b.get('idx_observed', None), np.ndarray):
                b['idx_observed'] = np.array([], dtype=int)

        non_empty = next((b for b in self.obs_map if b['idx_observed'].size > 0), None)
        if non_empty is not None:
            i0 = int(non_empty['idx_observed'].min())
            i1 = int(non_empty['idx_observed'].max())
            print(f"[debug] first obs_map block idx range: {i0}–{i1} "
                f"(m={non_empty['idx_observed'].size}, vars={non_empty.get('vars')})")
        
        elif hasattr(ob, 'wind_obs') and ob.wind_obs: 
            # Allow proceed if we have Wind Obs (WDG1/WSG1) even if no standard obs
            print(f"[debug] No standard observations found, but wind_obs detected: {list(ob.wind_obs.keys())}. Proceeding.")
        
        else:
            raise RuntimeError("No observations selected after applying obs_plc—"
                            "check obs_plc and time-level filters (TG0 vs TG1).")


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
        import numpy as _np
        import torch as _torch

        # -------- CONFIG --------
        self._cycle_idx = getattr(self, "_cycle_idx", 0) + 1
        DEBUG_EVERY = 20
        # ------------------------
        # ------------------------
        cycle_k = int(getattr(self, "current_cycle_k", 0))  # from prepare_analysis

        track_file = os.path.join(self.nm.path, "sde_tracking.nc")
        psteps = int(getattr(self, "p_time_step", 200))

        if cycle_k == 0:
            nc_init = Dataset(track_file, "w")
            nc_init.createDimension("cycle", None)
            nc_init.createDimension("block", None)
            nc_init.createDimension("psteps", psteps)
            nc_init.createDimension("var", 5)
            nc_init.createDimension("pts", len(self.track_gridpoint_locs))
            nc_init.createDimension("ens", None)

            xt_state_mean = nc_init.createVariable(
                "xt_state_mean", "f4",
                ("cycle", "block", "psteps", "var", "ens"),
                zlib=True
            )
            xt_state_gridpoint = nc_init.createVariable(
                "xt_state_gridpoint", "f4",
                ("cycle", "block", "psteps", "var", "pts", "ens"),
                zlib=True
            )
            xt_norm_mean = nc_init.createVariable(
                "xt_norm_mean", "f4",
                ("cycle", "block", "psteps", "var", "ens"),
                zlib=True
            )
            xt_norm_gridpoint = nc_init.createVariable(
                "xt_norm_gridpoint", "f4",
                ("cycle", "block", "psteps", "var", "pts", "ens"),
                zlib=True
            )
            xt_force_prior_gridpoint = nc_init.createVariable(
                "xt_force_prior_gridpoint", "f4",
                ("cycle", "block", "psteps", "var", "pts", "ens"),
                zlib=True
            )
            xt_force_like_gridpoint = nc_init.createVariable(
                "xt_force_like_gridpoint", "f4",
                ("cycle", "block", "psteps", "var", "pts", "ens"),
                zlib=True
            )

            var_names = nc_init.createVariable("var_names", str, ("var",))
            var_list = ["UG1", "VG1", "TG1", "TRG1", "PSG1"]
            for ii, name in enumerate(var_list):
                var_names[ii] = name

            nc_init.close()

        nc = Dataset(track_file, "a")
        xt_state_mean = nc["xt_state_mean"]
        xt_state_gridpoint = nc["xt_state_gridpoint"]
        xt_norm_mean = nc["xt_norm_mean"]
        xt_norm_mean = nc["xt_norm_mean"]
        xt_norm_gridpoint = nc["xt_norm_gridpoint"]
        # Safe loading for existing files that might lack new vars
        if "xt_force_prior_gridpoint" in nc.variables:
            xt_force_prior_gridpoint = nc["xt_force_prior_gridpoint"]
            xt_force_like_gridpoint = nc["xt_force_like_gridpoint"]
        else:
            xt_force_prior_gridpoint = None
            xt_force_like_gridpoint = None

        
        OBS_INCLUDE = getattr(self, "obs_include_vars", None)
        if OBS_INCLUDE is not None and not isinstance(OBS_INCLUDE, set):
            OBS_INCLUDE = set(OBS_INCLUDE)

        # Ensure device / RNG
        if not hasattr(self, "device"):
            self.device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        _torch.manual_seed(getattr(self, "rng_seed", 42))
        device = self.device
        
        # Print GPU status
        if device.type == "cuda":
            print(f"[ReverseSDE] ✓ GPU ENABLED: Using {_torch.cuda.get_device_name(0)} (CUDA device {_torch.cuda.current_device()})")
        else:
            print(f"[ReverseSDE] ⚠ GPU NOT AVAILABLE: Using CPU only")

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
                # >>> CHANGE B: helper to map each local index in a block to its var name
                # >>> CHANGE B (REPLACE): build var-name array by offsets, not indices
        def _block_varnames_array(block_idx, n_block):
            """
            Return an array of length n_block mapping each local state row j in this block
            to its variable name (e.g., 'TG1', 'UG0', ...). The block is a concatenation
            of slices listed in self.nm.mask_cor[block_idx], where each item is:
                it[0] -> (v_idx, lev)
                it[1] -> (lat, lon)
            and each slice length is lat*lon.
            """
            import numpy as _np
            vnames = _np.empty(n_block, dtype=object)
            off = 0
            try:
                items = self.nm.mask_cor[block_idx]
                for it in items:
                    (v_idx, _lev) = it[0]
                    lat, lon = it[1]  # (lat, lon), not an index array
                    n = int(lat) * int(lon)
                    vname = self.nm.var_names[v_idx]
                    vnames[off:off+n] = vname
                    off += n
                if off != n_block:
                    print(f"[ReverseSDE][block={block_idx}] WARNING: offset {off} != n_block {n_block}")
            except Exception as e:
                print(f"[ReverseSDE][block={block_idx}] WARNING: could not build var mapping: {e}")
                return None
            return vnames



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
                self._write_unified_nc_block(block_idx, XB_block=XB_block, XA_block=XA_block, obs_info=None)

            # Publish and return
            self.map_vector_states()
            return
        # DEBUG
        print("ReverseSDE cycle index (k):", cycle_k)
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
                self._write_unified_nc_block(block_idx, XB_block=XB_block, XA_block=XA_block, obs_info=None)
                continue

            XB = XB_info["XB_b"]            # (n_block, Nens)
            init_std = XB_info["std_init"]  # (n_block,)
            n_block, Nens = XB.shape

            # Get observation map (may be None)
                        # Get observation map (may be None)
            OM = self.obs_map[block_idx] if block_idx < om_len else None

            # Default fallback: inflated background
            XA_block_fallback = self.covariance_inflation(XB)

            # ============================================================
            # (Pre-A) WIND OBSERVATION SETUP (AUXILIARY)
            # ============================================================
            # Check if this block is UG1 or VG1 and if we have wind obs in self.obs.wind_obs
            is_wind_update = False
            wind_mode = None 
            
            # Identify Block Variable
            block_var_name = None
            block_level = None
            if len(self.nm.mask_cor[block_idx]) > 0:
                 v_inf, r_inf = self.nm.mask_cor[block_idx][0]
                 block_var_idx = v_inf[0]
                 block_level   = v_inf[1]
                 block_var_name = self.nm.var_names[block_var_idx]
            
            # Determine Mode
            if block_var_name == 'UG1':
                wind_mode = "updating_u"
            elif block_var_name == 'VG1':
                wind_mode = "updating_v"
            
            target_wdg_data = None
            target_wsg_data = None
            
            # Use self.current_cycle_k which is set in prepare_analysis
            k_step = getattr(self, 'current_cycle_k', 0)
            
            # Initialize for safety
            partner_block_idx = -1

            if wind_mode and hasattr(ob, 'y_wind_obs'):
                 # WDG
                 if 'WDG1' in ob.wind_obs and block_level in ob.wind_obs['WDG1']:
                      # Metadata
                      meta = ob.wind_obs['WDG1'][block_level]
                      # Data
                      if k_step < len(ob.y_wind_obs):
                          raw_w_step = ob.y_wind_obs[k_step]
                          data = raw_w_step.get('WDG1', {}).get(block_level)
                          
                          if data is None and block_idx == 4:
                              print(f"[DEBUG] WDG1 MISSING k={k_step} lev={block_level}. Keys: {list(raw_w_step.keys())}")
                              
                          if data is not None:
                              target_wdg_data = {
                                  'idx_observed': np.array(meta['stations']),
                                  'y': data.flatten(), # Mean obs (1D)
                                   'sigma': np.full(len(data), self.wind_err.get('WDG1', 1.0)) # Error std (1D)
                              }

                 # WSG
                 if 'WSG1' in ob.wind_obs and block_level in ob.wind_obs['WSG1']:
                      meta = ob.wind_obs['WSG1'][block_level]
                      if k_step < len(ob.y_wind_obs):
                          data = ob.y_wind_obs[k_step].get('WSG1', {}).get(block_level)
                          if data is not None:
                              target_wsg_data = {
                                  'idx_observed': np.array(meta['stations']),
                                  'y': data.flatten(), # 1D
                                   'sigma': np.full(len(data), self.wind_err.get('WSG1', 1.0)) # 1D
                              }

            # If we don't have explicit OM, but we have wind data, synthesize!
            synth_idx = None
            is_synthesized_wind = False # Flag to suppress linear score

            if ((OM is None) or ("idx_observed" not in OM) or (OM["idx_observed"] is None) or (OM["idx_observed"].size == 0)):
                 if wind_mode and (target_wdg_data or target_wsg_data):
                     # Union of indices
                     idxs = []
                     if target_wdg_data: idxs.extend(target_wdg_data['idx_observed'])
                     if target_wsg_data: idxs.extend(target_wsg_data['idx_observed'])
                     
                     if idxs:
                         synth_idx = np.unique(idxs)
                         # Create Dummy OM
                         m_synth = synth_idx.size
                         y_dummy = _np.zeros(m_synth, dtype=_np.float32)
                         sigma_dummy = _np.full(m_synth, 1.0e9, dtype=_np.float32)
                         
                         OM = {
                             "idx_observed": synth_idx,
                             "y": y_dummy,
                             "sigma": sigma_dummy
                         }
                         # Flag as wind update
                         is_wind_update = True
                         is_synthesized_wind = True
            
            # If we DO have OM, we still set is_wind_update if we have wind data
            if wind_mode and (target_wdg_data or target_wsg_data):
                is_wind_update = True

            # ... [Rest of code handles SDE] ...
            # I need to ensure target_wdg_data / target_wsg_data are passed to _perform_assimilation_block
            # But here we are IN _perform_assimilation_block (inline).
            # So I can just use them.
            
            # --- base "no obs" gate ---
            if (OM is None) or ("idx_observed" not in OM) or (OM["idx_observed"] is None):
                self.XA_map.append(XA_block_fallback)
                self._write_unified_nc_block(block_idx, XB_block=XB, XA_block=XA_block_fallback, obs_info=OM)
                continue

            idx_ob = OM["idx_observed"]
            if idx_ob.size == 0:
                self.XA_map.append(XA_block_fallback)
                self._write_unified_nc_block(block_idx, XB_block=XB, XA_block=XA_block_fallback, obs_info=OM)
                continue
            
            # ============================================================
            # Build obs-space index sets
            # ============================================================
            vnames_block = _np.empty(n_block, dtype=object)
            levs_block   = _np.empty(n_block, dtype=int)
            off = 0
            for (v_info, res) in self.nm.mask_cor[block_idx]:
                (v_idx, lev) = v_info
                lat_n, lon_n = res
                N = int(lat_n) * int(lon_n)
                # handle if v_idx is out of bound? (Should not happen for valid mask_cor)
                if v_idx < len(self.nm.var_names):
                    vname = self.nm.var_names[v_idx]
                else:
                    vname = "UNKNOWN"
                
                vnames_block[off:off+N] = vname
                levs_block[off:off+N]   = int(lev)
                off += N
            
            if off != n_block:
                print(f"[ReverseSDE][{label}] WARNING: offset {off} != n_block {n_block}")

            # Restrict to observation locations
            vnames_obs = vnames_block[idx_ob]
            levs_obs   = levs_block[idx_ob]

            # Tracking logic...
            tracking_obs_indices = []
            specs = [
                ("UG1",  7),
                ("VG1",  7),
                ("TG1",  7),
                ("TRG1", 7),
                ("PSG1", None),
            ]
            for base_name, lev_target in specs:
                if lev_target is None:
                    mask = (vnames_obs == base_name)
                else:
                    mask = (vnames_obs == base_name) & (levs_obs == lev_target)

                idxs_j = _np.where(mask)[0].astype(_np.int64)  # obs indices j in 0..m-1
                tracking_obs_indices.append(idxs_j)
            # Do we actually have any obs for the tracked var/lev combos in this block?
            has_any_tracked = any(arr.size > 0 for arr in tracking_obs_indices)
            if not has_any_tracked:
                # No UG7/VG7/TG7/TRG7@lev7 or PSG1 in this block -> skip tracking for this block
                # Still do the SDE and analysis normally, just don't write to xt_state.
                do_track = False
            else:
                do_track = True

            # >>> CHANGE C: if user requested a subset of variables/time-levels, filter idx_observed down
            if OBS_INCLUDE is not None and len(OBS_INCLUDE) > 0:
                varnames_in_block = _block_varnames_array(block_idx, n_block)
                if varnames_in_block is not None:
                    import numpy as _np
                    keep_mask = _np.array([varnames_in_block[j] in OBS_INCLUDE for j in idx_ob], dtype=bool)
                    if keep_mask.size != idx_ob.size:
                        print(f"[ReverseSDE][{label}] WARNING: keep_mask size mismatch; skipping filter")
                    else:
                        if not keep_mask.any():
                            # Entire block has no kept obs after filtering
                            self.XA_map.append(XA_block_fallback)
                            self._write_unified_nc_block(block_idx, XB_block=XB, XA_block=XA_block_fallback, obs_info=OM)
                            continue
                        # filter all obs-space vectors consistently
                        idx_ob = idx_ob[keep_mask]
                        if "y" in OM and OM["y"] is not None:
                            OM["y"] = OM["y"].reshape(-1)[keep_mask]
                        if "sigma" in OM and OM["sigma"] is not None:
                            OM["sigma"] = OM["sigma"].reshape(-1)[keep_mask]

            # Re-check after filtering
            if idx_ob.size == 0:
                self.XA_map.append(XA_block_fallback)
                self._write_unified_nc_block(block_idx, XB_block=XB, XA_block=XA_block_fallback, obs_info=OM)
                continue

            # ============================================================
            # (A) PRE-NORMALIZATION (shared for both linear & nonlinear)
            # ============================================================
            idx_ob = OM["idx_observed"]
            y_np = OM["y"].reshape(-1)
            sigma_np = OM["sigma"].reshape(-1)
            m = idx_ob.size

            # Convert to Torch
            prior_ens_np = XB.T.copy() # (Nens, n_block)
            X0_obs_np = prior_ens_np[:, idx_ob] # (Nens, m)
            
            # Move to device
            X0_obs = _torch.from_numpy(X0_obs_np.astype(_np.float32)).to(device)
            y = _torch.from_numpy(y_np.astype(_np.float32)).to(device)
            sigma = _torch.from_numpy(sigma_np.astype(_np.float32)).to(device)
            
            mu_vec_np = OM.get("mu_vec")
            std_vec_np = OM.get("std_vec")
            if mu_vec_np is not None:
                mu_vec_t = _torch.from_numpy(mu_vec_np.astype(_np.float32)).to(device)
                std_vec_t = _torch.from_numpy(std_vec_np.astype(_np.float32)).to(device)
            else:
                mu_vec_t = _torch.tensor(0.0, device=device)
                std_vec_t = _torch.tensor(1.0, device=device)

            # --- NORMALIZE FLAG LOGIC ---
            if self.normalize:
                # Always compute true ensemble stats on device to preserve flow-dependent covariance scaling
                mean_X0 = _torch.mean(X0_obs, dim=0) # (m,)
                std_X0 = _torch.std(X0_obs, dim=0) # (m,)
                # Prevent div by zero
                std_X0 = _torch.clamp(std_X0, min=1e-5)
                
                X0_obs_n = (X0_obs - mean_X0) / std_X0
            else:
                # No normalization: mean=0, std=1
                mean_X0 = _torch.zeros(m, device=device, dtype=_torch.float32)
                std_X0 = _torch.ones(m, device=device, dtype=_torch.float32)
            X0_obs_n = (X0_obs - mean_X0) / std_X0 # (Nens, m)

            # ============================================================
            # (B) TOGGLE: LINEAR vs NONLINEAR OBSERVATION HANDLING
            # ============================================================
            if self.nonlinear_obs:
                # --- Nonlinear case: Climatological Z-Score ---
                sf = float(getattr(self, "scalefact", 1.0))

                # Prior conditioning: physical-normalized (same as linear)
                X0_obs_n_t = X0_obs_n

                # y remains in arctan space. No unstable tan(y) required!
                y_n_t = y
                sigma_n_t = sigma
                
                # sigma_n for diagnostics
                sigma_n = sigma / std_X0


            else:
                # --- Linear case (Vanilla logic) ---
                y_n = (y - mean_X0) / std_X0
                sigma_n = sigma / std_X0

                X0_obs_n_t = X0_obs_n
                y_n_t = y_n
                sigma_n_t = sigma_n

            # ============================================================
            # (C) INITIALIZE REVERSE SDE STATE (common)
            # ============================================================
            xt = _torch.randn((Nens, m), device=device, dtype=_torch.float32)
            # Normalize initial noise (matching vanilla implementation)
            xt = (xt - xt.mean(dim=0)) / (_torch.std(xt, dim=0) + 1e-5)
            xt_means_hist = _torch.zeros((hist_len, m), device=device, dtype=_torch.float32)
            t = 1.0

            print(f"[ReverseSDE][{label}] starting reverse-SDE (m={m}, Nens={Nens})")
            # Tiny summary of which vars survived in this block
            try:
                vnames = _block_varnames_array(block_idx, n_block)
                if vnames is not None:
                    import numpy as _np
                    kept = _np.unique(vnames[idx_ob])
                    print(f"[ReverseSDE][{label}] kept vars: {list(kept)}  (m={m})")
            except Exception:
                pass

            block_numeric_ok = True

            # [USER-REQUEST] Setup for tracking pressure (PSG1, index 4) at (27,32) for cycles 5-8
            track_pressure_data = {
                "xt": [],
                "prior": [],
                "like": [],
                "steps": []
            }
            do_plot_pressure = (cycle_k in [5, 6, 7, 8])

            obs_wdg_vals = None
            obs_wdg_sigma = None
            obs_wdg_mask = None # Boolean mask over m (the current local indices)

            if wind_mode and target_wdg_data is not None:
                try:
                    # DEBUG
                    if block_idx == 4:
                        print(f"[DEBUG] WDG1 block entered: k={k_step}, lev={block_level}")
                    
                    # Get WDG obs data from our lookup
                    idx_wdg = target_wdg_data["idx_observed"]
                    y_wdg   = target_wdg_data["y"]
                    sig_wdg = target_wdg_data["sigma"]
                    
                    # We need to find overlap: which of our 'idx_ob' (U indices) are also in 'idx_wdg'?
                    # idx_ob are the global indices of the points we are tracking
                    common_mask = _np.isin(idx_ob, idx_wdg)

                    if block_idx == 4:
                        print(f"[DEBUG] WDG1 common_mask.any() = {common_mask.any()}, sum = {common_mask.sum()}")

                    if common_mask.any():
                        is_wind_update = True
                        
                        # Create aligned arrays (size m)
                        y_wdg_aligned = _np.full(m, _np.nan)
                        sig_wdg_aligned = _np.full(m, _np.nan)
                        
                        # Flatten y and sigma for lookup
                        y_wdg_flat = y_wdg.flatten()
                        sig_wdg_flat = sig_wdg.flatten()
                        
                        # Create lookup dicts
                        wdg_lookup = dict(zip(idx_wdg, y_wdg_flat))
                        sig_lookup = dict(zip(idx_wdg, sig_wdg_flat))
                        
                        # Fill aligned
                        for k in _np.where(common_mask)[0]:
                            g_idx = idx_ob[k]
                            y_wdg_aligned[k] = wdg_lookup[g_idx]
                            sig_wdg_aligned[k] = sig_lookup[g_idx]
                            
                        obs_wdg_vals  = _torch.from_numpy(y_wdg_aligned.astype(_np.float32)).to(device)
                        obs_wdg_sigma = _torch.from_numpy(sig_wdg_aligned.astype(_np.float32)).to(device)
                        obs_wdg_mask  = _torch.from_numpy(common_mask).to(device)
                        
                        print(f"[ReverseSDE][{label}] Linked WDG1 obs: {common_mask.sum()} points")

                except Exception as e:
                    print(f"[ReverseSDE][{label}] Failed to link WDG obs: {e}")
                    is_wind_update = False

            # Load WSG Obs if available
            obs_wsg_vals = None
            obs_wsg_sigma = None
            obs_wsg_mask = None

            if wind_mode and target_wsg_data is not None:
                try:
                    idx_wsg = target_wsg_data["idx_observed"]
                    y_wsg   = target_wsg_data["y"]
                    sig_wsg = target_wsg_data["sigma"]
                    
                    common_mask_wsg = _np.isin(idx_ob, idx_wsg)

                    if common_mask_wsg.any():
                            # We treat WSG as another potential wind update source
                            # If we have either WDG or WSG, we need partner state
                            # is_wind_update might already be True from WDG
                            is_wind_update = True
                            
                            y_wsg_aligned = _np.full(m, _np.nan)
                            sig_wsg_aligned = _np.full(m, _np.nan)
                            
                            wsg_lookup = dict(zip(idx_wsg, y_wsg))
                            sig_lookup = dict(zip(idx_wsg, sig_wsg))
                            
                            for k in _np.where(common_mask_wsg)[0]:
                                g_idx = idx_ob[k]
                                y_wsg_aligned[k] = wsg_lookup[g_idx]
                                sig_wsg_aligned[k] = sig_lookup[g_idx]
                                
                            obs_wsg_vals  = _torch.from_numpy(y_wsg_aligned.astype(_np.float32)).to(device)
                            obs_wsg_sigma = _torch.from_numpy(sig_wsg_aligned.astype(_np.float32)).to(device)
                            obs_wsg_mask  = _torch.from_numpy(common_mask_wsg).to(device)
                            
                            print(f"[ReverseSDE][{label}] Linked WSG1 obs: {common_mask_wsg.sum()} points")

                except Exception as e:
                    print(f"[ReverseSDE][{label}] Failed to link WSG obs: {e}")
                    # Don't set is_wind_update = False here, might still have WDG
                    pass
            
            # Fetch Partner State (Constant during this update)
            partner_state = None
            
            # Re-identify partner block index if we need wind update
            if is_wind_update and partner_block_idx == -1:
                 # Standard offset assumption: UG1 is block I, VG1 is block I+1 (or vice versa)
                 if wind_mode == "updating_u":
                      # Partner is VG1 which is likely block_idx + 1
                      candidate = block_idx + 1
                 else:
                      # Partner is UG1 which is likely block_idx - 1
                      candidate = block_idx - 1
                 
                 if 0 <= candidate < len(self.XB_map):
                      partner_block_idx = candidate

            if is_wind_update and partner_block_idx != -1:
                 try:
                     # We need the Partner Ensembles at the locations of 'idx_ob'
                     # To do this correctly, we access the global XB_map
                     XB_partner_info = self.XB_map[partner_block_idx]
                     XB_partner_full = XB_partner_info["XB_b"] # (n_block, Nens)
                     
                     # Extract at our indices (idx_ob)
                     # Assuming blocks are aligned 1-to-1 (Indices 0..N match)
                     XB_partner_sub = XB_partner_full[idx_ob, :] # (m, Nens)
                     
                     # Convert to Torch
                     partner_state = _torch.from_numpy(XB_partner_sub.T.astype(_np.float32)).to(device) # (Nens, m)
                     
                 except Exception as e:
                     print(f"[ReverseSDE][{label}] Failed to fetch partner state: {e}")
                     is_wind_update = False

            for i in range(1, psteps+1):
                if i == 1:  # only print once per block
                    try:
                        std_val   = std_X0.mean().item()
                        sigma_val = sigma.mean().item()
                        sigma_n_val = sigma_n.mean().item()
                        sigma_n_min = sigma_n.min().item()
                        sigma_n_max = sigma_n.max().item()
                        std_min = std_X0.min().item()
                        print(f"[ReverseSDE][{label}] init stats: "
                            f"ens_std_mean={std_val:.3e}, min_ens_std={std_min:.3e}, obs_err={sigma_val:.3e}")
                        print(f"[ReverseSDE][{label}] sigma_n stats: mean={sigma_n_val:.3e}, min={sigma_n_min:.3e}, max={sigma_n_max:.3e}")
                    except Exception as e:
                        print(f"[ReverseSDE][{label}] could not compute init stats: {e}")

                alpha_t = self._cond_alpha(t)
                sigma2_t  = self._cond_sigma_sq(t)
                f       = self._f(t)
                g       = self._g(t)
                tau     = self._g_tau(t)
                
                # =========================================================
                # Record SDE state (if this block has any tracked variables)
                # =========================================================
                if do_track:
                    # Convert to CPU NumPy for tracking/writing
                    xt_np = xt.detach().cpu().numpy()                  # (Nens, m)
                    mean_X0_np = mean_X0.detach().cpu().numpy()
                    std_X0_np = std_X0.detach().cpu().numpy()
                    
                    x_obs_step = mean_X0_np[None, :] + std_X0_np[None, :] * xt_np   # (Nens, m)
                    x_norm_step = xt_np # (Nens, m) - Already normalized

                    values_mean = _np.full((5, Nens), _np.nan, dtype=_np.float32)
                    values_gridpoint = _np.full((5, len(self.track_gridpoint_locs), Nens), _np.nan, dtype=_np.float32)
                    
                    values_norm_mean = _np.full((5, Nens), _np.nan, dtype=_np.float32)
                    values_norm_gridpoint = _np.full((5, len(self.track_gridpoint_locs), Nens), _np.nan, dtype=_np.float32)
                    
                    values_force_prior_gridpoint = _np.full((5, len(self.track_gridpoint_locs), Nens), _np.nan, dtype=_np.float32)
                    values_force_like_gridpoint = _np.full((5, len(self.track_gridpoint_locs), Nens), _np.nan, dtype=_np.float32)

                    for vidx, obs_idx in enumerate(tracking_obs_indices):
                        if obs_idx.size == 0:
                            continue  # leave row as NaN
                        
                        # 1. Always track spatial mean
                        sub = x_obs_step[:, obs_idx]     # (Nens, n_pts)
                        values_mean[vidx, :] = sub.mean(axis=1).astype(_np.float32)
                        
                        sub_norm = x_norm_step[:, obs_idx]
                        values_norm_mean[vidx, :] = sub_norm.mean(axis=1).astype(_np.float32)

                        # 2. Track gridpoint if requested
                        if self.track_gridpoint_locs:
                            for pt_idx, loc in enumerate(self.track_gridpoint_locs):
                                lat_target, lon_target = loc
                                
                                base_name, lev_target = specs[vidx]
                                found_target = False
                                current_offset = 0
                                
                                for (v_info, res) in self.nm.mask_cor[block_idx]:
                                    (v_idx_loop, lev_loop) = v_info
                                    lat_n, lon_n = res
                                    N = int(lat_n) * int(lon_n)
                                    vname_loop = self.nm.var_names[v_idx_loop]
                                    
                                    is_target_var = (vname_loop == base_name)
                                    if lev_target is not None:
                                        is_target_var = is_target_var and (int(lev_loop) == lev_target)
                                    
                                    if is_target_var:
                                        target_flat_idx = lat_target * int(lon_n) + lon_target
                                        if target_flat_idx < N:
                                            block_target_idx = current_offset + target_flat_idx
                                            matches = _np.where(idx_ob == block_target_idx)[0]
                                            if matches.size > 0:
                                                k_idx = matches[0]
                                                val = x_obs_step[:, k_idx]
                                                values_gridpoint[vidx, pt_idx, :] = val.astype(_np.float32)
                                                
                                                val_norm = x_norm_step[:, k_idx]
                                                values_norm_gridpoint[vidx, pt_idx, :] = val_norm.astype(_np.float32)
                                                found_target = True
                                        
                                    current_offset += N
                                    if found_target:
                                        break
                            

                            
                        if (i % 20) == 0 or i == 1:
                            var_list = ["UG1","VG1","TG1","TRG1","PSG1"]
                            # Only print if we actually have values (not all NaN)
                            # Check first element to see if it's NaN
                            if not _np.isnan(values_mean[vidx, 0]):
                                row = values_mean[vidx, :]
                                print(f"[{label}] pstep={i:03d} {var_list[vidx]} mean={row.mean():.4e}")

                    # Write into NetCDF only if this block had tracked obs
                    xt_state_mean[cycle_k, block_idx, i-1, :, :] = values_mean
                    xt_state_gridpoint[cycle_k, block_idx, i-1, :, :] = values_gridpoint
                    xt_norm_mean[cycle_k, block_idx, i-1, :, :] = values_norm_mean
                    xt_norm_gridpoint[cycle_k, block_idx, i-1, :, :] = values_norm_gridpoint

                # ====== SDE UPDATE ======
                # Log-probability gradient of the Gaussian prior: \nabla \log N(x; \mu, \beta) = - (x - \mu) / \beta
                prior_term = -(xt - alpha_t * X0_obs_n_t) / sigma2_t
                
                # Likelihood score
                if self.nonlinear_obs:
                    sf = float(self.scalefact)
                    # 1. Expand `xt` to physical space using ensemble statistics
                    X_phys = xt * std_X0 + mean_X0
                    
                    # 2. Compress to the observation's localized-climatology Z-space
                    X_c_norm = (X_phys - mu_vec_t) / std_vec_t
                    
                    # 3 & 4. Evaluate Observation Forward Operator & Exact Chain Rule Jacobian
                    if self.nonlinear_operator_type == 'arctan_sq':
                        u = sf * X_c_norm
                        h_xt = _torch.atan(u ** 2)
                        jacobian = (2.0 * u / (1.0 + u ** 4)) * (sf * std_X0 / std_vec_t)
                    else:
                        h_xt = _torch.atan(sf * X_c_norm)
                        # Exact Chain Rule Jacobian: d(h_xt)/d(xt)
                        # Note: std_X0 and std_vec_t are (m,) vectors.
                        jacobian = (sf / (1.0 + (sf * X_c_norm) ** 2)) * (std_X0 / std_vec_t)
                    
                    # 5. Likelihood Score
                    like_score = -(h_xt - y) / (sigma ** 2) * jacobian
                else:
                    like_score = -(xt - y_n_t) / (sigma_n_t ** 2)

                # FOR "WIND-ONLY" SYNTHESIZED BLOCKS:
                # The linear score is based on dummy observations (y=0, sigma=huge).
                # To be absolutely safe, we force it to zero here.
                if is_synthesized_wind:
                     like_score = _torch.zeros_like(like_score)

                # --- (E) ADD NONLINEAR WIND SCORES ---
                if is_wind_update and partner_state is not None:
                     # 1. De-normalize current state to physical space
                     x_phy = mean_X0 + xt * std_X0
                     dx_phys_dxt = std_X0
                     
                     # 2. Identify U and V components
                     if wind_mode == "updating_u":
                         u_vec = x_phy
                         v_vec = partner_state # (Nens, m) already physical
                     else:
                         u_vec = partner_state
                         v_vec = x_phy
                     
                     # Pre-compute magnitude squared for gradients
                     res_sq = u_vec**2 + v_vec**2
                     res_sq = _torch.clamp(res_sq, min=1e-6) # avoid div0

                     # --- WDG Contribution ---
                     if obs_wdg_vals is not None:
                         # WDG = atan2(u, v)
                         pred_wdg = _torch.atan2(u_vec, v_vec) # (Nens, m)
                         
                         # Residual (Innovation)
                         diff = pred_wdg - obs_wdg_vals 
                         # Wrap to [-pi, pi]
                         diff = (diff + _np.pi) % (2 * _np.pi) - _np.pi
                         
                         # Gradient of h w.r.t current state
                         if wind_mode == "updating_u":
                             grad_h_xphy = v_vec / res_sq
                         else:
                             grad_h_xphy = -u_vec / res_sq
                             
                         # Chain rule: dh_wind/dxt = dh_wind/dx_phys * dx_phys/dxt
                         grad_h_xt = grad_h_xphy * dx_phys_dxt
                         
                         # Score
                         wind_score = -(diff / (obs_wdg_sigma**2)) * grad_h_xt
                         wind_score = _torch.where(obs_wdg_mask, wind_score, _torch.zeros_like(wind_score))
                         like_score = like_score + wind_score

                     # --- WSG Contribution ---
                     if obs_wsg_vals is not None:
                         # WSG = sqrt(u^2 + v^2)
                         pred_wsg = _torch.sqrt(res_sq)
                         
                         diff_wsg = pred_wsg - obs_wsg_vals
                         
                         # Gradients
                         # dh/du = u / speed, dh/dv = v / speed
                         speed_safe = _torch.clamp(pred_wsg, min=1e-6)
                         
                         if wind_mode == "updating_u":
                             grad_wsg_xphy = u_vec / speed_safe
                         else:
                             grad_wsg_xphy = v_vec / speed_safe
                             
                         # Chain rule: dh_wsg/dxt = dh_wsg/dx_phys * dx_phys/dxt
                         grad_wsg_xt = grad_wsg_xphy * dx_phys_dxt
                         
                         wsg_score = -(diff_wsg / (obs_wsg_sigma**2)) * grad_wsg_xt
                         wsg_score = _torch.where(obs_wsg_mask, wsg_score, _torch.zeros_like(wsg_score))
                         like_score = like_score + wsg_score
                
                like_tau = tau * like_score
                pull = (g ** 2) * like_tau

                # --- LATE TRACKING (Gridpoint Forces) ---
                if do_track and self.track_gridpoint_locs:
                     prior_np = prior_term.detach().cpu().numpy()
                     like_np  = like_tau.detach().cpu().numpy()
                     
                     for vidx, obs_idx in enumerate(tracking_obs_indices):
                         for pt_idx, loc in enumerate(self.track_gridpoint_locs):
                             lat_target, lon_target = loc
                             base_name, lev_target = specs[vidx]
                             
                             # Find offset
                             found_target = False
                             current_offset = 0
                             for (v_info, res) in self.nm.mask_cor[block_idx]:
                                    (v_idx_loop, lev_loop) = v_info
                                    lat_n, lon_n = res
                                    N = int(lat_n) * int(lon_n)
                                    vname_loop = self.nm.var_names[v_idx_loop]
                                    is_target_var = (vname_loop == base_name)
                                    if lev_target is not None:
                                        is_target_var = is_target_var and (int(lev_loop) == lev_target)
                                    if is_target_var:
                                        target_flat_idx = lat_target * int(lon_n) + lon_target
                                        if target_flat_idx < N:
                                            block_target_idx = current_offset + target_flat_idx
                                            # Map to obs space
                                            matches = _np.where(idx_ob == block_target_idx)[0]
                                            if matches.size > 0:
                                                k_idx = matches[0]
                                                values_force_prior_gridpoint[vidx, pt_idx, :] = prior_np[:, k_idx]
                                                values_force_like_gridpoint[vidx, pt_idx, :]  = like_np[:, k_idx]
                                                found_target = True
                                        
                                    current_offset += N
                                    if found_target:
                                        break
                                        
                # Update NetCDF with force components
                if do_track and xt_force_prior_gridpoint is not None:
                     xt_force_prior_gridpoint[cycle_k, block_idx, i-1, :, :, :] = values_force_prior_gridpoint
                     xt_force_like_gridpoint[cycle_k, block_idx, i-1, :, :, :] = values_force_like_gridpoint
                
                # [USER-REQUEST] Collect pressure data for plotting
                if do_plot_pressure and do_track:
                    # Dynamically find PSG1 index in the local specs list
                    # (User noted index 9 globally, but locally it depends on specs definition)
                    try:
                        psg_idx = [s[0] for s in specs].index("PSG1")
                    except ValueError:
                        psg_idx = -1

                    if psg_idx >= 0:
                        v_psg = values_gridpoint[psg_idx] 
                        if not _np.isnan(v_psg).all():
                             prior_psg = values_force_prior_gridpoint[psg_idx]
                             like_psg = values_force_like_gridpoint[psg_idx]
                             
                             track_pressure_data["xt"].append(v_psg.copy())
                             track_pressure_data["prior"].append(prior_psg.copy())
                             track_pressure_data["like"].append(like_psg.copy())
                             track_pressure_data["steps"].append(i)

                # Diagnostics
                if  DEBUG_EVERY > 0 and (i % DEBUG_EVERY == 0 or i == 1):
                    with _torch.no_grad():
                        abs_base = _torch.abs(like_score)
                        abs_tau  = _torch.abs(like_tau)
                        abs_pull = _torch.abs(pull)
                        m_fin = _torch.isfinite(abs_pull) & _torch.isfinite(abs_tau) & _torch.isfinite(abs_base)
                        if m_fin.any():
                            print(
                                f"[ReverseSDE][{label}] step={i:03d} "
                                f"|like_score| mean={abs_base[m_fin].mean().item():.4e} max={abs_base[m_fin].max().item():.4e}  "
                                f"|like_tau| mean={abs_tau[m_fin].mean().item():.4e} max={abs_tau[m_fin].max().item():.4e}  "
                            )
                        else:
                            print(f"[ReverseSDE][{label}] step={i:03d} diagnostics non-finite")

                # --- Drift Calculation (with optional clipping) ---
                # The total analytical score matches the log-probability gradient
                score = prior_term + like_tau
                
                if self.score_clip is not None:
                     # dynamically clamp the raw score to avoid Forward Euler integration jumps
                     max_score = 100.0 
                     score = _torch.clamp(score, -max_score, max_score)
                     
                # Mathematically Exact Continuous-Time Reverse SDE: dx = [-f*x + g^2 * score] dt + g dW
                drift = -f * xt + (g ** 2) * score

                noise = _torch.sqrt(_torch.tensor(dt, device=device, dtype=xt.dtype)) * g * _torch.randn_like(xt)
                xt_next = xt + dt * drift + noise

                # --- 4. SAFETY CLAMP ---
                if self.state_clip is not None:
                    # Check if anything is out of bounds before clamping (for warning)
                    with _torch.no_grad():
                        outliers = _torch.abs(xt_next) > self.state_clip
                        if outliers.any():
                             # Rate limit warnings: only on step 50, 100, etc? Or just once per block?
                             # Let's just print max value if it's huge
                             max_val = xt_next.abs().max().item()
                             if max_val > self.state_clip * 1.5: # massive spike
                                 print(f"[ReverseSDE][{label}] step={i} CLAMPING: state hit {max_val:.1f} > {self.state_clip}")
                    xt_next = _torch.clamp(xt_next, -self.state_clip, self.state_clip)

                if not _torch.isfinite(xt_next).all():
                    print(f"[ReverseSDE][{label}] State became non-finite at step {i}. Using fallback for this block.")
                    block_numeric_ok = False
                    break

                # --- EARLY STOPPING (Prototype feature) ---
                if self.enable_early_stopping:
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
            
                xt = xt_next
                t = max(0.0, t - dt)

            
            # Produce XA_block (either fallback or from xt)
            if not block_numeric_ok:
                self.XA_map.append(XA_block_fallback)
                self._write_unified_nc_block(block_idx, XB_block=XB, XA_block=XA_block_fallback, obs_info=OM)
                continue

            # Convert back to NumPy for final output
            xt_np = xt.detach().cpu().numpy()                # (Nens, m)
            mean_X0_np = mean_X0.detach().cpu().numpy()
            std_X0_np = std_X0.detach().cpu().numpy()
            
            x_obs_ana = mean_X0_np + xt_np * std_X0_np       # (Nens, m)
            x_full = prior_ens_np.copy()                     # (Nens, n_block)
            x_full[:, idx_ob] = x_obs_ana


            mu = x_full.mean(axis=0)                         # (n_block,)
            sd = x_full.std(axis=0) + 1e-12
            Xstd = (x_full - mu) / sd
            x_full = Xstd * init_std + mu

            XA_block = self.covariance_inflation(x_full.T)   # (n_block, Nens)
            self.XA_map.append(XA_block)
            self._write_unified_nc_block(block_idx, XB_block=XB, XA_block=XA_block, obs_info=OM)

            # (optional) save clean XB for gaussianity if enabled
            # (Removed SAVE_GAUSS_BLOCKS)

        nc.close()
        # Final sanity: if XA_map is short, pad with inflated background per block
        if len(self.XA_map) < mc_len:
            print(f"[ReverseSDE] WARNING: XA_map has {len(self.XA_map)} of {mc_len} blocks. Padding with fallbacks.")
            for block_idx in range(len(self.XA_map), mc_len):
                XB_block = self.get_ensemble_block(self.nm.mask_cor[block_idx])
                XA_block = self.covariance_inflation(XB_block)
                self.XA_map.append(XA_block)
                # Unified write for padding blocks
                self._write_unified_nc_block(block_idx, XB_block=XB_block, XA_block=XA_block, obs_info=None)

        self.map_vector_states()
        # >>> write one unified netcdf per requested field, now that analysis exists
        # self._write_unified_nc_reverseSDE(cycle_k=self.current_cycle_k)

class sequential_method:
      
      method_name = None;
      
      def __init__(self, method_name):
          self.method_name = method_name;
      
      def get_instance(self, nm, infla, Nens, nonlinear_obs=False, scalefact=1.0, wind_nonlinear_operator=False, wind_err=None, normalize_nonlinear=True, nonlinear_operator_type='arctan'):
        if self.method_name == "EnKF_MC":
            return EnKF_MC(nm, infla, Nens);
        elif self.method_name == "EnKF_MC_obs":
            # Pass wind_err if provided, else default to None/empty
            return EnKF_MC_obs(nm, infla, Nens, nonlinear_obs=nonlinear_obs, scalefact=scalefact, wind_err=wind_err, nonlinear_operator_type=nonlinear_operator_type);
        elif self.method_name == "ReverseSDE":
             return ReverseSDE(nm, infla, Nens, nonlinear_obs=nonlinear_obs, scalefact=scalefact, wind_err=wind_err, nonlinear_operator_type=nonlinear_operator_type);
        elif self.method_name == "LETKF":
            return LETKF(nm, infla, Nens, nonlinear_obs=nonlinear_obs, scalefact=scalefact, wind_nonlinear_operator=wind_nonlinear_operator, wind_err=wind_err, normalize_nonlinear=normalize_nonlinear, nonlinear_operator_type=nonlinear_operator_type);
        else:
            print("Method not found : ", self.method_name);
            return None;