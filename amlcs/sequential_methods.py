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
        # store unperturbed obs for each block (same source ReverseSDE used)
        self.Y_unp = [ob.y_obs[k][block] for block in range(0, N_blocks)]
        self._cycle_k = k
    
    def perform_assimilation(self, ob):
        self.XA_map = [];
        for block, (XB_info, Ys_block, R_info, H_block) in enumerate(zip(self.XB_map, self.Ys, ob.obs_R_sparse, ob.obs_H_sparse)):
            XB_block = XB_info['XB_b'];
            R_block  = R_info['R'];
            Binv_sqrt_block = XB_info['Binv_s_b'];
            XA_block = self.perform_assimilation_block(XB_block, Binv_sqrt_block, H_block, R_block, Ys_block);
            XA_block = self.covariance_inflation(XA_block);
            self.XA_map.append(XA_block);
            # write unified CSVs for this block/cycle
            self._write_unified_csv_block(block, H_block, R_block, XB_block, XA_block)
        self.map_vector_states(); #Update ensemble folders
    
    def perform_assimilation_block(self, XB, Binv_sqrt, H, R, Ys):
      
          Ds = Ys - H @ XB;
           
          H_spar = H.toarray();
          
          P = spa.linalg.spsolve_triangular(Binv_sqrt, H_spar.T, lower=False);
          
          Inno = R + P.T @ P;

          Q_temp = P @ spa.linalg.spsolve(Inno, Ds);
          
          #Q_temp = spa.linalg.spsolve_triangular(Binv_sqrt,Q_temp,lower=False);
          
          DXa = spa.linalg.spsolve_triangular(Binv_sqrt.T, Q_temp, lower=True);

          XA = XB + DXa;  
          
          return XA;
    
    # ------------------ minimal CSV writer (NetCDF truth/NoDA) ------------------
    def _write_unified_csv_block(self, block, H_block, R_block, XB_block, XA_block):
        # ensemble means
        xb_mean_full = XB_block.mean(axis=1)        # (n_block,)
        xa_mean_full = XA_block.mean(axis=1)        # (n_block,)

        # observation indices for this block (sorted by row)
        Hc = H_block.tocoo()
        order = np.argsort(Hc.row)
        obs_idx_block = Hc.col[order].astype(int)

        # unperturbed obs and sigma
        y_unp = self.Y_unp[block].reshape(-1)
        obs_vals = y_unp[: obs_idx_block.size]
        try:
            R_diag = R_block.diagonal()
        except Exception:
            R_diag = np.array(R_block.todense()).diagonal()
        sigma_vec = np.sqrt(np.asarray(R_diag)).reshape(-1)[: obs_idx_block.size]

        # load truth / noDA snapshots for this cycle
        k = self._cycle_k
        ref_nc = os.path.join(self.nm.snapshots, f"reference_solution_{k}.nc")
        fru_nc = os.path.join(self.nm.free_run,    f"free_run_{k}.nc")
        X_ref = self.nm.load_netcdf_file(ref_nc)   # list per variable
        X_nod = self.nm.load_netcdf_file(fru_nc)

        # iterate var/level slices inside this block using mask_cor
        offset = 0
        for (v_info, res) in self.nm.mask_cor[block]:
            v_idx, lev = v_info
            lat, lon = res
            n = int(lat) * int(lon)
            start, end = offset, offset + n
            offset = end

            var_name = self.nm.var_names[v_idx]
            lev_tag  = f"lev{lev}"

            # segment xb/xa
            xb = xb_mean_full[start:end]
            xa = xa_mean_full[start:end]

            # segment truth/noDA directly from NetCDF arrays
            if 'PSG' in var_name:
                tr2d = X_ref[v_idx][:, :]
                nd2d = X_nod[v_idx][:, :]
            else:
                tr2d = X_ref[v_idx][lev, :, :]
                nd2d = X_nod[v_idx][lev, :, :]
            tr = tr2d.reshape(-1)
            nd = nd2d.reshape(-1)

            # obs fields projected into this segment
            obs = np.full(n, np.nan, float)
            sig = np.full(n, np.nan, float)
            iso = np.zeros(n, dtype=int)
            in_seg = (obs_idx_block >= start) & (obs_idx_block < end)
            if in_seg.any():
                local = obs_idx_block[in_seg] - start
                obs[local] = obs_vals[in_seg]
                sig[local] = sigma_vec[in_seg]
                iso[local] = 1

            # write CSV
            df = pd.DataFrame({
                "idx":     np.arange(n, dtype=int),
                "xb_mean": xb.astype(float),
                "xa_mean": xa.astype(float),
                "truth":   tr.astype(float),
                "noda":    nd.astype(float),
                "obs":     obs,
                "sigma":   sig,
                "is_obs":  iso
            })
            out_dir = self.nm.path
            os.makedirs(out_dir, exist_ok=True)
            fn = os.path.join(out_dir, f"{var_name}_{lev_tag}_cycle{k}.csv")
            df.to_csv(fn, index=False)


      
         
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
                 eps_beta: float = 0.025,
                 nonlinear_obs: bool = False,
                 rng_seed: int = 42):
        super().__init__(nm, infla, Nens)
        self.p_time_step = int(pseudo_time_steps)
        self.eps_alpha = float(eps_alpha)
        self.scalefact = float(scalefact)
        self.eps_beta = float(eps_beta)
        self.rng = np.random.RandomState(int(rng_seed))
        self._cycle_idx = 0  # incremented once per perform_assimilation() call
        self.nonlinear_obs = bool(nonlinear_obs)  # ← add this line
        # Per-block working storage
        self.XB_map = []      # background info (like other methods)
        self.obs_map = []     # per-block obs mappings prepared in prepare_analysis
        self.XA_map = None
    # ===== BEGIN: generic grid dump helpers =====


    def _var_index(self, name: str) -> int:
        return list(self.nm.var_names).index(name)

    def _find_block_and_slice(self, var_index: int, level: int):
        """
        Return (block_idx, start, end, lat, lon) for (var, level) within the
        block vectorization defined by nm.mask_cor. None if not found.
        """
        for b_idx, msk in enumerate(self.nm.mask_cor):
            offset = 0
            for (v_info, res) in msk:
                vi, lev = v_info
                lat, lon = res
                n = lat * lon
                if vi == var_index and lev == level:
                    return b_idx, offset, offset + n, lat, lon
                offset += n
        return None

    def _load_state_flat_from_nc(self, nc_file: str, var_index: int, level: int):
        """
        Load a single (var,level) field from a NetCDF file and return as flat (lat*lon,).
        Uses the repo's own loader to keep var naming consistent.
        """
        import numpy as np
        try:
            xs = self.nm.load_netcdf_file(nc_file)  # list of arrays in var_names order
        except Exception as e:
            print(f"[unified-csv] WARNING: failed to load {nc_file}: {e}")
            return None

        arr = xs[var_index]
        vname = self.nm.var_names[var_index]
        # TRG may have tracer dim (1, lev, lat, lon)
        if "TRG" in vname and arr.ndim == 4:
            arr = arr[0, ...]
        # PSG is 2D (lat, lon); others are (lev, lat, lon)
        if "PSG" in vname:
            fld = arr
        else:
            fld = arr[level, :, :]
        return np.asarray(fld, dtype=float).ravel()

    def _dump_unified_flat(self, cycle_k: int):
        """
        ONE flattened CSV per (var,level) in _UNIFIED_DUMP, with columns:
        idx, xb_mean, xa_mean, truth, noda, obs, sigma, is_obs
        Notes:
        - xb_mean, xa_mean from ensemble means in the proper block slice.
        - truth from snapshots/reference_solution_{k}.nc
        - noda  from free_run/free_run_{k}.nc
        - obs/sigma/is_obs from self.obs_map (block-local columns).
        """
        import os
        import numpy as np
        import pandas as pd

        # file templates for truth / noda at cycle k
        ref_nc  = os.path.join(self.nm.snapshots, f"reference_solution_{cycle_k}.nc")
        noda_nc = os.path.join(self.nm.free_run,   f"free_run_{cycle_k}.nc")
        # ---- Unified CSV config: which (var,level) to dump per cycle ----
        # ---- Unified CSV dump configuration ----
        # Dump for five primary level-1 variables (no TG0 etc.)
        # Each tuple = (variable_name, level_index)
        _UNIFIED_DUMP = [
            ("TG1", 7),   # temperature at level 1
            ("UG1", 7),   # zonal wind at level 1
            ("VG1", 7),   # meridional wind at level 1
            ("TRG1", 7),  # humidity traces at level 1
            ("PSG1", 0),  # surface pressure (2-D)
            ]
        # add more tuples as needed
        for vname, lev in _UNIFIED_DUMP:
            # locate field within blocks
            try:
                v_idx = self._var_index(vname)
            except ValueError:
                print(f"[unified-csv] WARNING: var {vname} not found in nm.var_names")
                continue

            info = self._find_block_and_slice(v_idx, lev)
            if info is None:
                print(f"[unified-csv] WARNING: ({vname}, lev={lev}) not present in any block")
                continue

            b_idx, start, end, lat, lon = info
            n = end - start

            # ensure background exists
            if b_idx >= len(self.XB_map) or "XB_b" not in self.XB_map[b_idx]:
                print(f"[unified-csv] WARNING: missing XB for block {b_idx} ({vname}@{lev})")
                continue

            # background mean (xb)
            XB_block = self.XB_map[b_idx]["XB_b"]      # (n_block, Nens)
            xb_slice = XB_block[start:end, :]          # (n, Nens)
            xb_mean  = xb_slice.mean(axis=1)           # (n,)

            # analysis mean (xa) -- available after perform_assimilation
            if self.XA_map is not None and b_idx < len(self.XA_map):
                XA_block = self.XA_map[b_idx]          # (n_block, Nens)
                xa_slice = XA_block[start:end, :]      # (n, Nens)
                xa_mean  = xa_slice.mean(axis=1)       # (n,)
            else:
                xa_mean  = np.full((n,), np.nan)

            # observations (block-local indices)
            obs   = np.full((n,), np.nan)
            sigma = np.full((n,), np.nan)
            is_obs = np.zeros((n,), dtype=int)
            if b_idx < len(self.obs_map):
                om = self.obs_map[b_idx]
                idx_cols = om.get("idx_observed", None)
                y_vals   = om.get("y", None)
                sg_vals  = om.get("sigma", None)  # <-- observation std (obs error)
                if idx_cols is not None and y_vals is not None and sg_vals is not None and idx_cols.size > 0:
                    mask = (idx_cols >= start) & (idx_cols < end)
                    if mask.any():
                        local = idx_cols[mask] - start
                        obs[local]   = y_vals[mask]
                        sigma[local] = sg_vals[mask]
                        is_obs[local] = 1

            # truth and NoDA from saved NetCDFs (best-effort)
            truth = self._load_state_flat_from_nc(ref_nc,  v_idx, lev)
            noda  = self._load_state_flat_from_nc(noda_nc, v_idx, lev)
            if truth is None: truth = np.full((n,), np.nan)
            if noda  is None: noda  = np.full((n,), np.nan)

            # assemble table
            df = pd.DataFrame({
                "idx":     np.arange(n, dtype=int),
                "xb_mean": xb_mean.astype(float),
                "xa_mean": xa_mean.astype(float),
                "truth":   np.asarray(truth, dtype=float),
                "noda":    np.asarray(noda,  dtype=float),
                "obs":     obs.astype(float),
                "sigma":   sigma.astype(float),  # obs std from R diagonal
                "is_obs":  is_obs.astype(int),
            })

            out_dir = os.path.join(self.nm.path)
            os.makedirs(out_dir, exist_ok=True)
            fn = os.path.join(out_dir, f"{vname}_lev{lev}_cycle{cycle_k}.csv")
            df.to_csv(fn, index=False)
            print(f"[unified-csv] {vname}@lev{lev}: wrote {fn} | N={n} obs={int(is_obs.sum())}")


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
        OBS_INCLUDE = getattr(self, "obs_include_vars", None)
        if OBS_INCLUDE is not None and not isinstance(OBS_INCLUDE, set):
            OBS_INCLUDE = set(OBS_INCLUDE)

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
                        # Get observation map (may be None)
            OM = self.obs_map[block_idx] if block_idx < om_len else None

            # Default fallback: inflated background
            XA_block_fallback = self.covariance_inflation(XB)

            # --- base "no obs" gate ---
            if (OM is None) or ("idx_observed" not in OM) or (OM["idx_observed"] is None):
                self.XA_map.append(XA_block_fallback)
                continue

            idx_ob = OM["idx_observed"]
            if idx_ob.size == 0:
                self.XA_map.append(XA_block_fallback)
                continue

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
                continue

            # ============================================================
            # (A) PRE-NORMALIZATION (shared for both linear & nonlinear)
            # ============================================================
            idx_ob = OM["idx_observed"]
            y = OM["y"].reshape(-1)
            sigma = OM["sigma"].reshape(-1)
            m = idx_ob.size

            prior_ens = XB.T.copy()                     # (Nens, n_block)
            X0_obs = prior_ens[:, idx_ob]               # (Nens, m)
            mean_X0 = _np.nanmean(X0_obs, axis=0)       # (m,)
            std_X0  = _np.nanstd(X0_obs, axis=0)
            std_X0  = _np.clip(std_X0, 1e-5, None)      # avoid div/0
            X0_obs_n = (X0_obs - mean_X0) / std_X0      # (Nens, m)

            # ============================================================
            # (B) TOGGLE: LINEAR vs NONLINEAR OBSERVATION HANDLING
            # ============================================================
            if getattr(self, "nonlinear_obs", False):
                # --- Nonlinear case ---
                eps = 1e-8
                sf = float(getattr(self, "scalefact", 1.0))

                # Clamp observations before applying tan() to avoid Inf
                y_clamped = _np.clip(y, -1.55, 1.55)
                tan_y = _np.tan(y_clamped)

                # Nonlinear normalization following Rev_SDE.normalize()
                y_n = _np.arctan(((tan_y / sf - mean_X0) / std_X0) * sf)

                sigma_eff = sigma.copy()
                sigma_eff = _np.where(_np.abs(y_n) < 1.55, sigma_eff, sigma_eff / 1.0e-6)
                sigma_n = sigma_eff / (0.01 * std_X0)

                # Nonlinear ensemble normalization (same transform)
                X0_obs_n_t = _torch.from_numpy(
                    _np.arctan(sf * X0_obs_n).astype(_np.float32)
                ).to(device)
                y_n_t = _torch.from_numpy(y_n.astype(_np.float32)).to(device)
                sigma_n_t = _torch.from_numpy(sigma_n.astype(_np.float32)).to(device)

            else:
                # --- Linear case (original) ---
                y_n = ((y - mean_X0) / std_X0).reshape(-1)
                sigma_n = (sigma / std_X0).reshape(-1)

                X0_obs_n_t = _torch.from_numpy(X0_obs_n.astype(_np.float32)).to(device)
                y_n_t = _torch.from_numpy(y_n.astype(_np.float32)).to(device)
                sigma_n_t = _torch.from_numpy(sigma_n.astype(_np.float32)).to(device)

            # ============================================================
            # (C) INITIALIZE REVERSE SDE STATE (common)
            # ============================================================
            xt = _torch.randn((Nens, m), device=device, dtype=_torch.float32)
            xt_means_hist = _torch.zeros((hist_len, m), device=device, dtype=_torch.float32)
            t = 1.0

            print(f"[ReverseSDE][{label}] starting reverse-SDE (m={m}, Nens={Nens})")
                        # >>> CHANGE D: tiny summary of which vars survived in this block
            try:
                vnames = _block_varnames_array(block_idx, n_block)
                if vnames is not None:
                    import numpy as _np
                    kept = _np.unique(vnames[idx_ob])
                    print(f"[ReverseSDE][{label}] kept vars: {list(kept)}  (m={m})")
            except Exception:
                pass

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
                # =====================================================
                #  TOGGLE: Linear vs Nonlinear likelihood
                # =====================================================
                if self.nonlinear_obs:
                    sf = float(self.scalefact)
                    h_xt = _torch.atan(sf * xt)
                    like_score = -(h_xt - y_n_t) / (sigma_n_t ** 2) * (
                        sf / (1.0 + (sf * xt) ** 2)
                    )
                else:
                    like_score = -(xt - y_n_t) / (sigma_n_t ** 2)
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

                # AFTER (canonical form)
                drift = -( f * xt + (g ** 2) * prior_term - like_tau )
                noise = _torch.sqrt(_torch.tensor(dt, device=device, dtype=xt.dtype)) * g * _torch.randn_like(xt)
                xt_next = xt + dt * drift + noise

                if not _torch.isfinite(xt_next).all():
                    print(f"[ReverseSDE][{label}] State became non-finite at step {i}. Using fallback for this block.")
                    block_numeric_ok = False
                    break

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
        self.map_vector_states()
        # >>> write one unified CSV per requested field, now that analysis exists
        self._dump_unified_flat(cycle_k=getattr(self, "current_cycle_k", 0))






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
          



    
    
    
    
        
    
