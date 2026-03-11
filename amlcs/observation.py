import numpy as np
import scipy.sparse as spa
from netCDF4 import Dataset


####################################################################################
####################################################################################
####################################################################################
class observation:
      H_vect = None; #Observation operator as vector
      H_spar = None; #Observation operator as matrix sparse
      H_grid = None; #Observation operator as numerical grid
      R_invs = None;
      R_spar = None;
      R_vect = None;
      R_sqrt = None;
      
      obs_network = None;
      obs_H = None;
      obs_n = None;
      obs_m = None;
      obs_H_sparse = None;
      obs_R_sparse = None;
      
      m = -1; #Number of observations
      m_var = -1; #Number of observations for each variable.
      err_obs = None;
      y = None;
      y_obs = None;
      
      #s denotes the sparsity pattern per layer
      #nm - numerical model
      #gr - grid
      
      obs_var = None;
      
      def __init__(self,err_obs, obs_indexes, nonlinear_obs=False, scalefact=1.0):
          self.err_obs = err_obs;
          self.obs_var = obs_indexes;
          self.nonlinear_obs = bool(nonlinear_obs)
          self.scalefact = float(scalefact)
          
          
 
      def get_stations_variables(self, lat, lon, p, s):
          S = np.zeros((lat, lon));
          for i in range(0,lat,s):
              for j in range(0,lon,s):
                  S[i,j] = 1;
          S_res = S.reshape(lon*lat, order='C');
          pos,=np.where(S_res>0);
          return list(pos+p);
          
          
      def build_observational_network(self, gr, nm, s):
          mesh = gr.mesh;
          mask = nm.mask_cor;
          self.obs_H = [];
          self.obs_n = [];
          self.obs_m = [];
          self.wind_obs = {} # Dictionary to store wind obs info: { 'WDG1': {lev: {'H':..., 'n':...}}, 'WSG1': ... }

          for mesh_, mask_ in zip(mesh, mask): #moving across meshes
              n_mesh = 0;
              m_mesh = 0;
              H_vect = [];
              o_data = [];
              p = 0;
              for v, n in zip(mesh_, mask_): #moving across variables
                  #exit();
                  mes_info = v[0];
                  res_info = v[1];
                  lat, lon = res_info[0], res_info[1];
                  #print(f'n[0] reads {n[0]}');
                  var_index = n[0][0]; #n[0] = [var_index, level]
                  
                  # Check if observations are available in this layer and this variable
                  # Handle case where obs_var is shorter than number of model variables (e.g. WDG1/WSG1 added but not in runner csv)
                  is_observed = False
                  if var_index < len(self.obs_var):
                      is_observed = self.obs_var[var_index]
                  
                  if is_observed: 
                     H_l = self.get_stations_variables(lat, lon, n_mesh, s);
                     m_mesh = len(H_l);
                     H_vect.extend(H_l);
                     o_data.append([n[0], m_mesh]);
                  n_mesh+= lat * lon; #we skip indexes from being observed at this layer-var
              self.obs_H.append(np.array(H_vect));
              self.obs_n.append(n_mesh);
              self.obs_m.append(o_data);
          
          # -------------------------------------------------------------------
          # Capture Wind Observations (WDG1/WSG1) - Indices 10 and 11
          # These are NOT in the state vector (mask_cor), so we handle them separately.
          # We assume they exist on the same grid (lat, lon) as other variables.
          # -------------------------------------------------------------------
          wind_map = {10: 'WDG1', 11: 'WSG1'}
          for w_idx, w_name in wind_map.items():
              if w_idx < len(self.obs_var) and self.obs_var[w_idx]:
                  self.wind_obs[w_name] = {}
                  # We assume wind obs are available at all levels 0..7
                  # We define their "H" relative to a theoretical grid of size lat*lon
                  # Since they are not in the state vector, we can't build a global sparse H matrix for them relative to 'x'.
                  # Instead, we just store the station indices relative to a single level's grid.
                  
                  # Use standard resolution (t21: 32x64)
                  # We can get this from nm.gs or just assume it matches UG1
                  lat, lon = gr.lat, gr.lon
                  
                  # Get station indices for a single level (p=0)
                  # Note: 's' (stride) is passed to build_observational_network
                  stations = self.get_stations_variables(lat, lon, 0, s)
                  m_obs = len(stations)
                  
                  for lev in range(8): # 8 levels
                      self.wind_obs[w_name][lev] = {
                          'stations': stations,
                          'm': m_obs,
                          'lat': lat,
                          'lon': lon
                      }
          
          self.build_observational_operator();
          self.build_data_error_covariance(gr, nm);
      
     
      def build_observational_operator(self):
          obs_H = self.obs_H;
          obs_n = self.obs_n;
          n_sub = len(obs_H);
          self.obs_H_sparse = [];
          for s in range(0, n_sub):
              n = obs_n[s];
              J = obs_H[s]; #indexes wherein observations are located
              V = np.ones_like(J);
              m = J.size;
              I = np.arange(0, m);
              H_sparse = spa.coo_matrix((V,(I,J)), shape=(m, n));
              self.obs_H_sparse.append(H_sparse);
      
      def build_data_error_covariance(self, gr, nm):
          obs_m = self.obs_m;
          self.obs_R_sparse = [];
          for om in obs_m:
              #print('* om es {0}'.format(om)); #[[var, level], number of observations]
              m_om = 0;
              Ig = [];
              V_R_g = [];
              V_Ri_g = [];
              V_Rs_g = [];
              for o in om:
                  var_info = o[0]
                  variable = var_info[0];
                  try:
                      err_obm = self.err_obs[variable]
                  except IndexError:
                      # WDG1/WSG1 (10/11) should not be reached here in standard loop
                      print(f"Warning: No obs error specified for var idx {variable}. Using default 1.0")
                      err_obm = 1.0
                  m = o[1];
                  I = np.arange(m_om, m_om+m);
                  Ig.extend(list(I));
                  V_R_g.extend(list((err_obm**2)*np.ones_like(I)))
                  V_Ri_g.extend(list((1/(err_obm**2))*np.ones_like(I)))
                  V_Rs_g.extend(list(err_obm*np.ones_like(I)))
                  m_om+=m;
               
              Ig = np.array(Ig);
              V_R_g = np.array(V_R_g); 
              V_Ri_g = np.array(V_Ri_g);  
              V_Rs_g = np.array(V_Rs_g);  
              
              #print([Ig.shape, V_R_g.shape]);   

              
              R_sparse = spa.coo_matrix((V_R_g,(Ig,Ig)), shape=(m_om, m_om));
              Rinv_sparse = spa.coo_matrix((V_Ri_g,(Ig,Ig)), shape=(m_om, m_om));
              Rsqr_sparse = spa.coo_matrix((V_Rs_g,(Ig,Ig)), shape=(m_om, m_om));
              
              self.obs_R_sparse.append({'R':R_sparse, 'Ri':Rinv_sparse, 'Rs':Rsqr_sparse, 'm':m_om});
           

          
          
      
      def build_synthetic_observations(self, nm, rs, M):
          reference_abs = nm.path+'reference/';
          reference_path = reference_abs+'snapshots/';
          mask_cor = nm.mask_cor;
          var_names = nm.var_names;
          self.y_obs = [];
          
          # Prepare wind errors if needed
          wind_map = {10: 'WDG1', 11: 'WSG1'}
          wind_errors = {}
          for w_idx, w_name in wind_map.items():
              if w_name in self.wind_obs:
                   try:
                       wind_errors[w_name] = self.err_obs[w_idx]
                   except Exception:
                       wind_errors[w_name] = 1.0
          
          for s in range(0, M):
              xs = rs.x_ref[s]; 
              y_ma = [];
              for block_idx, (ma, R_data, H_sparse) in enumerate(zip(mask_cor, self.obs_R_sparse, self.obs_H_sparse)):
                  x_data = [];
                  for m in ma:
                      var_info = m[0];
                      variable = var_info[0];
                      level = var_info[1];
                      var_name = var_names[variable];
                      #print([var_name, m[1]]);
                      if 'TRG' in var_name:
                         x_ma = xs[variable][level,:,:].reshape((-1,1));
                      elif 'PSG' in var_name:
                         x_ma = xs[variable][:,:].reshape((-1,1));
                      else:
                         x_ma = xs[variable][level,:,:].reshape((-1,1));
                      x_data.extend(list(x_ma));
                      #print(x_ma);
                      
                  x_data = np.array(x_data);
                  R_sqrt = R_data['Rs'];
                  m_obs = R_sqrt.size;
                  
                  # Apply observation operator Hx
                  Hx = H_sparse @ x_data
                  
                  # Apply nonlinear operator if requested (Standard Vars)
                  if self.nonlinear_obs:
                      # Nonlinear observation operator: h(x) = arctan(sf * x)
                      Hx = np.arctan(self.scalefact * Hx)
                  
                  # Add noise
                  y = Hx + R_sqrt @ np.random.randn(m_obs,1);
                  y_ma.append(y);
              
              if not hasattr(self, 'y_wind_obs'):
                   self.y_wind_obs = [] # List of dicts, one per time step
              
              y_wind_t = {}
              
              # Calculate wind fields from Reference State (xs)
              # We need UG1 (idx 5) and VG1 (idx 6)
              # Assuming standard var_names order: 
              # 'UG0','VG0','TG0','TRG0','PSG0','UG1','VG1','TG1','TRG1','PSG1'
              try:
                  u_idx = var_names.index('UG1')
                  v_idx = var_names.index('VG1')
                  ug1_ref = xs[u_idx]
                  vg1_ref = xs[v_idx]
                  
                  for w_name in self.wind_obs: # WDG1, WSG1
                      # DEBUG PRINT
                      if s == 0: print(f"[DEBUG] Generating synthetic obs for {w_name}")
                      y_wind_t[w_name] = {}
                      err_std = wind_errors.get(w_name, 1.0)
                      
                      for lev in range(8):
                          # Get stations
                          stations = self.wind_obs[w_name][lev]['stations']
                          
                          # Extract U/V at stations
                          u_lev = ug1_ref[lev,:,:].flatten()[stations]
                          v_lev = vg1_ref[lev,:,:].flatten()[stations]
                          
                          if w_name == 'WDG1':
                              # True Wind Direction
                              true_wdg = np.arctan2(u_lev, v_lev)
                              # Add Noise
                              obs_wdg = true_wdg + err_std * np.random.randn(len(stations))
                              y_wind_t[w_name][lev] = obs_wdg
                              
                          elif w_name == 'WSG1':
                              # True Wind Speed
                              true_wsg = np.sqrt(u_lev**2 + v_lev**2)
                              # Add Noise
                              obs_wsg = true_wsg + err_std * np.random.randn(len(stations))
                              y_wind_t[w_name][lev] = obs_wsg
                              
              except Exception as e:
                  print(f"Warning: Could not generate synthetic wind obs: {e}")

              self.y_obs.append(y_ma);
              self.y_wind_obs.append(y_wind_t)
          
          print('* ENDJ - Synthetic observations have been created');
          
  
      def get_perturbed_observations(self, block, k, N):
          np.random.seed(seed=10+k); #To replicate observations
          y_k = self.y_obs[k];
          y_block = y_k[block];
          R_sqrt_block = self.obs_R_sparse[block]['Rs'];
          H_block = self.obs_H_sparse[block];
          m_block = self.obs_R_sparse[block]['m'];
          Ys_block = y_block + R_sqrt_block @ np.random.randn(m_block, N);
          return Ys_block;
      
      def get_R(self, block):
          return self.obs_R_sparse[block]['R'];

      def get_H(self, block):
          return self.obs_H_sparse[block];

      def get_synthetic_noise(self,N):
          return self.R_sqrt @ np.random.randn(self.m,N);
