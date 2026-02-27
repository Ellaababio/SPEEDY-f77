# My Approach to Wind Assimilation in ReverseSDE

Here is a brief overview of how I implemented wind speed and direction assimilation. My goal was to stick strictly to the **score-based approach** Zixiang used in his `Rev_SDE_vanilla.py`, where we analytically calculate the gradient of the observation operator $\nabla h(\mathbf{x})$ and inject it into the score function.

## 1. Generating Synthetic Wind Data
**File:** `amlcs/observation.py`
First, I had to generate the synthetic observations. Since `WDG1` (direction) and `WSG1` (speed) are *derived* from `UG1` and `VG1`, I calculate them from the reference truth `xs`.

For direction, I use `arctan2(v, u)` to get the true angle, and for speed, `sqrt(u^2 + v^2)`. I then add Gaussian noise just like we do for other variables. This happens right where we build the "truth" for the experiment.

```python
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
                              true_wdg = np.arctan2(v_lev, u_lev)
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
```

---

## 2. Linking Operations to State Variables
**File:** `amlcs/sequential_methods.py`

This was the tricky part. In `Rev_SDE_vanilla.py`, Zixiang passes `indx_indxob_nonlinear` to tell the code which indices are nonlinear. Since my grid is processed in blocks, I have to dynamically find which observations "belong" to the current block (i.e., the `UG1` or `VG1` chunk I'm updating).

I do this by finding the intersection (`common_mask`) between the current block's grid indices (`idx_ob`) and the observation stations (`idx_wdg`). If they overlap, I enable the wind update for this block.

```python
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
```

---

## 3. The Nonlinear Score Update (Following Zixiang's Method)
**File:** `amlcs/sequential_methods.py`

This is the core of the method. In `Rev_SDE_vanilla.py`, Zixiang adds the *nonlinear* likelihood score:

$$
\text{score} = - \frac{(h(\mathbf{x}) - \mathbf{y})}{\sigma^2} \nabla h(\mathbf{x})
$$

For wind direction $h(u, v) = \text{atan2}(v, u)$, the gradients are:
$$
\frac{\partial h}{\partial u} = \frac{-v}{u^2 + v^2}, \quad \frac{\partial h}{\partial v} = \frac{u}{u^2 + v^2}
$$

I implemented exactly this. I grab the partner state (pair $u$ with $v$), calculate the residual `diff` (handling the $\pi$ wrapping for direction), compute the gradients `grad_h_xphy`, and then add it to `like_score`.

This is **exactly** analogous to how Zixiang handled `arctan(x)` in his vanilla code, just generalized to 2D vector fields.

```python
                # FOR "WIND-ONLY" SYNTHESIZED BLOCKS:
                # The linear score is based on dummy observations (y=0, sigma=huge).
                # To be absolutely safe, we force it to zero here.
                if is_synthesized_wind:
                     like_score = _torch.zeros_like(like_score)

                # --- (E) ADD NONLINEAR WIND SCORES ---
                if is_wind_update and partner_state is not None:
                     # 1. De-normalize current state
                     x_phy = mean_X0 + xt * std_X0 
                     
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
                         # WDG = atan2(v, u)
                         pred_wdg = _torch.atan2(v_vec, u_vec) # (Nens, m)
                         
                         # Residual (Innovation)
                         diff = pred_wdg - obs_wdg_vals 
                         # Wrap to [-pi, pi]
                         diff = (diff + _np.pi) % (2 * _np.pi) - _np.pi
                         
                         # Gradient of h w.r.t current state
                         if wind_mode == "updating_u":
                             grad_h_xphy = -v_vec / res_sq
                         else:
                             grad_h_xphy = u_vec / res_sq
                             
                         # Chain rule
                         grad_h_xt = grad_h_xphy * std_X0
                         
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
                             
                         grad_wsg_xt = grad_wsg_xphy * std_X0
                         
                         wsg_score = -(diff_wsg / (obs_wsg_sigma**2)) * grad_wsg_xt
                         wsg_score = _torch.where(obs_wsg_mask, wsg_score, _torch.zeros_like(wsg_score))
                         like_score = like_score + wsg_score
                
                like_tau = tau * like_score
                pull = (g ** 2) * like_tau
```
# My Approach to Wind Assimilation in EnKF (EnKF_MC_obs)

The existing EnKF implementation in `amlcs` uses a specific matrix-based formulation (solving the analysis equation using an explicit $\mathbf{H}$ matrix and a Cholesky-decomposed background covariance, $\mathbf{B}_{inv}^{1/2}$). 

My goal was to implement wind assimilation as a **natural extension of this specific codebase**. Since the solver (`perform_assimilation_block`) is already designed to ingest linearized operators ($\mathbf{H}$) and observation error matrices ($\mathbf{R}$), I adopted an approach that dynamically constructs these matrices for the nonlinear wind variables.

## 1. Dynamic Observation Operator Extension and Linearization
**File:** `amlcs/sequential_methods.py`

In the standard EnKF implementation, we typically map state variables directly to observations (identity matrix). However, for wind speed (`WSG1`) and direction (`WDG1`), the relationship is nonlinear: $h(u, v) = \sqrt{u^2+v^2}$ and $h(u, v) = \text{atan2}(v, u)$.

To handle this within the existing linear framework, I dynamically extend the observation vector `Ys`, the model equivalent `Hb_X`, and the observation error covariance `R`. This involves:

1. **Retrieving Partner Variabes**: When processing a `u` block, I find the corresponding `v` block (and vice versa) to get the full 2D wind vector. 

2. **Helper Function `process_wind_type`**: This function iterates through available wind observations, finds their corresponding grid points, and computes:
    - **Innovation**: The difference between observed and predicted wind (using full nonlinear $h(\mathbf{x})$).
    - **Jacobian Row ($\mathbf{H}$)**: The tangent linear gradient of $h(\mathbf{x})$ evaluated at the **ensemble mean** ($\bar{u}, \bar{v}$), which allows us to treat the nonlinear observation as a linearized constraint.

Here is the complete logic implementation:

```python
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
                                           
                                           # Grading (Linearization) - CRITICAL STEP
                                           # We linearize around the ensemble mean to get H
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
                       def calc_wdg(u,v): return np.arctan2(v,u)
                       def calc_wsg(u,v): return np.sqrt(u**2 + v**2)
                       
                       def grad_wdg(u,v,mode):
                           s2 = max(u**2+v**2, 1e-6)
                           return -v/s2 if mode=='u' else u/s2
                       
                       def grad_wsg(u,v,mode):
                           sm = max(np.sqrt(u**2+v**2), 1e-6)
                           return u/sm if mode=='u' else v/sm
                       
                       # Process with configured sigmas
                       # Default to tuned values if not in config
                       sigma_wdg = self.wind_err.get('WDG1', 0.2)
                       sigma_wsg = self.wind_err.get('WSG1', 1.0)
                       
                       process_wind_type('WDG1', calc_wdg, grad_wdg, sigma_wdg, True)
                       process_wind_type('WSG1', calc_wsg, grad_wsg, sigma_wsg, False)
```

This approach ensures that wind assimilation fits seamlessly into the existing EnKF structure without requiring a fundamental change to the algorithm (like an iterative solver or particle filter), while still capturing the first-order effects of wind nonlinearity via the linearized gradients. 

**Note**: Because the solver updates one state block at a time, each wind observation contributes a Jacobian entry only for the currently-updated variable (u or v), treating the partner component as fixed for that block update.

## 3. Implementation Note: Consistency with Codebase Architecture

The core `EnKF_MC_obs.perform_assimilation_block` method assumes the analysis update can be solved via the linear system:
$$ \mathbf{P} = (\mathbf{B}_{inv} + \mathbf{H}^T \mathbf{R}^{-1} \mathbf{H})^{-1} $$
This structure **requires** explicit $\mathbf{H}$ and $\mathbf{R}$ matrices as inputs.

While a "Standard EnKF" often avoids explicit matrices by using ensemble covariances in observation space, adopting that approach here would have required rewriting the entire solver logic. Instead, I chose to **extend the existing framework** by:
1.  **Linearizing Locally**: Constructing the $\mathbf{H}$ matrix rows using the tangent linear gradient of wind speed/direction at the ensemble mean.
2.  **Preserving Solver Logic**: This allows the nonlinear wind observations to be passed into the *exact same* solver routine as all other variables, ensuring code consistency and maintainability.

This makes the wind assimilation a direct, logical extension of the existing `perform_assimilation_block` implementation.

