import os
import numpy as np
import pandas as pd

# Initialization of system parameters for the Data Stream simulation
N = 25000
fi = 0.95 # Autoregression coefficient (phi) representing system inertia
stability_factor = 0.01

# Selected variables for the multivariate Data Stream
columns = [
    'SPEEDKNOTS', 'MERPM', 'MEPOWER', 
    'RANGEX', 'RANGEY', 'INCLINOMETERXZC', 'INCLINOMETERYZC'
]

# Initial mean vector representing the normal operational state
means = np.array([
    5.74,       # SPEEDKNOTS
    17.29,      # MERPM
    3268.67,    # MEPOWER
    12.44,      # RANGEX
    12.14,      # RANGEY
    10.79,      # INCLINOMETERXZC
    6.56        # INCLINOMETERYZC
])

# Standard deviations corresponding to the selected variables
std_devs = np.array([
    5.20,       # SPEEDKNOTS
    22.22,      # MERPM
    5553.74,    # MEPOWER
    15.05,      # RANGEX
    8.95,       # RANGEY
    14.21,      # INCLINOMETERXZC
    9.27        # INCLINOMETERYZC
])

# Correlation matrix defining the interdependence of the variables
R = np.array([
    # SPD   RPM   POW    RX     RY     XZC    YZC
    [ 1.00, 0.85, 0.80, -0.25, -0.30, -0.15, -0.25], # SPEED
    [ 0.85, 1.00, 0.95,  0.15,  0.20,  0.10,  0.15], # RPM
    [ 0.80, 0.95, 1.00,  0.20,  0.25,  0.15,  0.20], # POWER
    [-0.25, 0.15, 0.20,  1.00,  0.60,  0.70,  0.40], # RANGEX
    [-0.30, 0.20, 0.25,  0.60,  1.00,  0.40,  0.70], # RANGEY
    [-0.15, 0.10, 0.15,  0.70,  0.40,  1.00,  0.50], # XZC
    [-0.25, 0.15, 0.20,  0.40,  0.70,  0.50,  1.00]  # YZC
])

# Calculation of the initial covariance matrix based on standard deviations and correlation
D = np.diag(std_devs)
cov_matrix = D @ R @ D

# Adjustment of the covariance matrix using the stability factor to restrict white noise variance
adjusted_cov = cov_matrix * stability_factor * (1 - fi**2)

# The mean vector for the multivariate white noise is zero
mean_noise = np.zeros(7) 

dynamic_means = np.zeros((N, 7))

# --- Definition of dynamic means for Epoch simulation via concept drifts ---

# 1. Rows 0 - 499: Initial Epoch (Normal operational state)
dynamic_means[0:500] = means 

# 2. Rows 500 - 999: Port conditions Epoch
dynamic_means[500:1000] = [0.0, 0.0, 0.0, 2.0, 2.0, 1.0, 1.0]

# 3. Rows 1000 - 1999: Voyage initiation Epoch
dynamic_means[1000:2000] = [14.0, 70.0, 20000.0, 10.0, 10.0, 8.0, 5.0]

# 4. Rows 2000 - 3999: Storm conditions Epoch (Proportional recovery based on RANGEX)
dynamic_means[2000:3000] = [14.0, 75.0, 25000.0, 35.0, 30.0, 25.0, 18.0]
dynamic_means[3000:3400] = [14.0, 74.5, 24600.0, 33.0, 28.3, 23.6, 17.0] 
dynamic_means[3400:3700] = [14.0, 73.4, 23600.0, 28.0, 24.2, 20.1, 14.5] 
dynamic_means[3700:4000] = [14.0, 71.5, 22000.0, 20.0, 17.5, 14.5, 10.5] 

# 5. Rows 4000 - 7999: Calm sea Epoch (Proportional acceleration based on SPEEDKNOTS)
dynamic_means[4000:6000] = [14.0, 68.0, 19000.0, 5.0, 5.0, 4.0, 3.0]
dynamic_means[6000:6800] = [14.2, 68.8, 19800.0, 5.1, 5.1, 4.1, 3.1] 
dynamic_means[6800:7500] = [15.0, 72.2, 23000.0, 5.2, 5.2, 4.2, 3.2]
dynamic_means[7500:8000] = [16.0, 76.5, 27000.0, 5.5, 5.5, 4.5, 3.5]

# 6. Rows 8000 - 15999: Increased speed Epoch
dynamic_means[8000:11000] = [18.0, 85.0, 35000.0, 6.0, 6.0, 5.0, 4.0]
dynamic_means[11000:12000] = [17.9, 84.7, 34800.0, 6.3, 6.3, 5.2, 4.2]
dynamic_means[12000:13000] = [17.7, 83.9, 34100.0, 7.3, 7.1, 6.0, 4.8] 
dynamic_means[13000:14000] = [17.2, 82.1, 32800.0, 9.2, 8.8, 7.4, 6.0] 
dynamic_means[14000:14800] = [16.7, 80.0, 31200.0, 11.6, 10.8, 9.2, 7.5] 
dynamic_means[14800:15400] = [15.7, 76.4, 28400.0, 15.7, 14.3, 12.1, 10.0] 
dynamic_means[15400:15800] = [14.6, 72.2, 25200.0, 20.6, 18.4, 15.7, 13.0] 
dynamic_means[15800:16000] = [13.0, 66.4, 20800.0, 27.0, 24.0, 20.5, 17.0] 

# 7. Rows 16000 - 25000: Severe storm Epoch 
dynamic_means[16000:18000] = [10.0, 55.0, 12000.0, 40.0, 35.0, 30.0, 25.0]
dynamic_means[18000:19500] = [10.0, 55.1, 12100.0, 39.7, 34.7, 29.8, 24.8] 
dynamic_means[19500:21000] = [10.1, 55.4, 12200.0, 39.0, 34.2, 29.3, 24.4] 
dynamic_means[21000:22500] = [10.3, 55.8, 12400.0, 37.8, 33.1, 28.4, 23.6] 
dynamic_means[22500:24000] = [10.5, 56.8, 13000.0, 35.2, 30.9, 26.5, 22.0]
dynamic_means[24000:25000] = [11.1, 58.7, 14000.0, 30.2, 26.6, 22.7, 18.8]


# --- Directory Setup for Datasets ---
output_dir = os.path.join('..', 'synthetic_data')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Data Stream Generation ---
base_seed = 42 # Base seed for reproducibility

for run_id in range(1, 11): # Loop from 1 to 10
    
    # Set the unique seed for this specific run
    np.random.seed(base_seed + run_id)
    
    # Initialization of the dataset array for the simulated Data Stream
    data = np.zeros((N, 7))

    # Establishment of the initial state (n=0)
    initial_state = means

    data[0] = initial_state + np.random.multivariate_normal(mean_noise, adjusted_cov)

    # Generation of the synthetic Data Stream using the Autoregressive Model of First Order AR(1)
    for i in range(1, N):
        e_n = np.random.multivariate_normal(mean_noise, adjusted_cov)
        data[i] = (fi * data[i-1]) + ((1 - fi) * dynamic_means[i]) + e_n

    # --- Application of physical and mechanical limits (clipping) ---
    # 1. SPEEDKNOTS: Restricted between 0 and 26 knots
    data[:, 0] = np.clip(data[:, 0], a_min=0.0, a_max=26.0)

    # 2. MERPM: Restricted between -31.20 (Astern) and 104.0 (Full Ahead)
    data[:, 1] = np.clip(data[:, 1], a_min=-31.20, a_max=104.0)

    # 3. MEPOWER: Restricted between 0 and 57100 kW
    data[:, 2] = np.clip(data[:, 2], a_min=0.0, a_max=57100)

    # 4. RANGEX & RANGEY: Restricted between 0 and 90 degrees
    data[:, 3] = np.clip(data[:, 3], a_min=0.0, a_max=90)
    data[:, 4] = np.clip(data[:, 4], a_min=0.0, a_max=90)

    # 5. INCLINOMETER ZC: Restricted to a maximum of 60 crossings
    data[:, 5] = np.clip(data[:, 5], a_min=0.0, a_max=60.0)
    data[:, 6] = np.clip(data[:, 6], a_min=0.0, a_max=60.0)
    # ----------------------------------------------

    # Rounding values to 2 decimal places to simulate real telemetry sensors
    data = np.round(data, 2)

    # Export the generated Data Stream to an Excel file in the designated folder
    df = pd.DataFrame(data, columns=columns)
    
    filename = os.path.join(output_dir, f'synthetic_ship_dataset_{run_id}.xlsx')
    df.to_excel(filename, index=False)

    print(f"Dataset {run_id}/10 successfully generated and saved to {filename}")

print("All datasets have been created successfully")