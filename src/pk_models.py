import numpy as np

class PKModel:
    """Base class for Pharmacokinetic Models (3-Compartment)"""
    def __init__(self):
        self.params = {}

    def get_params(self, age, weight, height, sex):
        raise NotImplementedError

    def step(self, state, infusion_rate, dt=1.0):
        """
        Euler integration step.
        state: [c1, c2, c3, ce] (amounts or concentrations depending on model)
        infusion_rate: drug input (unit/min)
        """
        raise NotImplementedError

class MarshPropofol(PKModel):
    """
    Marsh Model for Propofol.
    V1 is proportional to weight. k parameters are constant.
    """
    def __init__(self):
        super().__init__()
        
    def get_params(self, weight_kg):
        # Marsh constants
        # V1 = 0.228 L/kg
        # V2 = 0.463 L/kg
        # V3 = 2.893 L/kg
        # k10 = 0.119 min^-1
        # k12 = 0.112 min^-1
        # k13 = 0.042 min^-1
        # k21 = 0.055 min^-1
        # k31 = 0.0033 min^-1
        # ke0 = 0.26 min^-1 (commonly used value, though original Marsh didn't specify Ke0)
        
        v1 = 0.228 * weight_kg
        v2 = 0.463 * weight_kg
        v3 = 2.893 * weight_kg
        
        k10 = 0.119
        k12 = 0.112
        k13 = 0.042
        k21 = 0.055
        k31 = 0.0033
        ke0 = 0.26 
        
        return {
            'v1': v1, 'v2': v2, 'v3': v3,
            'k10': k10, 'k12': k12, 'k13': k13,
            'k21': k21, 'k31': k31, 'ke0': ke0
        }

class SchniderPropofol(PKModel):
    """
    Schnider Model for Propofol.
    Parameters depend on Age, Weight, Height, Sex (LBM).
    """
    def get_lbm(self, weight, height, sex):
        # James Equation for LBM
        # weight in kg, height in cm
        if sex == 'M':
            lbm = 1.1 * weight - 128 * (weight / height)**2
        else:
            lbm = 1.07 * weight - 148 * (weight / height)**2
        return lbm

    def get_params(self, age, weight, height, sex):
        lbm = self.get_lbm(weight, height, sex)
        
        # V1, V3 are fixed
        v1 = 4.27
        v3 = 238
        
        # V2 depends on age
        v2 = 18.9 - 0.391 * (age - 53)
        
        # Clearances
        cl1 = 1.89 + 0.0456 * (weight - 77) - 0.0681 * (lbm - 59) + 0.0264 * (height - 177)
        cl2 = 1.29 - 0.024 * (age - 53)
        cl3 = 0.836
        
        # Calculate rate constants
        k10 = cl1 / v1
        k12 = cl2 / v1
        k21 = cl2 / v2
        k13 = cl3 / v1
        k31 = cl3 / v3
        
        ke0 = 0.456 # min^-1
        
        return {
            'v1': v1, 'v2': v2, 'v3': v3,
            'k10': k10, 'k12': k12, 'k13': k13,
            'k21': k21, 'k31': k31, 'ke0': ke0
        }

class MintoRemifentanil(PKModel):
    """
    Minto Model for Remifentanil.
    Depends on Age, Weight, Height, Sex (LBM).
    """
    def get_lbm(self, weight, height, sex):
        if sex == 'M':
            lbm = 1.1 * weight - 128 * (weight / height)**2
        else:
            lbm = 1.07 * weight - 148 * (weight / height)**2
        return lbm

    def get_params(self, age, weight, height, sex):
        lbm = self.get_lbm(weight, height, sex)
        
        # Base parameters
        v1 = 5.1 - 0.0201 * (age - 40) + 0.072 * (lbm - 55)
        v2 = 9.82 - 0.0811 * (age - 40) + 0.108 * (lbm - 55)
        v3 = 5.42
        
        cl1 = 2.6 - 0.0162 * (age - 40) + 0.0191 * (lbm - 55)
        cl2 = 2.05 - 0.0301 * (age - 40)
        cl3 = 0.076 - 0.00113 * (age - 40)
        
        k10 = cl1 / v1
        k12 = cl2 / v1
        k21 = cl2 / v2
        k13 = cl3 / v1
        k31 = cl3 / v3
        
        ke0 = 0.595 - 0.007 * (age - 40)
        
        return {
            'v1': v1, 'v2': v2, 'v3': v3,
            'k10': k10, 'k12': k12, 'k13': k13,
            'k21': k21, 'k31': k31, 'ke0': ke0
        }

def simulate_pk(params, infusion_array, dt=1.0/60.0):
    """
    Simulate PK model for a single patient.
    infusion_array: numpy array of infusion rates (mg/min or mcg/min) per second? 
                    Actually, if dt is small, we can just iterate.
                    But python loop is slow.
    
    We can use scipy.integrate.odeint or just manual Euler/Runge-Kutta.
    For PK models, analytical solution exists for constant infusion, but since infusion changes every second, 
    discrete update is better.
    
    State: x = [A1, A2, A3, Ce] (Amount in compartments, and Effect Site Conc)
    dA1/dt = I(t) - (k10 + k12 + k13)*A1 + k21*A2 + k31*A3
    dA2/dt = k12*A1 - k21*A2
    dA3/dt = k13*A1 - k31*A3
    dCe/dt = ke0*(Cp - Ce)  where Cp = A1 / V1
    
    Matrix form: dx/dt = M x + u
    """
    k10, k12, k13 = params['k10'], params['k12'], params['k13']
    k21, k31, ke0 = params['k21'], params['k31'], params['ke0']
    v1 = params['v1']
    
    # Precompute transition matrix for discrete step dt
    # This is an approximation. For exact solution, use matrix exponential.
    # Given the high sampling rate (1Hz = 1/60 min), Euler is probably fine, but RK4 or exact is safer.
    # Let's use analytical solution for constant input over dt.
    # x(t+dt) = e^(A*dt) * x(t) + (e^(A*dt) - I) * A^-1 * B * u
    # But that's heavy.
    # Simple Euler: x_new = x + (Ax + Bu)*dt
    
    # System Matrix A (units: min^-1)
    # x = [A1, A2, A3, Ce]
    # Cp = A1 / V1
    # dCe/dt = ke0 * A1/V1 - ke0 * Ce
    
    n_steps = len(infusion_array)
    # infusion_array is per second or per minute?
    # Usually we process at 1Hz (dt = 1/60 min).
    # Infusion input u(t) is in mg/min.
    # So in time dt (min), amount added is u(t) * dt.
    
    # Construct A matrix
    # [ -(k10+k12+k13)   k21    k31    0   ]
    # [    k12          -k21     0     0   ]
    # [    k13            0    -k31    0   ]
    # [   ke0/v1          0      0   -ke0  ]
    
    A = np.array([
        [-(k10+k12+k13), k21, k31, 0],
        [k12, -k21, 0, 0],
        [k13, 0, -k31, 0],
        [ke0/v1, 0, 0, -ke0]
    ])
    
    # We can precompute Propagator P = (I + A*dt) for Euler
    # Or P = expm(A*dt) for exact
    # Since dt is constant, let's try approximate Euler first for speed.
    # 1 Hz -> dt = 1/60.
    
    P = np.eye(4) + A * dt
    
    # States array: (N, 4)
    states = np.zeros((n_steps, 4))
    
    # Input vector B (only affects A1)
    # u is scalar infusion rate. B = [1, 0, 0, 0]^T
    
    x = np.zeros(4)
    
    # Loop
    # Vectorized scan is hard in pure numpy without explicit loop or scipy lfilter.
    # We can cast this as a filter design? 
    # Or just use numba for speed.
    
    for i in range(n_steps):
        u = infusion_array[i] # rate in mass/min
        
        # Euler update
        # x = P @ x
        # x[0] += u * dt
        
        # Unroll for speed
        a1, a2, a3, ce = x
        
        da1 = u - (k10 + k12 + k13)*a1 + k21*a2 + k31*a3
        da2 = k12*a1 - k21*a2
        da3 = k13*a1 - k31*a3
        dce = ke0*(a1/v1 - ce)
        
        x[0] += da1 * dt
        x[1] += da2 * dt
        x[2] += da3 * dt
        x[3] += dce * dt
        
        states[i] = x
        
    return states # Returns [A1, A2, A3, Ce]

class ApproximateEleveldPropofol(PKModel):
    """
    Approximate Eleveld 2018 Propofol Model.
    Based on Reference Values for 35y, 70kg Male and Standard Allometric Scaling.
    Ref: Eleveld et al., BJA 2018.
    """
    def __init__(self):
        super().__init__()
        # Reference Values (35y, 70kg, 170cm, Male)
        self.ref_vals = {
            'V1': 6.28,
            'V2': 25.5,
            'V3': 273.0,
            'Cl': 1.79,
            'Q2': 1.75,
            'Q3': 1.11,
            'ke0': 0.146
        }
        self.ref_wgt = 70.0

    def get_params(self, age, weight, height=170, sex='M'):
        # Allometric Scaling Factors
        w_ratio = weight / self.ref_wgt
        
        v1 = self.ref_vals['V1'] * w_ratio
        v2 = self.ref_vals['V2'] * w_ratio
        v3 = self.ref_vals['V3'] * w_ratio
        
        cl = self.ref_vals['Cl'] * (w_ratio ** 0.75)
        q2 = self.ref_vals['Q2'] * (w_ratio ** 0.75)
        q3 = self.ref_vals['Q3'] * (w_ratio ** 0.75)
        
        if age > 35:
            cl *= (1 - 0.005 * (age - 35))
        
        ke0 = self.ref_vals['ke0']
        
        k10 = cl / v1
        k12 = q2 / v1
        k21 = q2 / v2
        k13 = q3 / v1
        k31 = q3 / v3
        
        return {
            'v1': v1, 'v2': v2, 'v3': v3,
            'k10': k10, 'k12': k12, 'k13': k13,
            'k21': k21, 'k31': k31, 'ke0': ke0
        }

    def get_ce50(self, age):
        ce50 = 3.4 - 0.0183 * (age - 20)
        return max(ce50, 1.0) # Safety clamp

def simulate_pk_step(state, infusion_rate, params, dt=5.0/60.0):
    """
    Single step simulation for PK model.
    state: [A1, A2, A3, Ce]
    infusion_rate: drug input in mass per unit time (same units as params, usually mg/min or mcg/min)
    dt: time step in minutes (default 5 seconds = 5/60 min)
    """
    a1, a2, a3, ce = state
    k10, k12, k13 = params['k10'], params['k12'], params['k13']
    k21, k31, ke0 = params['k21'], params['k31'], params['ke0']
    v1 = params['v1']
    
    # Differential equations
    da1 = infusion_rate - (k10 + k12 + k13)*a1 + k21*a2 + k31*a3
    da2 = k12*a1 - k21*a2
    da3 = k13*a1 - k31*a3
    
    cp = a1 / v1
    dce = ke0 * (cp - ce)
    
    # Euler update
    a1_new = a1 + da1 * dt
    a2_new = a2 + da2 * dt
    a3_new = a3 + da3 * dt
    ce_new = ce + dce * dt
    
    return [a1_new, a2_new, a3_new, ce_new]

