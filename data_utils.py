import numpy as np

try:
    from bbhx.waveformbuild import BBHWaveformFD
    from bbhx.utils.constants import *
except:
    print("\n > Did not find the BBHX module, any attempt at computing the waveform will fail!\n")
try:
    from gbgpu.gbgpu import GBGPU
    from gbgpu.utils.constants import *
    from bbhx.utils.constants import *
except:
    print("\n > Did not find the GBGPU module, any attempt at computing the waveform will fail!\n")

from eryn.utils import TransformContainer
  
try:
    import cupy as xp
    gpu_available = True
except ModuleNotFoundError:
    import numpy as xp
    gpu_available = False
    
class ucb:
    """
    Generating the waveform of a Ultra Compact Binary source and projecting it onto the LISA arms.
    Using the GBGPU package developed by M. Katz that can be found here: https://github.com/mikekatz04/GBGPU
    
    Needs the frequency array, the set of modes, the gpu flag, the start time of the measurement,
    the end time of the measurement, and the reference frequency.
    
    """
    def __init__(self, f_lims=None, f_inds=None, gpu=False, tobs=1.0, dt=1.0, use_tdi2=False):
        
        self._template = GBGPU(use_gpu=gpu_available)
        self._inds = f_inds
        self._tdi2 = use_tdi2
        
        if self._inds is None or f_lims is None:
            raise TypeError("Error: Please define the frequency limits and the indices to be used for the waveform computation.")
        
        self._waveform_kwargs = dict(N=f_inds[1]-f_inds[0], 
                                     dt=dt, 
                                     T=tobs,
                                     tdi2=self._tdi2)
        
        trans = {
            0: (lambda x: 10**(x)),
            1: (lambda x: x * 1e-3),
            2: (lambda x: 10**(x)),
            5: (lambda x: np.arccos(x)),
            8: (lambda x: np.arcsin(x)),}

        self._ndim_full = 9
        fill_dict = {"fill_inds": np.array([3]), "ndim_full": self._ndim_full, "fill_values": np.array([0.0])}
        self._transform_fn = TransformContainer(
            parameter_transforms=trans, fill_dict=fill_dict)

    def eval(self, params):
        """ Evaluating the model at a given parameter set.
        Args:
            params (ndarray): The parameter values to evaluate the likelihood on.
            ``params.shape=(num_params,)`` if 1D or
            ``params.shape=(num_params, num_binaries)`` if 2D for more than one binary.
            
        Returns:
            ndarray: A, E, T channels of the waveform. 
        """

        params = np.insert(params, 3, 0.0, axis=1) # Fix the second frequency derivative at zero.
        injct = self._transform_fn.transform_base_parameters(params, 
                        return_transpose=False).reshape(-1, self._ndim_full).T
        self._template.run_wave(*injct, **self._waveform_kwargs,)
        A_temp, E_temp = self._template.A, self._template.E
        
        return np.atleast_2d(A_temp), np.atleast_2d(E_temp), np.atleast_2d(injct)
    
# SMBHB Template class
class smbhb:
    """
    Generating the waveform of a SMBHB source and projecting it onto the LISA arms.
    Using the BBHx package developed by M. Katz that can be found here: https://github.com/mikekatz04/BBHx
    
    Needs the frequency array, the set of modes, the gpu flag, the start time of the measurement,
    the end time of the measurement, and the reference frequency.
    
    """
    def __init__(self, f=None, modes=[(2,2)], gpu=False, tstart=None, tend=0.0, fRef=0.0):
        
        self._template = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=True), use_gpu=gpu_available)
        self._f = xp.array(f)
        self._modes = modes
        self._fRef = fRef
        self._tstart = tstart
        self._tend = tend
        
    def get_m1_m2_from_chirp_and_eta(self, Mc, q):
        """Utility function to return m1 and m2 from Mchirp and q """
        m1 = 0.5*Mc*((q/((q+1.)**2))**(-3.0/5.0) + ((q/((q+1.)**2))**(-6.0/5.0) - 4.0*(q/((q+1.)**2))**(-1.0/5.0))**(0.5))
        m2 = 0.5*Mc*((q/((q+1.)**2))**(-3.0/5.0) - ((q/((q+1.)**2))**(-6.0/5.0) - 4.0*(q/((q+1.)**2))**(-1.0/5.0))**(0.5))
        return m1, m2   
        
    def eval(self, params):
        """ Evaluating the model at a given parameter set.
        Args:
            params (ndarray): The parameter values to evaluate the likelihood on.
            ``params.shape=(num_params,)`` if 1D or
            ``params.shape=(num_params, num_binaries)`` if 2D for more than one binary.
            
        Returns:
            ndarray: A, E, T channels of the waveform. 
        """
        Mc, q, chi1, chi2, logdist, tRef, ci, sb, lam, psi, phi0 = params[:, 0], params[:, 1], params[:, 2], \
                                                                   params[:, 3], params[:, 4], params[:, 5], \
                                                                   params[:, 6], params[:, 7], params[:, 8], \
                                                                   params[:, 9], params[:, 10]
        
        m1, m2 = self.get_m1_m2_from_chirp_and_eta(Mc, q)

        input_params = [
            m1,                 # Redshifted mass of body 1 (solar masses)
            m2,                 # Redshifted mass of body 2 (solar masses)
            chi1,               # Dimensionless spin of body 1 (in [-1, 1])
            chi2,               # Dimensionless spin of body 2 (in [-1, 1])
            (10**logdist) * 1e6 * PC_SI, # Luminosity distance (Mpc)
            phi0,               # Observer's azimuthal phase (rad)
            self._fRef*np.ones(m1.shape), # f_ref -> fmax = max(f^2A(f))
            np.arccos(ci),      # Inclination angle (rad)
            lam,                # Source longitude (rad)
            np.arcsin(sb),      # Source latitude (rad)
            psi,                # Polarization angle (rad)
            tRef * 3600         # Time shift (s), input is in (hrs)  
        ] 

        wf = self._template(
            *input_params, 
            freqs=self._f, 
            modes=self._modes,
            direct=False, 
            combine=False,
            length=1024, 
            squeeze=False,
            fill=True, 
            t_obs_start=self._tstart,
            t_obs_end=self._tend)

        return wf[:,0,:], wf[:,1,:], wf[:,2,:]

# Noise model definition
clight = 299792458.0
lisaLT = 2.5e9/clight

class noise_model:
    """
      Class for the nosie model. We only need the f-vector to evaluate it
    """
    def __init__(self, fvec):
        self._f = xp.array(fvec) # Ensure we get cuda array
        self.df = self._f[1] - self._f[0]
        
    def eval(self, params):
        """LISA noise model for AE channels. 
        The parameters are the log-base-10 of the acceleration and displacement noises.
        """
        params = xp.array(params) # <---- Ensure it's a numpy or cupy array
        if params.ndim == 1:
            Sa, Si = params[:,None]
        else:
            Sa, Si = params[:,0], params[:,1]
        x   = 2.0 * xp.pi * lisaLT * self._f
        Spm = 10**(Sa[:,None]) * (1.0 + (0.4e-3/self._f)**2) * (1.0+(self._f/8e-3)**4) * (2.0*xp.pi*self._f)**(-4.) * (2.0*xp.pi*self._f/clight)**2 
        Sop = 10**(Si[:,None]) * (1.0 + (2.e-3/self._f)**4) * (2.0 * xp.pi * self._f/clight)**2
        S0  = 8.0 * xp.sin(x)**2 * (2.0 * Spm * (3.0 + 2.0*xp.cos(x) + xp.cos(2.*x)) + Sop * (2.0 + xp.cos(x)))
        return xp.array(S0)
    def __call__(self, p):
        return self.eval(p)
    def noise_gen(self, params):
        """Generate noise for the AE channels. 
        The parameters are the log-base-10 of the acceleration and displacement noises.
        It returns a dictionary with the noise for the A and E channels.
        """
        Sn = self.eval(params)
        # Independent frequency-domain noise in the A and E channels.
        n_real = xp.random.normal(scale=xp.sqrt(Sn) / (2.0 * xp.sqrt(self.df)))
        n_imag = xp.random.normal(scale=xp.sqrt(Sn) / (2.0 * xp.sqrt(self.df))) * 1j
        data_A = xp.array(n_real) + xp.array(n_imag)

        n_real = xp.random.normal(scale=xp.sqrt(Sn) / (2.0 * xp.sqrt(self.df)))
        n_imag = xp.random.normal(scale=xp.sqrt(Sn) / (2.0 * xp.sqrt(self.df))) * 1j
        data_E = xp.array(n_real) + xp.array(n_imag)
        return data_A, data_E