import jax.numpy as jnp
from jax import jit
from jax import lax

import os

from scipy.interpolate import interp1d

from Utils.spherical_bessel_transform_ncol_jax import SphericalBesselTransform
from Utils.qfuncfft_jax import QFuncFFT_JAX as QFuncFFT
from Utils.loginterp_jax import loginterp_jax as loginterp

class Zenbu:
    '''
    Class to calculate power spectra up to one loop.
    
    Based on velocileptors
    
    https://github.com/sfschen/velocileptors/blob/master/LPT/cleft_fftw.py
    
    The bias parameters are ordered in pktable as
    1, b1, b1^2, b2, b1b2, b2^2, bs, b1bs, b2bs, bs^2.
    Note that these are the component spectra (b_i, b_j) and not the coefficient multiplying b_i b_j in the auto.
    
    Can combine into a full one-loop real-space power spectrum using the function combine_bias_terms_pk.
    
    '''

    def __init__(self, k, p, kmin=1e-3, kmax=3, nk=100, kvec=None, cutoff=10, jn=5, N = 2000, threads=None, extrap_min = -5, extrap_max = 3):

        self.N = N
        self.extrap_max = extrap_max
        self.extrap_min = extrap_min
        
        self.cutoff = cutoff
        self.kint =  jnp.logspace(extrap_min,extrap_max,self.N)
        self.qint =  jnp.logspace(-extrap_max,-extrap_min,self.N)
        
        self.update_power_spectrum(k,p)        
        self.pktable = None
        self.num_power_components = 4

        
        self.jn = jn
        self.sph = SphericalBesselTransform(self.qint, L=self.jn, ncol=self.num_power_components)        
        
        if kvec is None:
            self.kmin, self.kmax = kmin, kmax
            self.nk = 100
    
            self.kv = jnp.logspace( jnp.log10(self.kmin),  jnp.log10(self.kmax), self.nk)
            self.pktable =  jnp.zeros([self.nk, self.num_power_components+1])
            self.pktable = self.pktable.at[:,0].set(self.kv)
        else:
            self.kmin, self.kmax = kvec[0], kvec[-1]
            self.nk = len(kvec)
            
            self.kv = kvec
            self.pktable =  jnp.zeros([self.nk, self.num_power_components+1])
            self.pktable = self.pktable.at[:,0].set(self.kv)

    def update_power_spectrum(self, k, p):
        # Updates the power spectrum and various q functions. Can continually compute for new cosmologies without reloading FFTW
        self.k = k
        self.p = p
        self.pint = loginterp(k,p)(self.kint) *  jnp.exp(-(self.kint/self.cutoff)**2)
        self.setup_powerspectrum()

    def setup_powerspectrum(self):
                
        self.qf = QFuncFFT(self.kint, self.pint, qv=self.qint, oneloop=True, shear=True, third_order=True)
        
        # linear terms
        self.Xlin = self.qf.Xlin
        self.Ylin = self.qf.Ylin
        
        self.XYlin = self.Xlin + self.Ylin; self.sigma = self.XYlin[-1]
        self.yq = self.Ylin / self.qint
        
        self.Ulin = self.qf.Ulin
        self.corlin = self.qf.corlin
    
        self.Xs2 = self.qf.Xs2
        self.Ys2 = self.qf.Ys2; self.sigmas2 = (self.Xs2 + self.Ys2)[-1]
        self.V = self.qf.V
        self.zeta = self.qf.zeta
        self.chi = self.qf.chi
        self.Xs4 = self.qf.Xs4
        self.Ys4 = self.qf.Ys4
        
        self.Psi2graddelta = self.qf.xi_l_n(0,0,3./7*(self.qf.R1+self.qf.R2))
        self.theta = self.qf.theta

    def p_integrals(self, k, D):
        '''
        Compute P(k) for a single k as a vector of all bias contributions.
        
        '''
        ksq = k**2; kcu = k**3; k4 = k**4
        expon =  jnp.exp(-0.5*ksq * D**2 * (self.XYlin - self.sigma))
        exponm1 =  jnp.expm1(-0.5*ksq * D**2 * (self.XYlin - self.sigma))
        suppress =  jnp.exp(-0.5 * ksq * D**2 * self.sigma)
        
        ret =  jnp.zeros(self.num_power_components)
        
        bias_integrands =  jnp.zeros( (self.num_power_components,self.N)  )
        
        for l in range(self.jn):
            # l-dep functions
            mu1fac = (l>0)/(k * self.yq * D**2)
            mu2fac = 1. - 2.*l/ksq/self.Ylin / D**2
            #mu3fac = (1. - 2.*(l-1)/ksq/self.Ylin / D**2) * mu1fac # mu3 terms start at j1 so l -> l-1
            #mu4fac = 1 - 4*l/ksq/self.Ylin/D**2 + 4*l*(l-1)/(ksq*self.Ylin*D**2)**2
            
            bias_integrands = bias_integrands.at[0,:].set(D**2 * self.corlin - D**4 * ksq*mu2fac*self.Ulin**2) # (delta, delta)
            bias_integrands = bias_integrands.at[1,:].set(D**4 * self.theta) # (delta, b3)
            bias_integrands = bias_integrands.at[2,:].set(D**4 * self.Psi2graddelta) # (delta, k.Psi2 nabla delta)

            bias_integrands = bias_integrands.at[-1,:].set(1) # matter Zeldovich

            # Multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon
                bias_integrands = bias_integrands - bias_integrands[:, -1][:, None]  # Note that expon(q = infinity) = 1
            else:
                bias_integrands = bias_integrands * expon * (self.yq*D**2) ** l
                
            ktemps, bias_ffts = self.sph.sph(l, bias_integrands)
            
            ii = jnp.where( (ktemps - k) > 0, size=self.N)[0][0]
            kl, kr = ktemps[ii-1], ktemps[ii]
            bias_ffts_interp = (bias_ffts[:,ii-1] * (kr-k) + bias_ffts[:,ii] * (k - kl))/(kr-kl)
            ret += k**l * bias_ffts_interp
            
        return 4*suppress* jnp.pi*ret

    def make_ptable(self,D):

        def _ptable_loopfunc(ii, tab):
            return tab.at[ii,1:].set(self.p_integrals(self.kv[ii],D))
    
        self.pktable = lax.fori_loop(0,self.nk,_ptable_loopfunc, self.pktable)
    
        return self.pktable
