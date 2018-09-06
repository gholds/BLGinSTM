from . import * # Packages imported in __init__.py


# Define some universal constants
pi=np.pi

#Energy

eVtoJ=1.60218*(10**(-19)) # eV to Joules
JtoeV=1/eVtoJ

InvAngs=(10**(10)) # Converts rad/m to inverse angstroms

# Electron
q=eVtoJ # fundamental charge
m = 9.1094 * 10**-31 # kg, mass of the electron

# Universal
hbar=1.0545718*(10**(-34)) # Joule-s
kB = 1.38064852*10**-23 # J/K 
e0=8.85*(10**(-12)) # Permittivity of free space



class BaseGraphene:
    """
    Base class for all types of graphene. Includes constants common to all
    types of graphene. Can be used to develop classes with
    more layers.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        self.a = 1.42 * (10**(-10)) # (m), Interatom spacing
        self.Ac = 3*np.sqrt(3)*(self.a**2) / 2 # (m^2), Area of unit cell of graphene
        self.g0 = 2.8*eVtoJ # (J), Interatom hopping potential
        self.vF = 3*self.a*self.g0/(2*hbar) # Fermi velocity
        self.W = 4.6 * eVtoJ # Work function of graphene. See 10.1038/nphys1022


class Bilayer(BaseGraphene):
    """
    Bilayer graphene class which inherits the parameters of the
    BaseGraphene class.
    """

    g1  = 0.358 * eVtoJ # (J), A1-B1 hopping potential
    g3  = 0.3   * eVtoJ # (J), A1-B2 hopping potential
    g4  = 0.12  * eVtoJ # (J), A1-A2 hopping potential (McCann Koshino 2013)
    d   = 3*(10**-10)   # (m), interlayer spacing
    approx_choices = ['None', 'Common', 'LowEnergy']
    C = e0 / d

    this_dir = os.path.dirname(os.path.realpath(__file__))

    def Hamiltonian(self,k,u):
        '''
        Returns the full tight-binding Hamiltonian of BLG.
        For array-like inputs of k, the Hamiltonian of the
        ith value of k is Hamiltonian[:,:,i]

        Parameters
        ----------
        k:  array-like
            Wavenumber (1/m).

        u:  scalar
            Interlayer potential energy difference (J).

        Returns
        ----------
        H:  array-like
            Tight-binding Hamiltonian of bilayer graphene.
        '''

        k = np.atleast_1d(k)
        length = np.shape(k)[0]
        ones = np.ones(length)

        # Diagonals
        H11 = H22 = -u/2 * ones
        H33 = H44 =  u/2 * ones

        # Intralayer
        H12 = H21 = H34 = H43 = hbar * self.vF * k

        # Interlayer A1-B2
        H23 = H32 = self.g1 * ones

        # Trigonal Warping
        H14 = H41 = np.sqrt(3/4) * self.g3 * self.a * k

        # g4
        H13 = H31 = H42 = H24 = - (3/4)**(1/2) * self.a * self.g4 * k

        H = np.array([  [H11, H12, H13, H14],
                        [H21, H22, H23, H24],
                        [H31, H32, H33, H34],
                        [H41, H42, H43,H44]]).squeeze()
        return H

    ######################
    ### Band Structure ###
    ######################

    def Dispersion(self,k,u,band,approx='Common'):
        '''
        Returns the energy (J) of an electron with wavevector k (rad/m)
        in first (band=1) or second (band=2) conduction band.
        Only approximation is g3=0.
        To get valence bands, simply result multiply by -1.
        '''
        p = hbar * self.vF * k

        if approx == 'Common':
            radical=(self.g1**4)/4 + (u**2 + self.g1**2)*(p**2)
            return np.sqrt( (self.g1**2)/2 + (u**2)/4 + p**2 + ((-1)**(band))*np.sqrt(radical) )

        if approx == 'LowEnergy':
            '''
            Low Energy effective. Eigenvalues of 

            H = ( ( u/2, p^2 / 2m ) , ( p^2/2m, -u/2 ) )

            '''

            meff = ( self.g1 / (2 * (self.vF)**2) )
            return np.sqrt( (u/2)**2 + ( (hbar * k)**2 / (2*meff) )**2 )

        if approx == 'None':
            '''
            No approximation. Compute eigenvalues of Hamiltonian
            '''
            k = k.squeeze()
            u = u.squeeze()
            disp = np.empty(np.shape(k))

            for i, wn in enumerate(k):
                disp[i] = linalg.eigvalsh(self.Hamiltonian(wn,u))[1+band]

            return np.array(disp).squeeze()

    def kmin(self,u, band=1):
        '''
        Returns positive wavenumber at the minimum of the first band in 1/m.

        Parameters
        ----------
        u :     array-like
                Interlayer potential energy difference in units J.

        band:   First (second) conduction band 1 (2).
        '''
        k2 = ( u**2 / (2*hbar*self.vF)**2 ) * ( (2*self.g1**2 + u**2) /( self.g1**2 + u**2 ) )
        return np.sqrt(k2)

    def emin(self,u):
        '''
        Returns minimum of the first band in Joules.
        '''
        emin2 = (u/2)**2 * (self.g1**2 / ( self.g1**2 + u**2 ) )
        return np.sqrt(emin2)

    def DOS(self, e, u):
        '''
        Returns the density of states per unit area (1/m^2) as 
        a function of energy given the gap u
        '''
        e = np.atleast_1d(abs(e))
        
        # Define the multiplicative factor out front
        # Set to 0 is energy is below the minimum
        mult = (e>self.emin(u)) * ( e / (pi * hbar**2 * self.vF**2) )
        
        # Calculate the discriminant
        # Remember, we wil need to divide by it's sqrt
        # So set values disc<=0 to 1
        # We will multiply the result by zero for these energies anyway later on.
        disc = e**2 * (self.g1**2 + u**2) - self.g1**2 * u**2 / 4
        disc = (e>self.emin(u))*disc + (e<=self.emin(u))*1
        
        # Calculate quantities proportional to derivatives of k^2
        propdkp2 = 2 + (self.g1**2 + u**2)/np.sqrt(disc)
        propdkm2 = 2 - (self.g1**2 + u**2)/np.sqrt(disc)
        
        # If energy is above sombrero region, add the positive solution
        # If within, also subtract the negative solution
        propdos = (e>self.emin(u))*propdkp2 - (e<=abs(u/2))*propdkm2
        return (mult * propdos)

    def Pdiff(self,k,vminus,approx='Common'):
        '''Returns the probability difference between finding an ELECTRON on the TOP layer minus the BOTTOM layer.'''
        
        u = -2*q*(vminus+np.sign(vminus)*0.0000001)
        
        if approx=='Common':
            e = self.Dispersion(k,u,1)

            K = hbar*self.vF*(k+1)
            
            numerator = (e**2 - u**2/4)**2 + 4*K**2*e**2 - K**4
            denominator = (e**2 - u**2/4)**2 + K**2*u**2 - K**4
            
            return - ( u / (2*e) ) * ( numerator / denominator )

        if approx=='LowEnergy':
            meff = ( self.g1 / (2 * (self.vF)**2) )
            denominator_squared = ( ( (hbar*k)**2/meff )**2 + u**2 )
            
            return - u / np.sqrt(denominator_squared)

        if approx=='None':
            k = np.atleast_1d(k).squeeze()
            u = np.atleast_1d(u).squeeze()
            deltapsi = []
            # Eigenvectors of 
            for i,wn in enumerate(k):
                v = linalg.eigh( self.Hamiltonian(wn,u) )[1]

                psi = v[:,-2] # Second highest band (first conduction)

                deltapsi.append(psi[0]**2 + psi[1]**2 - psi[2]**2 - psi[3]**2)

            return np.array(deltapsi).squeeze()

    def kFermi(self,n,u,pm):
        '''
        Returns Fermi vector kF+ for pm=1 and kF- for pm=2 in units rad/m
        '''
            
        # Define the more complicated factors and terms
        numerator = (pi * hbar**2 *self.vF**2 * n)**2 + ( self.g1*u )**2
        denominator = self.g1**2 + u**2
        pmterm = 2*pi*hbar**2 * self.vF**2 * abs(n) # abs for fact that electrons and holes symmetric
        
        # Factor proportional to k**2
        propk2 = ( numerator / denominator ) + u**2 + (-1)**(pm-1) * pmterm
        
        # If the fermi level is above u/2, set kF- to zero
        # This says that the region of occupied states is now a disk
        if pm%2==0:
            propk2 = (propk2 >= 0) * propk2
            propk2 = (self.Dispersion(self.kFermi(n,u,1),u,1)<u/2) * propk2
        
        return np.sqrt( propk2 ) / (2*hbar*self.vF)

    def eFermi(self,n,u):
        '''
        Returns the Fermi level (Joules) given density n and interlayer potential energy difference u
        Positive n returns a positive Fermi level, meaning positive carrier densities are electrons by convention.
        '''
        
        numerator = (hbar**2 * self.vF**2 * n *pi)**2 + (self.g1 * u)**2
        denominator = 4 * (self.g1**2 + u**2)
        
        return np.sign(n) * np.sqrt( numerator / denominator )

    #########################
    ### Carrier Densities ###
    #########################

    def nplusT0(self,vplus,vminus,approx='Common'):
        """
        Analytically computes the electron density at zero temperature.
        Faster than Bilayer.nplus() since this function allows
        for vectorized operations.
        """

        # Convert voltages to energies
        eF = eVtoJ*vplus
        u  = -2*eVtoJ*vminus

        if approx == 'Common':
            # Calculate the radical
            radical = (self.g1**2+u**2) * eF**2 - self.g1**2 * u**2 / 4

            # For energies within the gap, radical is negative, so set it to 0 instead
            radical = (radical>=0)*radical

            # Proportional to the square of the Fermi wavevectors
            kFp2 = (eF**2 + u**2/4) + np.sqrt(radical)
            kFm2 = (eF**2 + u**2/4) - np.sqrt(radical)

            # For kFm2, if eF > u/2, set to zero
            kFm2 = (abs(eF) <= abs(u/2)) * kFm2
            
            # Calculate the proportionality factor
            # Includes:
            #     1/(hbar vF)**2 from formula for kF
            #     1/pi from n = (kFp2 - kFm2)/pi
            #     Sets to zero if Fermi in the gap
            prop = (abs(eF)>self.emin(u))*np.sign(eF)*(1 / (hbar**2 * self.vF**2 * pi))

            return prop * (kFp2 - kFm2)


        if approx == 'LowEnergy':
            """
            See Young and Levitov 2011.
            """
            meff = ( self.g1 / (2 * (self.vF)**2) )

            nu0 = 2 * meff * q  / (pi * hbar**2)

            energy_diff = (np.abs(eF)>np.abs(u/2)) * (eF**2 - (u/2)**2)
            return (nu0/q) * np.sign(eF) * np.sqrt(energy_diff)

    def nminusT0(self,vplus,vminus):

        meff = ( self.g1 / (2 * (self.vF)**2) )
        nu0 = 2 * meff * q  / (pi * hbar**2)

        prop = nu0 * vminus
        
        # Find cutoff energy. Approximate it as the vminus=0 case
        Lambda = self.Dispersion( 1 / (np.sqrt(3) * self.a), -2*q*vminus, 1 ) / q
        
        # Compute the denominator of the log
        metal = abs(vplus) >= abs(vminus)
        den = (metal) * np.abs(vplus) + np.sqrt( metal * vplus**2 + (-1)**metal * vminus**2 ) 
        
        return prop * np.log(2 * Lambda / den)


    #################
    ### Screening ###
    #################

    def screened_vminus(self,nplus,vminus):
        """
        The screened value of vminus given the total charge nplus
        """
        vminus = np.atleast_1d(vminus)
        a = -1
        b = 1

        vminus_screened = []

        for vm in vminus:

            vp = self.eFermi(nplus, -2*q*vm) / q

            def f1(vm1):
                return (vm1 - vm) + (q / (4*self.C))*self.nminus(vp,vm1,0)

            vm1 = optimize.brentq(f1,a,b)
            vminus_screened.append(vm1)

        return np.array(vminus_screened).squeeze()

    def screened_vminus2(self,nplus,vminus):
        """
        The screened value of vminus given the total charge nplus
        """
        a, b = -1, 1

        vminus_screened = []

        for vm in vminus:
            vm0 = vm
            vp0 = self.eFermi(nplus, -2*q*vm) / q

            def f1(vm1):
                return (vm1 - vm) + (q / (4*self.C))*self.nminus(vp0,vm1,0)

            vm1 = optimize.brentq(f1,a,b)
            vp1 = self.eFermi(nplus, -2*q*vm1) / q

            while (vm1-vm0)**2 + (vp1-vp0)**2 > 0.0001:
                vp0, vm0 = vp1, vm1

                def f1(vm1):
                    return (vm1 - vm) + (q / (4*self.C))*self.nminus(vp0,vm1,0)

                vm1 = optimize.brentq(f1,a,b)
                vp1 = self.eFermi(nplus, -2*q*vm1) / q
            
            vminus_screened.append(vm1)

        return np.array(vminus_screened)

    def screened_newton(self,vplus,vminus):
        n = self.nplus(vplus,vminus,0)

        def f1(v):
            return (v[1] - vminus) + (q / (4*self.C))*self.nminusT0(v[0],v[1])

        def f2(v):
            return n - self.nplus(v[0],v[1],0)

        v = Newton.Newton2D(f1,f2,np.array([vplus,vminus]))

        return v


    ##################
    ### OLD METHOD ###
    ##################

    def nplus(self,vplus,vminus, T, approx='Common',points = 10000):
        '''
        Returns the electron carrier density for various electrostatic potentials vplus, vminus.
        Convention is that electrons have positive carrier density while holes have negative.
        '''

        # Treat inputs as ndarrays so we can take advantage of broadcasting
        vplus = np.atleast_1d(vplus)
        vminus = np.atleast_1d(vminus)

        vplus = vplus.reshape(1,1,len(vplus))
        vminus = vminus.reshape(1,len(vminus),1)

        # Domain over first Brillouin zone
        ks = np.linspace(0,1/(np.sqrt(3)*self.a), num=points).reshape((points,1,1))

        # Calculate the kinetic energy
        KE = self.Dispersion(ks, -2*q*vminus,1,approx)

        # Evaluate Fermi-Dirac
        FD = (FermiDirac(KE-q*vplus,T)-FermiDirac(KE+q*vplus,T))

        # Define integrand
        integrand = ( 2 / np.pi ) * ks * FD

        return np.squeeze(integrate.trapz(integrand,ks,axis=0))

    def nminus(self,vplus,vminus, T, approx='Common', points=10000):
        '''
        Returns the electron carrier density for various electrostatic potentials vplus.
        Convention is that electrons have positive carrier density while holes have negative.
        '''

        if approx == 'None':
            # print('Not yet supported')
            return
        # Treat inputs as ndarrays so we can take advantage of broadcasting
        vplus = np.atleast_1d(vplus)
        vminus = np.atleast_1d(vminus)

        vplus = vplus.reshape(1,1,len(vplus))
        vminus = vminus.reshape(1,len(vminus),1)

        # Domain over first Brillouin zone
        ks = np.linspace(0,1/(np.sqrt(3)*self.a), num=points).reshape((points,1,1))

        # Calculate the kinetic energy
        KE = self.Dispersion(ks, -2*q*vminus,1, approx)

        # Evaluate Fermi-Dirac
        # Minus sign comes from...
        FD = (FermiDirac(KE-q*abs(vplus),T))#-Temperature.FermiDirac(-KE-q*vplus,T)

        # Define integrand
        integrand =  ( 2 /np.pi ) * ks * self.Pdiff(ks,vminus,approx='LowEnergy') * FD

        return np.squeeze(integrate.trapz(integrand,ks,axis=0))

    def generate_nplus_nminus(self,vplus,vminus,T):
        """
        Generates and saves high-resolution surfaces of nplus(vplus,vminus)
        and nminus(vplus,vminus). Only generate for first quadrant (vplus,vminus > 0)
        since surfaces have symmetry properties.
        """
        save_dir = os.path.join(self.this_dir,
                                'CarrierDensities',
                                'Temp_{:.2E}'.format(T))

        if os.path.exists(save_dir):
            # print('Carrier densities for T = {} K have already been generated'.format(T))
            return
        #else:
        #    os.makedirs(save_dir)

        if np.any(vplus< 0) or np.any(vminus<0):
            # print('Some voltages were negative in the ranges\n')
            # print(  vplus[0].squeeze(),
            #        ' < vplus < ',
            #        vplus[-1].squeeze(),
            #        ' {} points'.format(np.shape(vplus)[0]))
            # print(  vminus[0].squeeze(),
            #        ' < vminus < ', vminus[-1].squeeze(),
            #        ' {} points'.format(np.shape(vminus)[0]))

            vplus   = np.linspace(0,vplus[-1],num=np.shape(vplus)[0])
            vminus  = np.linspace(0,vminus[-1],num=np.shape(vminus)[0]).reshape(np.shape(vminus))

            # print('\nInstead, generating over the ranges\n')
            # print(  '0 < vplus < ', vplus[-1].squeeze(),
            #         ' {} points'.format(np.shape(vplus)[0]))
            # print(  '0 < vminus< ', vminus[-1].squeeze(),
            #        ' {} points'.format(np.shape(vminus)[0]))
            # print()

        # Choose the size of the batches we will generate
        d = 10

        # Check that it is compatible with the lengths of vplus and vminus
        if len(vplus) % d != 0 or len(vminus) % d != 0:
            # print('Batch size (d) incompatible with voltage arrays')
            # print('d= {} does not evenly divide either len(vplus)= {} or len(vminus) = {}'.format(d,len(vplus),len(vminus)))
            return

        nplus_surface = np.empty(np.shape(vminus*vplus))
        nminus_surface = np.empty(np.shape(vminus*vplus))

        for i in range(int(len(vplus)/d)):
            i_frac = i / int(len(vplus)/d)
            for j in range(int(len(vminus)/d)):
                j_frac  = (j / int(len(vminus)/d)) * (1 / int(len(vplus)/d))
                percentage = round(100* (i_frac + j_frac),2)
                # print('{} % Finished'.format(percentage))
                nplus_surface[d*j:d*j+d,d*i:d*i+d]=self.nplus(vplus[d*i:d*i+d],vminus[d*j:d*j+d,:],T)
                nminus_surface[d*j:d*j+d,d*i:d*i+d]=self.nminus(vplus[d*i:d*i+d],vminus[d*j:d*j+d,:],T)

        # Save the surfaces
        np.save(save_dir+'nplus_surface.npy',nplus_surface)
        np.save(save_dir+'nminus_surface.npy',nminus_surface)

        # Save the voltages
        np.save(save_dir+'vplus.npy', vplus)
        np.save(save_dir+'vminus.npy', vminus)

    def get_vplus(self,T):
        """
        Returns the vplus array saved in ...
        Doubles the range to negative values
        """
        save_dir = os.path.join(self.this_dir,
                                'CarrierDensities',
                                'Temp_{:.2E}'.format(T))
        vplus = np.load(save_dir+'vplus.npy')
        return np.concatenate((-vplus[:0:-1],vplus))

    def get_vminus(self,T):
        save_dir = os.path.join(self.this_dir,
                                'CarrierDensities',
                                'Temp_{:.2E}'.format(T))
        vminus = np.load(save_dir+'vminus.npy')
        return np.concatenate((-vminus[:0:-1],vminus))

    def get_nplus(self,T):
        save_dir = os.path.join(self.this_dir,
                                'CarrierDensities',
                                'Temp_{:.2E}'.format(T))
        nplus_surface = np.load(save_dir+'nplus_surface.npy')
        nplus_surface = np.concatenate((nplus_surface[:0:-1,:],nplus_surface))
        nplus_surface = np.concatenate((-nplus_surface[:,:0:-1],nplus_surface),axis = 1)
        return nplus_surface

    def get_nminus(self,T):
        save_dir = os.path.join(self.this_dir,
                                'CarrierDensities',
                                'Temp_{:.2E}'.format(T))
        nminus_surface = np.load(save_dir+'nminus_surface.npy')
        nminus_surface = np.concatenate((-nminus_surface[:0:-1,:],nminus_surface))
        nminus_surface = np.concatenate((nminus_surface[:,:0:-1],nminus_surface),axis = 1)
        return nminus_surface


def FermiDirac(en, T):
    """
    The Fermi-Dirac distribution.

    Parameters
    ----------
    en:         array-like
                Energy of state in units J

    T:          scalar
                Temperature in units K

    Returns
    ----------
    FD:         array-like
                Fermi-Dirac probability of occupation of state at energy en.

    """

    # Using logaddexp reduces chance of underflow error
    # Adds a tiny offset to temperature to avoid division by zero.
    FD = np.exp( -np.logaddexp(en/(kB*(T+0.000000000001)),0) )

    return FD


class BLGinSTM:

    """
    A Tunneling experiment containing information to calculate tunnel
    current from BLG. Is also capable of plotting the results.

    Parameters
    ----------

    d1, d2  :   The top gate-sample and bottom gate-sample distances
                respectively in nanometers. Converted to meters once initialized.
    
    e1, e2  :   The relative permittivities in the top gate-sample and
                bottom gate-sample regions respectively. Multiplied by e0
                once initialized.

    T       :   Experiment Temperature in Kelvin

    Wtip    :   Work function of top gate in eV. Converted to J for calculations

    material:   A material chosen from the Materials package. So far, only bilayer
                graphene is available.

    screening:  Boolean; "False" (default) to neglect screening between BLG layers and "True"
                to include the effects. Calculations take much longer if "True".

    """

    def __init__(self, d1, d2, e1, e2, T, Wtip, screening=False):
        self.d1 = d1 * 10**-9
        self.d2 = d2 * 10**-9
        self.e1 = e1 * e0
        self.e2 = e2 * e0
        self.T  = T
        self.Wtip = Wtip * eVtoJ

        self.C1 = self.e1 / self.d1
        self.C2 = self.e2 / self.d2

        self.BLG = Bilayer()

        self.phibar = (self.Wtip + self.BLG.W) / 2  # Average work function
        self.dW     = (self.Wtip - self.BLG.W)      # Difference between work functions
        self.dWe    = self.dW / (-q)                # Potential diff due to work function diff

        self.I = None

        if screening == False:
            self.screen_func = lambda x, y: y

        if screening == True:
            self.screen_func = self.BLG.screened_vminus

    def vplus_n0(self,VT,VB):
        """
        Potential of the BLG layer when no charge has accumulated

        Parameters
        ----------

        VT:     scalar or array-like, tip voltage

        VB:     scalar or array-like, gate voltage

        """
        num = self.e1*self.d2*(VT-self.dWe) + self.e2*self.d1*VB
        den = self.d1*self.e2 + self.d2*self.e1
        return num / den

    def vminus_n0(self,VT,VB):
        """
        Potential difference between the layers when no charge has accumulated.

        Parameters
        ----------

        VT:     scalar or array-like; tip voltage

        VB:     scalar or array-like; backgate voltage
        """
        num = self.BLG.d * (self.e1 + self.e2)
        den = 4*(self.e2*self.d1 + self.e1*self.d2)
        return (num/den)*(VT-VB-self.dWe)

    def n_exists(self,VT,VB):
        """
        Boolean function that returns whether or not charge has accumulated

        Parameters
        ---------

        VT:     scalar or array-like; tip voltage

        VB:     scalar or array-like; backgate voltage
        """

        vplus = self.vplus_n0(VT,VB)

        u = -2*q*self.vminus_n0(VT,VB)
        minimum = self.BLG.emin(u) / (q)

        return abs(vplus) >= abs(minimum)

    def nElectron(self,vplus,VT,VB):
        '''
        Returns electron density (m^-2) as a function of
        electrode-sample potential differences.

        Parameters
        ----------

        vplus:      scalar or array-like; potential of the BLG

        VT:         scalar or array-like; tip voltage

        VB:         scalar or array-like; backgate voltage

        '''
        s1 = (VT-vplus-self.dWe) * self.C1
        s2 = (VB-vplus) * self.C2
        
        return (s1 + s2) / q

    def vminus_n1(self,vplus,VT,VB):
        """
        The voltage difference when charge has accumulated.
        """
        nplus = self.nElectron(vplus, VT, VB)
        vm_unscreened = (self.BLG.d / 4) * ( (VT-vplus-self.dWe)/self.d1 - (VB-vplus)/self.d2 )
        return self.screen_func(nplus,vm_unscreened)

    def vplus_root(self,vplus,VT,VB):
        """
        Self-consistency equation for the fermi level
        """
        u = -2*q*self.vminus_n1(vplus,VT,VB)
        n = self.nElectron(vplus,VT,VB)

        term1 = 4 * (q*vplus *self.BLG.g1)**2
        term2 = ( 4*(q*vplus)**2 - self.BLG.g1**2 )* u**2
        term3 = - (hbar**2 * self.BLG.vF**2 * pi)**2 * n**2

        return term1 + term2 + term3

    def vplus_n1(self,VT,VB):
        """
        Returns the Fermi level when charge has accumulated.
        Does so by finding a root to a self-consistent equation.
        """

        # otherwise we need to find the self-consistent fermi level
        f = lambda x: self.vplus_root(x,VT,VB)

        vplus = self.vplus_n0(VT,VB)

        a = min(0,vplus)
        b = max(0,vplus)

        return optimize.brentq(f,a,b)

    def v_eq(self,VT,VB):
        """
        Returns the Fermi Level first checking for charge. Uses the appropriate vplus formula.
        """

        if not self.n_exists(VT,VB):
            vp = self.vplus_n0(VT,VB)
            return (vp, self.vminus_n0(VT,VB))

        else:
            vp = self.vplus_n1(VT,VB)
            return (vp,self.vminus_n1(vp,VT,VB))

    def generate_vplus_vminus(self,VT,VB,method):
        '''Finds equilibrium values of V+ and V- for a grid of
        tip voltages VT and gate voltages VG.

        Parameters
        ----------
        VT:     array-like, voltages over which to sweep the tip.

        VB:     array-like, voltages over which to sweep the backgate.
        
        method: Method used to compute equilibrium voltages. Only 'DasSarma' works.
        '''

        if method == 'DasSarma':

            num_vts = np.shape(VT)[0]
            num_vbs = np.shape(VB)[0]

            VT = VT.reshape((num_vts,1))
            VB = VB.reshape((1,num_vbs))

            vp = np.empty(np.shape(VT*VB))
            vm = np.empty_like(vp)

            for i in range(num_vts):
                for j in range(num_vbs):
                    if not self.n_exists(VT[i,0],VB[0,j]):
                        vp[i,j] = self.vplus_n0(VT[i,0],VB[0,j])
                        vm[i,j] = self.vminus_n0(VT[i,0],VB[0,j])
                    else:
                        vp[i,j] = self.vplus_n1(VT[i,0],VB[0,j])
                        vm[i,j] = self.vminus_n1(vp[i,j],VT[i,0],VB[0,j])

            return (vp, vm)

        if method == 'YoungLevitov':
            # Load values of vplus and vminus to search over

            # print('YoungLevitov method is not supported')
            return

            # print('Loading Carrier Densities')
            vplus = self.BLG.get_vplus(self.T)
            vplus = vplus.reshape(1,len(vplus),1,1).astype('float32')
            vminus = self.BLG.get_vminus(self.T)
            vminus = vminus.reshape(len(vminus),1,1,1).astype('float32')

            # b refers to 'batch size'
            # We loop over the range of tip and gate voltages
            # and broadcast in batches to avoid memory errors
            b = 5

            # load the carrier densities from their files
            nplus_array = self.BLG.get_nplus(self.T).astype('float32')
            nminus_array = self.BLG.get_nminus(self.T).astype('float32')

            nplus_array = nplus_array[:,:,np.newaxis,np.newaxis]
            nminus_array = nminus_array[:,:,np.newaxis,np.newaxis]

            # Choose tip and backgate voltages
            num_vts = int(100 * num_vts_100) # number of points for VT
            num_vbs = int(100 * num_vbs_100) # number of points for VB

            # Create array of Tip and Backgate voltages
            VT = np.linspace(VTrange[0],VTrange[1],num=num_vts)
            VT = VT.reshape(1,1,num_vts,1).astype('float32')
            VB = np.linspace(VBrange[0],VBrange[1],num=num_vbs)
            VB = VB.reshape(1,1,1,num_vbs).astype('float32')

            # Arrays where we will load the solutions
            vplus0 = np.zeros((num_vts,num_vbs))
            vminus0 = np.zeros((num_vts,num_vbs))

            # print('Computing equilibrium voltages')
            for i in range(int(num_vts/b)):
                i_frac = i / int(num_vts/b)
                
                for j in range(int(num_vbs/b)):
                    j_frac = ( j / int(num_vbs/b) ) * (1/int(num_vts/b))
                    # print('{} % finished'.format( round(100*(i_frac+j_frac)) ), end='\r' )
                    VGplus = (1 / 2) * (VT[:,:,b*i:b*i+b,:]+VB[:,:,:,b*j:b*j+b])
                    VGminus = (1 / 2) * (VT[:,:,b*i:b*i+b,:]-VB[:,:,:,b*j:b*j+b])

                    # Generate the intersecting planes for each pair of voltages
                    plane_minus_array = self.planeminus(vplus, vminus,VGplus,VGminus)
                    plane_plus_array  = self.planeplus(vplus,vminus,VGplus,VGminus)

                    # Generate the electrostatic equations
                    f1 = plane_plus_array - (-q)*nplus_array
                    f2 = plane_minus_array - (-q)*nminus_array

                    # Find the magnitudes
                    F = f1**2 + f2**2

                    # Minimum for each pair of voltages
                    Fmins = np.min(F,axis=(0,1))

                    # Array where we will put the indices of the voltages
                    Fmins_args= np.empty(np.shape(Fmins),dtype=(int,2))

                    # Find the indices of V+ and V-
                    for k in range(b):
                        for l in range(b):
                            Fmins_args[k,l] = np.where(F[:,:,k,l]==Fmins[k,l])

                    # Record the values of V+ and V- based on the indices
                    vplus0[b*i:b*i+b,b*j:b*j+b] = vplus[:,Fmins_args[:,:,1].flatten('C')].squeeze().reshape(b,b)
                    vminus0[b*i:b*i+b,b*j:b*j+b] = vminus[Fmins_args[:,:,0].flatten('C')].squeeze().reshape(b,b)

            # print('100 % Finished')
            return (vplus0, vminus0)

    def fermidirac(self, x, vt):
        return FermiDirac(x-q*vt,0) - FermiDirac(x,0)

    def tunnelcurrent(self,vplus,vminus,VT,T):
        '''Returns tunnel current.'''

        eF = -q*vplus
        u = -2*q*vminus

        # Estimate prefactor C0
        C0 = (4*pi*q / hbar) * 1 * self.BLG.Ac *1

        kappa0 = np.sqrt(2*m*self.phibar)/hbar

        fermidirac = lambda x : FermiDirac(x-q*VT,T) - FermiDirac(x,T)
        integrand = lambda x : fermidirac(x) * self.BLG.DOS(eF+x,u) * np.exp((x)*kappa0*self.d1/(2*self.phibar))

        # Points which are divergences or discontinuities or the bounds
        # At one point the end points were changing the result, but no longer
        bounds = np.sort( np.array([u/2, -u/2, self.BLG.emin(u), -self.BLG.emin(u),eF,eF+q*VT]) ) 
        bounds = bounds - eF

        tc = np.empty(len(bounds)-1)

        for i in range(len(tc)):
            tc[i] = integrate.quad(integrand,bounds[i],bounds[i+1])[0]

        return tc.sum()

    def generate_tunnelcurrent(self,VT,VB,method='DasSarma',return_current=False):
        '''
        Computed tunnel current over range VT by VB.

        Parameters
        ----------
        VT:     array-like, tip voltages

        VB:     array-like, backgate voltages

        method: Method used to compute the equilibrium voltages.
                Only 'DasSarma' is valid

        return_current: Boolean, whether to return the current or simply
                        save to the object.
        
        '''

        # Get equilibrium values of voltages
        vplus0, vminus0 = self.generate_vplus_vminus(VT,VB,method)


        tc = np.empty(np.shape(vplus0))

        #print('Computing tunnel currents')
        for i in range(np.shape(tc)[0]):
            # print('{} % finished'.format( 100* i / int(np.shape(tc)[0])),end='\r')
            for j in range(np.shape(tc)[1]):
                tc[i,j] = self.tunnelcurrent(vplus0[i,j],vminus0[i,j],VT[i],0)


        if return_current==True:
            return tc

        # if return_current is not chosen, then voltages and currents
        # will be saved.
        self.VT = VT
        self.VB = VB
        self.I = tc

    def plot_dIdV(self,show=True,save=False,norm=False):
        """
        Plots a color map of dI/dV spectrum (VT vs VB).

        Parameters
        ----------
        show:       Boolean; Select "True" to show the image. Select "False" (default)
                    to simply generate it. Useful if you would just like to save the image.

        save:       Boolean; Select "True" to save to (file location)

        norm:       Boolean; Select "True" to normalize and plot (dI/dV)/(I/VT) spectrum.
        """

        dIdV = np.gradient(self.I,axis=0) # dI/dV

        if norm == True:
            IV = self.I / self.VT[:,np.newaxis] # I/V
            dIdV = dIdV / IV

        fig, ax = plt.subplots(figsize=(7,6))

        dIdV_plot = plt.imshow(dIdV,cmap=cm.RdYlGn,origin='lower',
                                aspect='auto',extent=(self.VB[0],self.VB[-1],1e3*self.VT[0],1e3*self.VT[-1]))
        fig.suptitle('$dI/dV$, Tip Height ={} nm'.format(self.d1*10**9))
        cbar = fig.colorbar(dIdV_plot,label='$dI/dV$ (S)')
        ax.set_xlabel('Gate Voltage (V)')
        ax.set_ylabel('Tip Voltage (mV)')
        if show == True:
            plt.show()

        if save == True:
            import os
            save_dir = os.path.join( os.path.dirname(__file__),
                                    'dIdV_Plots')
            fig.savefig(os.path.join(save_dir,'tip_height_{}ang.png'.format(round(self.d1*10**10))))

    def plot_dIdV_waterfall(self,VT,VB):
        """
        Plots a waterfall plot of dIdV spectrum

        Parameters
        ----------
        
        VT:     array-like, array of tip voltages to plot

        VB:     array-like, array of gate voltages to plot
        """

        fig, ax = plt.subplots(figsize=(6,10))

        fig.suptitle('dI/dV')

        current = self.generate_tunnelcurrent(VT,VB,return_current=True)

        dIdV = np.gradient(current, axis=0)

        num_points = np.shape(dIdV)[0]
        num_plots = np.shape(dIdV)[1]

        offsets = np.linspace(0,num_plots-1,num=num_plots).reshape(1,num_plots)*3*10**16

        ax.plot(VT,dIdV+offsets,color='b')

        ax.text(VT[-int(num_points/4)],(dIdV+offsets)[0,-1],"$V_B={}$ V".format(VB[-1]))
        ax.text(VT[-int(num_points/4)],(dIdV+offsets)[0,0],"$V_B={}$ V".format(VB[0]))


        plt.show()
   
    def planeplus(self,vplus, vminus,VGplus,VGminus):
        return (self.C1+self.C2)*(vplus - VGplus) + (self.C1-self.C2)*(vminus - VGminus)

    def planeminus(self,vplus,vminus,VGplus,VGminus):
        return (self.C1+self.C2)*(vminus - VGminus) + (self.C1 - self.C2)*(vplus - VGplus) - 4*self.BLG.C*vminus
