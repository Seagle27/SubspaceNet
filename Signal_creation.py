import numpy as np
from matplotlib import pyplot as plt
from system_model import *

def create_DOA_with_gap(M:int, gap:float) :        
    """Create sources with minimal gap value for simulations
    Args:
        M (int): number of sources
        gap (float): minimal gap value

    Returns:
        np.ndarray: Doa array
    """
    while(True):
        DOA = np.round(np.random.rand(M) *  180 ,decimals = 2) - 90
        DOA.sort()
        difference_between_angles = np.array([np.abs(DOA[i+1] - DOA[i]) for i in range(M-1)])
        if(np.sum(difference_between_angles > gap) == M - 1 and np.sum(difference_between_angles < (180 - gap)) == M - 1):
            break
    # print(DOA)
    return DOA

def create_doa_permutations(M:int, gap:float) :        
    """Create all available doa combinations out of sources with minimal gap value for simulations
    Args:
        M (int): number of sources
        gap (float): minimal gap value

    Returns:
        np.ndarray: Doa array
    """
    while(True):
        DOA = np.round(np.random.rand(M) *  180 ,decimals = 2) - 90
        DOA.sort()
        difference_between_angles = np.array([np.abs(DOA[i+1] - DOA[i]) for i in range(M-1)])
        if(np.sum(difference_between_angles > gap) == M - 1 and np.sum(difference_between_angles < (180 - gap)) == M - 1):
            break
    # print(DOA)
    return DOA

def create_closely_spaced_DOA(M:int, gap:float):
    """Create closely spaced sources scenario with minimal gap value for simulations

    Args:
        M (int): _description_
        gap (float): _description_

    Returns:
        np.ndarray: _description_
    """
    if (M == 2):
        first_DOA = np.round(np.random.rand(1) *  160 ,decimals = 2) - 80.00
        second_DOA = ((first_DOA + gap + 90 ) % 180) - 90
        return np.array([first_DOA, second_DOA])
    DOA = [np.round(np.random.rand(1) *  180 ,decimals = 2) - 90.00]
    while(len(DOA) < M):
        candidate_DOA = np.round(np.random.rand(1) *  180 ,decimals = 2) - 90.00
        difference_between_angles = np.array([np.abs(candidate_DOA - DOA[i]) for i in range(len(DOA))])
        if(np.sum(difference_between_angles < gap) == len(DOA) or np.sum((180 - difference_between_angles) < gap) == len(DOA)):
            DOA.append(candidate_DOA)
    return np.array(DOA)

class  Samples(SystemModel):
    '''
    Class used for defining and creating signals and observations.
    inherit from SystemModel class
    ...

    Attributes
    ----------
    says_str : str
        a formatted string to print out what the animal says
    name : str
        the name of the animal
    sound : str
        the sound that the animal makes
    num_legs : int
        the number of legs the animal has (default 4)

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes
    '''
    
    def __init__(self, scenario:str , N:int, M:int,
                    observations:int, freq_values:list = None):
        super().__init__(scenario, N, M, freq_values)
        self.T = observations
        # if DOA == None:
        #   self.DOA = (np.pi / 180) * np.array(create_DOA_with_gap(M = self.M, gap = 15)) # (~0.2 rad)
        # else: 
        #   self.DOA = (np.pi / 180) * np.array(DOA)                              # define DOA angels
    
    def set_doa(self, doa):
        if doa == None:
            self.DOA = (np.pi / 180) * np.array(create_DOA_with_gap(M = self.M, gap = 15)) # (~0.2 rad)
        else: 
          self.DOA = (np.pi / 180) * np.array(doa)                              # define DOA angels
        
    def samples_creation(self, mode, N_mean= 0, N_Var= 1, S_mean= 0, S_Var= 1, SNR= 10, eta = 0, geo_noise_var = 0):
        '''
        @mode = represent the specific mode in the specific scenario
                e.g. "Broadband" scenario in "non-coherent" mode
        '''
        
        if self.scenario.startswith("NarrowBand"):
            signal = self.signal_creation(mode, S_mean, S_Var, SNR)
            noise = self.noise_creation(N_mean, N_Var)
            A = np.array([self.steering_vec(theta, eta=eta, geo_noise_var=geo_noise_var) for theta in self.DOA]).T
            
            samples = (A @ signal) + noise 
            return samples, signal, A, noise

        elif self.scenario.startswith("Broadband"):
            samples = []
            SV = []
            f_axis = []
 
            signal = self.signal_creation(mode, S_mean, S_Var, SNR)
            noise = self.noise_creation(N_mean, N_Var)
            
            # TODO: check if the data creation became much slower
            
            for idx in range(self.f_sampling["Broadband"]):
                
                # mapping from index i to frequency f
                if idx > int(self.f_sampling["Broadband"]) // 2:
                    f = - int(self.f_sampling["Broadband"]) + idx
                else:
                    f = idx
                A = np.array([self.steering_vec(theta, f) for theta in self.DOA]).T
                samples.append((A @ signal[:, idx]) + noise[:, idx])
                # samples.append((A @ signal[:, idx % (int(self.f_sampling) // 2)]) + noise[:, idx])
                # samples.append((A @ signal[:, f]) + noise[:, idx])
                # samples.append((A @ signal[:, np.abs(f)]))
                SV.append(A)
                f_axis.append(f)
            samples = np.array(samples)
            SV = np.array(SV)
            samples_time_domain = np.fft.ifft(samples.T, axis=1)[:, :self.T]
            return samples_time_domain, signal, SV, noise

    def noise_creation(self, N_mean, N_Var):
        # for NarrowBand scenario Noise represented in the time domain
        if self.scenario.startswith("NarrowBand"):
            return np.sqrt(N_Var) * (np.sqrt(2) / 2) * (np.random.randn(self.N, self.T) + 1j * np.random.randn(self.N, self.T)) + N_mean
        
        # for Broadband scenario Noise represented in the frequency domain
        elif self.scenario.startswith("Broadband"):
            noise = np.sqrt(N_Var) * (np.sqrt(2) / 2) * (np.random.randn(self.N, len(self.time_axis["Broadband"]))\
                            + 1j * np.random.randn(self.N, len(self.time_axis["Broadband"]))) + N_mean
            return np.fft.fft(noise)
    
    def signal_creation(self, mode:str, S_mean = 0, S_Var = 1, SNR = 10):
        '''
        @mode = represent the specific mode in the specific scenario
                e.g. "Broadband" scenario in "non-coherent" mode
        '''
        amplitude = (10 ** (SNR / 10))
        ## NarrowBand signal creation 
        if self.scenario == "NarrowBand":
            if mode == "non-coherent": 
                # create M non-coherent signals
                return amplitude * (np.sqrt(2) / 2) * np.sqrt(S_Var) * (np.random.randn(self.M, self.T) + 1j * np.random.randn(self.M, self.T)) + S_mean
        
            elif mode == "coherent": 
                # Coherent signals: same amplitude and phase for all signals 
                sig = amplitude * (np.sqrt(2) / 2) * np.sqrt(S_Var) * (np.random.randn(1, self.T) + 1j * np.random.randn(1, self.T)) + S_mean
                return np.repeat(sig, self.M, axis = 0)
        
        
        ## Broadband signal creation
        if self.scenario.startswith("Broadband_simple"):
            # generate M random carriers
            carriers = np.random.choice(self.f_rng["Broadband"], self.M).reshape((self.M, 1))
                        
            # create M non-coherent signals
            if mode == "non-coherent":
                carriers_amp = amplitude * (np.sqrt(2) / 2) * (np.random.randn(self.M) + 1j * np.random.randn(self.M))
                carriers_signals = carriers_amp * np.exp(2 * np.pi * 1j * carriers\
                                @ self.time_axis["Broadband"].reshape((1, len(self.time_axis["Broadband"])))).T
                return np.fft.fft(carriers_signals.T)
            
            # Coherent signals: same amplitude and phase for all signals 
            if mode == "coherent":
                carriers_amp = amplitude * (np.sqrt(2) / 2) * (np.random.randn(1) + 1j * np.random.randn(1))
                carriers_signals = carriers_amp * np.exp(2 * np.pi * 1j * carriers[0] * self.time_axis["Broadband"])
                return np.tile(np.fft.fft(carriers_signals), (self.M, 1))

        ## Broadband signal creation
        if self.scenario.startswith("Broadband_OFDM"):
            num_sub_carriers = self.max_freq["Broadband"]   # number of subcarriers per signal
            # create M non-coherent signals
            signal = np.zeros((self.M, len(self.time_axis["Broadband"]))) + 1j * np.zeros((self.M, len(self.time_axis["Broadband"])))
            if mode == "non-coherent":
                for i in range(self.M):
                    for j in range(num_sub_carriers):
                        sig_amp = amplitude * (np.sqrt(2) / 2) * (np.random.randn(1) + 1j * np.random.randn(1))
                        signal[i] += sig_amp * np.exp(1j * 2 * np.pi * j * len(self.f_rng["Broadband"]) * self.time_axis["Broadband"] / num_sub_carriers)
                    signal[i] *=  (1/num_sub_carriers)          
                return np.fft.fft(signal)
             
            # Coherent signals: same amplitude and phase for all signals 
            signal = np.zeros((1, len(self.time_axis["Broadband"]))) + 1j * np.zeros((1, len(self.time_axis["Broadband"])))
            if mode == "coherent":
                for j in range(num_sub_carriers):
                    sig_amp = amplitude * (np.sqrt(2) / 2) * (np.random.randn(1) + 1j * np.random.randn(1))
                    signal += sig_amp * np.exp(1j * 2 * np.pi * j * len(self.f_rng["Broadband"]) * self.time_axis["Broadband"] / num_sub_carriers)
                signal *=  (1/num_sub_carriers)
                return np.tile(np.fft.fft(signal), (self.M, 1))
                
        else:
            return 0

if __name__ == "__main__":
    samp = Samples(scenario= "Broadband_simple", N= 8, M= 4, DOA = [10, 20, 30, 50], observations=10, freq_values = [0, 1000])
    samples_time_domain, signal, SV, noise = samp.samples_creation(mode="non-coherent")
    xFT = np.fft.fft(samples_time_domain)
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1000), np.abs(xFT[0, :1000]))
    plt.plot(200, np.abs(signal[0, 200]))
    plt.plot(600, np.abs(signal[0, 600]))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Signal amplitude')
    plt.show()
    plt.show()