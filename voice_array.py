"""
This code uses BSS and whisper identification to direction-find from an audio array.
"""
import numpy as np
from scipy import signal
import sounddevice as sd

# Receive Signal
def rx_sig():
    # should return a 12-channel signal. Real time? Is it possible?

# DoA Estimation
def doa_est(X):
    c = 3e8
    delta = 0.5
    M = 12 #num of elements/channels
    S = X@X.T /M
    [eigvec, eigval, ~] = np.linalg.svd(S)
    theta = np.linspace(-90,90,1e3)
    E_N = eigvec[:,M:-1]
    Pmu = np.zeros(len(theta))
    for k in range(0,len(theta))
        a = np.exp(-1j*2*np.pi*np.sind(theta[k]).T@(delta*np.arange(0,M))).T
        Pmu[k] = (a.T@(E_N@E_N.T)@a)**-1
    Pmusic = np.real(10*np.log10(Pmu))
    locs, _ = signal.findpeaks(Pmusic) #need some way to sort
    angles = theta[locs]
    # then what -> ask joe
    # probably make separate vectors based on number of speakers identified
    Y = np.zeros((len(X),len(angles)))
    for k in range(0,len(angles)):
        # "pre-whiten"
        a = np.exp(-1j*2*np.pi*np.sind(angles[k]).T@(delta*np.arange(0,M))).T
        Y[:,k] = (a @ a.T)**-1 @ (a @ a.T @ X) #"flaten"?
    return Y  
        

# Initial Filter
def preprocess(s,fs):
    v = signal.resample(s,10000*len(v)/fs) #decimation
    # VAD (Spectral Power)
    npoint = 1024
    N = len(v)
    # Windowing (hamming)
    win = signal.windows.hamming(1024,False)
    S = signal.fft.stft(v,window=win,noverlap=512)
    I = S * np.conj(S)/N 
    I[0] = I[1] # I gives power spectral density estimate over frequency
    # NR (weiner filter)
    IW = signal.weiner(I)
    # Framing (20-40ms block frames)
    blocklen = 0.03*fs # number of samples in 30ms block
    IW_block = np.reshape(IW,(blocklen,-1))
    
    IW_block = signal.windows.hamming(1025)
    # Normalization (Cepstral Mean and Variance Normalization CMVN)




# Blind Source Separation

# Whisper Identification

# Cepstrum Coefficient Calculation

# Subword Acoustic Phonetic Models

# TTS, Tracking, Iteration

if __name__ == "__main__":