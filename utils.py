import numpy as np
import math

# Amplitude Clipping
def Clipping(x, CL):
    # sigma: RMS of x
    sigma = np.sqrt(np.mean(np.square(np.abs(x))))
    CL = CL * sigma
    x_clipped = x
    clipped_idx = abs(x_clipped) > CL
    x_clipped[clipped_idx] = np.divide((x_clipped[clipped_idx] * CL), abs((x_clipped[clipped_idx])))
    return x_clipped

def PAPR(x):
    Power = np.abs(x)**2
    PeakP = np.max(Power)
    AvgP = np.mean(Power)
    PAPR_dB = 10 * np.log10(PeakP / AvgP)
    return PAPR_dB

def Modulation(bits, mu=2):
    bits_r = bits.reshape(int(len(bits) / mu), mu)
    # QAM modulation: (0,0)->-1-1j, (0,1)->(-1+1j), (1,0)->(1-1j), (1,1)->(1+1j)
    return (2 * bits_r[:,0] - 1) + 1j * (2 * bits_r[:,1] - 1)

def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

def addCP(OFDM_time, CP, CP_flag, mu, K):
    if not CP_flag:
        bits_noise = np.random.binomial(1, 0.5, size=(K * mu,))
        codeword_noise = Modulation(bits_noise, mu)
        OFDM_data_noise = codeword_noise
        OFDM_time_noise = np.fft.ifft(OFDM_data_noise)
        cp = OFDM_time_noise[-CP:]
    else:
        cp = OFDM_time[-CP:]
    return np.hstack([cp, OFDM_time])

def channel(signal, channelResponse, SNRdB):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdB / 10)
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*convolved.shape) + 1j * np.random.randn(*convolved.shape))
    return convolved + noise

def removeCP(signal, CP, K):
    return signal[CP:(CP+K)]

def ofdm_simulate(codeword, channelResponse, SNRdb, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers, Clipping_Flag, CR=1):
    payloadBits_per_OFDM = mu * len(dataCarriers)
    # Training Input (Pilot symbol)
    if P < K:
        bits = np.random.binomial(1, 0.5, size=(payloadBits_per_OFDM, ))
        QAM = Modulation(bits, mu)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue
        OFDM_data[dataCarriers] = QAM
    else:
        OFDM_data = pilotValue
    #IDFT,addCP
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time, CP, CP_flag, mu, K)
    OFDM_TX = OFDM_withCP

    if Clipping_Flag:
        OFDM_TX = Clipping(OFDM_TX, CR)

    OFDM_RX = channel(OFDM_TX, channelResponse, SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX, CP, K)

    #Target Inputs (Data Symbol)
    codeword_qam = Modulation(codeword, mu)

    if len(codeword_qam) != K:
        print('Length of code word is not equal to K, error!')

    symbol = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_codeword = addCP(OFDM_time_codeword, CP, CP_flag, mu, K)

    if Clipping_Flag:
        OFDM_withCP_codeword = Clipping(OFDM_withCP_codeword, CR)

    OFDM_RX_codeword = channel(OFDM_withCP_codeword, channelResponse, SNRdb)
    OFDM_noCP_codeword = removeCP(OFDM_RX_codeword, CP, K)

    #Concatenate real and imaginary parts of both received symbols for NN input
    #Size: K*2 + K*2 = 256
    result = np.concatenate((
        np.concatenate((np.real(OFDM_RX_noCP), np.imag(OFDM_RX_noCP))),
        np.concatenate((np.real(OFDM_noCP_codeword), np.imag(OFDM_noCP_codeword)))
    ))

    return result, abs(channelResponse)