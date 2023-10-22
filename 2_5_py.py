# %%
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import toeplitz
from xcorr import xcorr
import soundfile as sf

# %%
sr = 16000 # from max 2 x 8k frequencies present in signals
noise = sio.loadmat('data/problem2_5_noise')
sig = sio.loadmat('data/problem2_5_signal')

s = sig['signal'].ravel()
w = noise['noise'].ravel()
x = s + w

# we set,
# desired signal to "s" = d
# input signal to "x" = u

sf.write('data/signal.wav', s, sr )
sf.write('data/noise.wav', w, sr )
sf.write('data/noisy_signal.wav', x, sr)

def estimated_signal(filter_input, source, filter_length):

    r_uu = xcorr(filter_input, filter_input, L-1)
    R_uu = toeplitz(r_uu[L-1:])
    r_su = xcorr(source, filter_input, L-1)
    theta = np.linalg.solve(R_uu, r_su[L-1:])

    # Filter noisy signal
    shat = signal.lfilter(theta, 1, filter_input)

    return shat

def meansqerr(source, estimation):
    error = source - estimation
    mse = np.mean(error**2)
    return mse

Ls = np.arange(1,101,1)
MMSEs = np.zeros(len(Ls))

for i, L in enumerate(Ls):

    shat = estimated_signal(x, s, L)

    MMSEs[i] = meansqerr(s, shat)

plt.figure(figsize=(12,6), dpi=300)

plt.plot(MMSEs[:-1])
plt.title('MMSE across filter lengths in the range 1-100')
plt.xlabel('Filter length')
plt.ylabel('MMSE')
plt.grid(True)
plt.xticks(np.arange(0, 100, 10))

plt.tight_layout()
plt.show()


# %%

# we choose a filter length of 20, since there is not much more to gain by going further

L = 20
r_xx = xcorr(x, x, L-1)
R_xx = toeplitz(r_xx[L-1:])
r_sx = xcorr(s, x, L-1)

theta = np.linalg.solve(R_xx, r_sx[L-1:])
theta = theta.ravel()

# Filter noisy signal
dhat = signal.lfilter(theta, 1, x) 

# Show frequency response of the filter
norm_freq, freq_response_filter = signal.freqz(theta, 1)

# Show frequency response of the signals
norm_freq, freq_response_signal = signal.freqz(s, 1)
_, freq_response_noise = signal.freqz(w, 1)
_, freq_response_xn = signal.freqz(x, 1)
_, freq_response_filtered_signal = signal.freqz(dhat, 1)

fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
ax1.set_title('Digital filter frequency response')
ax1.plot(norm_freq, 20 * np.log10(abs(freq_response_filter)), 'b', label='filter')
ax1.plot(norm_freq, 20 * np.log10(abs(freq_response_signal)), color='r', label='sn')
ax1.plot(norm_freq, 20 * np.log10(abs(freq_response_xn)), 'orange', label='xn')
ax1.plot(norm_freq, 20 * np.log10(abs(freq_response_filtered_signal)), 'g', label='filtered signal')
ax1.set_ylabel('Amplitude [dB]')
ax1.set_xlabel('Frequency [rad/sample]')
ax1.grid()
ax1.legend()

plt.show()

