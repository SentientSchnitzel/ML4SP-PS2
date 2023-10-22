# %%
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# %%
# load a wave file
# y: audio time series
# sr: sampling rate of y
y, sr = librosa.load('data/problem2_6.wav')
y_half = y[:y.__len__()//2]

# Define custom parameters
window_sizes = [2**8, 2**6, 2**4]
hop_sizes = [2**10, 2**8, 2**6]

# Window function
window_type = 'hann'

# Create subplots
fig, axs = plt.subplots(len(hop_sizes), len(window_sizes), figsize=(15, 8), dpi=300, sharex=True, sharey=True)

# Initialize variables to store min and max values for color normalization
vmin, vmax = float('inf'), -float('inf')

# Compute the STFT for each combination of window size and hop size
for i, window_size in enumerate(window_sizes):
    for j, hop_size in enumerate(hop_sizes):
        D = librosa.amplitude_to_db(librosa.stft(y_half, n_fft=window_size, hop_length=hop_size, window=window_type), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, hop_length=hop_size, x_axis='time', y_axis='log', ax=axs[i, j])
        axs[i, j].set_title(f'WS: {window_size}, HS: {hop_size}')
        axs[i, j].set_ylim([0, sr/2])

# Add colorbar
fig.subplots_adjust(right=1.05)
cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.7])
fig.colorbar(img, cax=cbar_ax, format="%+2.0f dB")

# Set labels
for ax in axs[-1, :]:
    ax.set_xlabel('Time (s)')
for ax in axs[:, 0]:
    ax.set_ylabel('Frequency (Hz)')
plt.tight_layout()
plt.show()

# %%
# plot spectrogram

window_size = 2**8
hop_size = 2**8
window_type = 'hann'

plt.figure(figsize=(10, 4), dpi=300)
D = librosa.amplitude_to_db(librosa.stft(y, n_fft=window_size, hop_length=hop_size, window=window_type), ref=np.max)
librosa.display.specshow(D, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Logarithm-frequency power spectrogram')
plt.xlabel('Time')
plt.ylabel('Logarithmic Frequency')
plt.tight_layout()
plt.show()



