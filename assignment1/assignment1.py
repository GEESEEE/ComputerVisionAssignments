import numpy as np
import cv2
import matplotlib.pyplot as plt

im = cv2.imread('car_dis.png', cv2.IMREAD_GRAYSCALE)
f = np.fft.fft2(im)
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(np.abs(fshift) + 1)

# Display and save the original image
plt.figure()
plt.imshow(im, cmap='gray')
plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.savefig('Original_Image.png')

# Display and save the magnitude spectrum
plt.figure()
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Log Magnitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Frequency (Hz)')
plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.colorbar(label='Magnitude (dB)')
plt.savefig('Log_Magnitude_Spectrum.png')

# Average filter
# Define the filter (e.g., a simple averaging filter)
kernel_size_vert = 1
kernel_size_hori = 9
kernel = np.ones((kernel_size_vert, kernel_size_hori), np.float32) / (kernel_size_vert * kernel_size_hori)

# Apply the filter using the default border type (zero-padding)
filtered_im = cv2.filter2D(im, -1, kernel, borderType=cv2.BORDER_CONSTANT)

# Using matplotlib to plot the filtered image with default border type
plt.figure(figsize=(6, 6))
plt.imshow(filtered_im, cmap='gray')
plt.title("Filtered Image (Default Border)")
plt.axis('off')
plt.tight_layout()
plt.savefig("filtered_default.png", dpi=150, bbox_inches='tight')

# Compute FFT and shift the zero-frequency component to the center
f_transform = np.fft.fft2(filtered_im)
f_shift = np.fft.fftshift(f_transform)
filtered_magnitude_spectrum = np.log(np.abs(f_shift))

# Display the log-magnitude spectrum
plt.figure()
plt.imshow(filtered_magnitude_spectrum, cmap='gray')
plt.title('Log Magnitude Spectrum of Filtered Image')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Frequency (Hz)')
plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.colorbar(label='Magnitude (dB)')
# Save the spectrum image
plt.savefig("Imlogmag_filtered.png", dpi=150, bbox_inches='tight')

# Apply the filter using 'reflect' border type
natural_im = cv2.filter2D(im, -1, kernel, borderType=cv2.BORDER_REFLECT)

# Using matplotlib to plot the filtered image
plt.figure(figsize=(6, 6))
plt.imshow(natural_im, cmap='gray')
plt.title("Natural Filtered Image")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Frequency (Hz)')
plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.colorbar(label='Magnitude (dB)')
plt.savefig("natural_filtered.png", dpi=150, bbox_inches='tight')

def psf2otf(psf, shape):
    """
    Convert Point Spread Function (PSF) to Optical Transfer Function (OTF).
    """
    # Padding PSF with zeros to match the shape
    pad_psf = np.zeros(shape)
    pad_psf[:psf.shape[0], :psf.shape[1]] = psf

    # Circularly shift PSF to have center at (0, 0)
    for i in range(psf.ndim):
        pad_psf = np.roll(pad_psf, -int(np.floor(psf.shape[i] / 2)), axis=i)

    # Compute the OTF
    otf = np.fft.fft2(pad_psf)
    return otf

otf = psf2otf(kernel, im.shape)
otf_shifted = np.fft.fftshift(otf)

# Extract the imaginary part and compute its maximum absolute value
max_imag_otf = np.max(np.abs(otf.imag))
print(f"max(abs(imag(OTF))) = {max_imag_otf}")

# Displaying the magnitude of OTF
plt.figure()
plt.imshow(np.abs(otf_shifted), cmap='gray')
plt.title("Magnitude of OTF")
plt.colorbar(label="Magnitude")
plt.xlabel("u")
plt.ylabel("v")
plt.tight_layout()
plt.savefig("magnitude_OTF.png", dpi=150, bbox_inches='tight')

N, M = im.shape

# Define the rectangle's center and size
rho = 28
row = N // 2
col = M // 2 + rho
s = 11

# Extracting the attenuation factor
attenuation_factor = np.abs(otf_shifted[row, col])
print(f"Attenuation factor of the dominant frequency: {attenuation_factor}")

# Calculate the range for the rectangle
rstart = row - s // 2
rend = row + s // 2
cstart = col - s // 2
cend = col + s // 2

# Construct the H matrix
H = np.ones((N, M))
H[rstart:rend+1, cstart:cend+1] = 0
H[N-rend:N-rstart+1, M-cend:M-cstart+1] = 0

# Compute the Fourier transform of the image
f = np.fft.fft2(im)
fshift = np.fft.fftshift(f)
fshift_filtered = fshift * H
f_ishift = np.fft.ifftshift(fshift_filtered)
im_back = np.fft.ifft2(f_ishift)

# Check the imaginary parts
max_imag = np.max(np.abs(im_back.imag))

# If the imaginary parts are below the threshold, set them to zero.
threshold = 1e-14
im_back[np.where(im_back < threshold)] = 0
print(f"max(abs(imag(imresult(:)))) = {max_imag}")

# Plotting the magnitude of the OTF (H)
plt.figure()
plt.imshow(H, cmap='gray')
plt.title("OTF Magnitude (H)")
plt.colorbar()

# Plotting the log-magnitude of the filtered image
plt.figure()
filtered_magnitude_spectrum_2 = 20 * np.log(np.abs(fshift_filtered) + 1) # Adding 1 to avoid log(0)
plt.imshow(filtered_magnitude_spectrum_2, cmap='gray')
plt.title("Log-Magnitude of Filtered Image")
plt.colorbar()
plt.savefig("logmag_filtered.png", dpi=150, bbox_inches='tight')

# Plotting the filtered image
plt.figure()
plt.imshow(np.real(im_back), cmap='gray')
plt.title("Newly Filtered Image")
plt.colorbar()
plt.savefig("newly_filtered.png", dpi=150, bbox_inches='tight')
plt.show()





