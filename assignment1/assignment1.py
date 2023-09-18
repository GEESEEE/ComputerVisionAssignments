import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read the image and convert it to grayscale
im = cv2.imread('car_dis.png', cv2.IMREAD_GRAYSCALE)

# Calculate the 2D Fourier Transform of the image
f = np.fft.fft2(im)

# Shift the zero frequency component to the center
fshift = np.fft.fftshift(f)

# Calculate the magnitude and logarithmic scale
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
kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

# Apply the filter using the default border type (zero-padding)
filtered_im = cv2.filter2D(im, -1, kernel, borderType=cv2.BORDER_CONSTANT)

# Using matplotlib to plot the filtered image with default border type
plt.figure(figsize=(6, 6))
plt.imshow(filtered_im, cmap='gray')
plt.title("Filtered Image (Default Border)")
plt.axis('off')
plt.tight_layout()

# Compute FFT and shift the zero-frequency component to the center
f_transform = np.fft.fft2(filtered_im)
f_shift = np.fft.fftshift(f_transform)
magnitude_spectrum = np.log(np.abs(f_shift))

# Display the log-magnitude spectrum
plt.figure()
plt.imshow(magnitude_spectrum, cmap='gray')
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
    # Pad the PSF with zeros to match the shape
    padded_psf = np.zeros(shape)
    padded_psf[:psf.shape[0], :psf.shape[1]] = psf

    # Circularly shift PSF to have the center at (0,0)
    for i in range(psf.ndim):
        padded_psf = np.roll(padded_psf, -psf.shape[i] // 2, axis=i)

    # Compute the OTF
    otf = np.fft.fftn(padded_psf)

    return otf


# Compute the OTF
otf = psf2otf(kernel, im.shape)
magnitude = np.abs(otf)

# Extract the imaginary part and compute its maximum absolute value
max_imaginary_value = np.max(np.abs(otf.imag))

print(f"Max absolute value of the imaginary part of OTF: {max_imaginary_value:.6f}")

# Displaying the magnitude of OTF
plt.figure(figsize=(6, 6))
plt.imshow(np.fft.fftshift(magnitude), cmap='gray', extent=(-im.shape[1] // 2, im.shape[1] // 2, -im.shape[0] // 2, im.shape[0] // 2))
plt.title('Magnitude of OTF')
plt.xlabel('u')
plt.ylabel('v')
plt.colorbar(label='Magnitude')
plt.tight_layout()


dominant_row = im.shape[0] // 2
dominant_col = im.shape[1] // 2

# Extracting the attenuation factor
attenuation_factor = np.abs(otf[dominant_row, dominant_col])

print(f"Attenuation factor is: {attenuation_factor:.6f}")


N, M = im.shape

# Define the rectangle's center and size
rho = 28
row = 128
col = 100  # You'll have obtained these from a previous step
s = 11  # The size of the rectangle, an odd integer

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

# Apply the filter H
fshift_filtered = fshift * H

# Inverse Fourier transform
f_ishift = np.fft.ifftshift(fshift_filtered)
im_back = np.fft.ifft2(f_ishift)

# Check the imaginary parts
max_imag = np.max(np.abs(im_back.imag))

# If the imaginary parts are below the threshold, set them to zero.
threshold = 1e-14
im_back[np.where(im_back < threshold)] = 0

print(f"max(abs(imag(imresult(:)))) = {max_imag}")


magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

# Plotting for visual inspection
plt.figure(figsize=(10,10))
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.colorbar()

# Plotting the magnitude of the OTF (H)
plt.figure()
plt.imshow(H, cmap='gray')
plt.title("OTF Magnitude (H)")
plt.colorbar()

# Plotting the log-magnitude of the filtered image
plt.figure()
magnitude_spectrum = 20*np.log(np.abs(fshift_filtered) + 1) # Adding 1 to avoid log(0)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Log-Magnitude of Filtered Image")
plt.colorbar()

# Plotting the original distorted image
plt.figure()
plt.imshow(im, cmap='gray')
plt.title("Original Distorted Image")

# Plotting the filtered image
plt.figure()
plt.imshow(np.real(im_back), cmap='gray')
plt.title("Newly Filtered Image")
plt.show()
