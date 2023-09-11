import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read the image and convert it to grayscale
img = cv2.imread('car_dis.png', cv2.IMREAD_GRAYSCALE)

# Calculate the 2D Fourier Transform of the image
f = np.fft.fft2(img)

# Shift the zero frequency component to the center
fshift = np.fft.fftshift(f)

# Calculate the magnitude and logarithmic scale
magnitude_spectrum = np.log(np.abs(fshift) + 1)

# Display and save the original image
plt.figure()
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.savefig('Original_Image.png')

# Display and save the magnitude spectrum
plt.figure()
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Log Magnitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Frequency (Hz)')
plt.xticks([]), plt.yticks([])
plt.colorbar(label='Magnitude (dB)')
plt.savefig('Log_Magnitude_Spectrum.png')

# Average filter
# You might need to adjust the kernel size (e.g., (5,5)) for best results.
filtered_im = cv2.blur(img, (5,5))

# Save filtered image
cv2.imwrite('IMfil.jpg', filtered_im)

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
plt.colorbar(label='Magnitude (dB)')

# Save the spectrum image
plt.savefig("Imlogmag_filtered.png", dpi=150, bbox_inches='tight')

plt.show()
