import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

image = np.array(Image.open('./images/glioma2.jpg').convert('L'))

fourier = np.fft.fft2(image)
magnitude = np.abs(fourier)
phase = np.angle(fourier)
real = np.real(fourier)
imaginary = np.imag(fourier)

plt.figure(figsize=(8, 8))

plt.subplot(221)
plt.imshow(np.fft.fftshift(magnitude))
plt.title('Magnitude')

plt.subplot(222)
plt.imshow(np.fft.fftshift(phase))
plt.title('Phase')

plt.subplot(223)
plt.imshow(np.fft.fftshift(real))
plt.title('Real Part')

plt.subplot(224)
plt.imshow(np.fft.fftshift(imaginary))
plt.title('Imaginary Part')

plt.show()

reconstructed_image = np.fft.ifft2(fourier).real

plt.figure()
plt.imshow(reconstructed_image)
plt.title('Reconstructed Image')
plt.show()

mean_original = np.mean(image)
mean_reconstructed = np.mean(reconstructed_image)
sys.stdout.write(f'Mean of original image: {mean_original}')
sys.stdout.write(f'Mean of reconstructed image: {mean_reconstructed}')

fourier_rows = np.array([np.fft.fft(image_row) for image_row in image])

reconstructed_image_forty_harmonics = np.array([np.fft.ifft(fourier_row)[:40] for fourier_row in fourier_rows])

plt.figure()
plt.imshow(np.abs(reconstructed_image_forty_harmonics))
plt.title('Reconstructed Image using first 40 harmonics')
plt.show()
