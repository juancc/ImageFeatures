import numpy as np
import cv2
import matplotlib.pyplot as plt


def show(im, window='window'):
    """Show cv2 image"""
    cv2.imshow(window,im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def segmentation(img):
    """Return a binary image
     DEPENDING OF THE CASE USE DIFERENT SEGMENTATION
    """
    _, binary_img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)
    return binary_img


def plot_contour(im, contours):
    """Plot largest contour over image"""
    # Create a copy of the original image (in color) to draw the contour
    img_contour = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) 

    # Draw the largest contour on the image
    contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(img_contour, [contour], -1, (0, 0, 255), 2) 
    show(img_contour)

def read_image_and_extract_boundary(image_path, show_ims=True):
    """Function to read the image and extract the boundary points"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Perform image segmentation
    # Replace this with your own segmentation
    binary_img = segmentation(img)
    if show_ims: show(binary_img)

    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    if show_ims: plot_contour(img, contours)

    # Convert contour to complex numbers (x + iy) for Fourier transform
    boundary_points = contour[:, 0, :]
    complex_boundary = boundary_points[:, 0] + 1j * boundary_points[:, 1]
    
    return complex_boundary


def extract_fourier_descriptors(complex_boundary):
    """Function to compute Fourier descriptors and normalize them"""
    # Compute the Fourier Transform (Discrete Fourier Transform)
    descriptors = np.fft.fft(complex_boundary)
    
    # Make descriptors invariant to translation (ignore the first term)
    descriptors[0] = 0
    # Scale invariance: Normalize by the magnitude of the second descriptor
    descriptors /= np.abs(descriptors[1])
    # Return the Fourier descriptors (magnitude-only if rotation invariance is required)
    magnitudes = np.abs(descriptors)
    return descriptors, magnitudes

def plot_shape_from_descriptors(descriptors, num_descriptors=None):
    """Function to plot the shape reconstructed from Fourier descriptors"""
    # Optionally use only a subset of descriptors
    if num_descriptors is not None:
        descriptors = np.copy(descriptors)
        descriptors[num_descriptors:] = 0  # Zero out high-frequency components
    
    # Compute inverse Fourier transform to get the reconstructed shape
    reconstructed_shape = np.fft.ifft(descriptors)
    
    # Plot the reconstructed shape
    plt.figure(figsize=(6, 6))
    plt.plot(reconstructed_shape.real, reconstructed_shape.imag, '-o')
    plt.title(f"Reconstructed Shape using {num_descriptors or len(descriptors)} Descriptors")
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinate system
    plt.show()


def plot_fourier_magnitudes(magnitudes, num_frequencies=None):
    """Function to plot the magnitude of Fourier descriptors"""
    if num_frequencies is not None:
        magnitudes = magnitudes[:num_frequencies]  # Limit to the specified number of frequencies
    
    plt.figure(figsize=(10, 4))
    plt.plot(magnitudes, '-o')
    plt.title(f"Fourier Magnitudes (First {num_frequencies or len(magnitudes)} Frequencies)")
    plt.xlabel("Frequency Index")
    plt.ylabel("Magnitude")
    plt.show()


def generate_wave(freq, amplitude, length=1000):
    """Function to generate a sine wave for a given frequency and magnitude"""
    t = np.linspace(0, 2 * np.pi, length)  # Time axis for one period
    return amplitude * np.sin(freq * t)


def plot_frequency_waves(magnitudes, num_frequencies=None, length=1000):
    """Function to plot Fourier magnitudes as waveforms"""
    if num_frequencies is not None:
        magnitudes = magnitudes[:num_frequencies]  # Limit to the specified number of frequencies
    
    plt.figure(figsize=(10, 6))

    # Generate and plot each frequency as a wave
    for i, mag in enumerate(magnitudes):
        wave = generate_wave(i + 1, mag, length)  # Frequency index starts from 1
        plt.plot(wave, label=f"Frequency {i+1}")

    plt.title(f"Sine Waves Representing Magnitudes of Fourier Descriptors (First {len(magnitudes)} Frequencies)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()


def main(image_path, num_descriptors=None, num_frequencies=None):
    # Extract the boundary from the image
    complex_boundary = read_image_and_extract_boundary(image_path)
    # Compute the Fourier descriptors and their magnitudes
    descriptors, magnitudes = extract_fourier_descriptors(complex_boundary)

    #Plots
    # Reconstructed shape using the Fourier descriptors
    plot_shape_from_descriptors(descriptors, num_descriptors)
    # Magnitudes of the Fourier descriptors
    plot_fourier_magnitudes(magnitudes, num_frequencies)
    # Plot as features as waves
    plot_frequency_waves(magnitudes, 10, length=1000)



if __name__ == '__main__':
    image_path = 'Assets/mango.jpg' 
    main(image_path, num_descriptors=50, num_frequencies=50)
