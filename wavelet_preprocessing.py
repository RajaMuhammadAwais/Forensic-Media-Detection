import pywt
import numpy as np
import cv2

class WaveletPreprocessor:
    """
    A lightweight, plug-and-play preprocessing module that applies Discrete Wavelet Transform (DWT)
    to video frames to enhance frequency-domain artifacts before feeding them into a model.
    """
    def __init__(self, wavelet='haar', level=1):
        """
        Initializes the WaveletPreprocessor.

        Args:
            wavelet (str): The name of the wavelet to use (e.g., 'haar', 'db1', 'sym2').
            level (int): The decomposition level.
        """
        self.wavelet = wavelet
        self.level = level

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies DWT to a single video frame.

        Args:
            frame (np.ndarray): A single video frame (H, W, C) in BGR or RGB format.

        Returns:
            np.ndarray: The concatenated wavelet coefficients (Approximation, Horizontal, Vertical, Diagonal)
                        for each color channel. The output shape will be (H, W, C * 4) if level=1.
        """
        # Convert to grayscale if not already, or process each channel
        if frame.ndim == 3:
            processed_channels = []
            for i in range(frame.shape[2]):
                channel = frame[:, :, i]
                coeffs = pywt.wavedec2(channel, self.wavelet, level=self.level)
                
                # Reconstruct and concatenate the coefficients for a fixed output size
                # For simplicity and a fixed output size, we'll use the approximation and detail coefficients
                # from the highest level of decomposition.
                
                # Extract coefficients at the highest level
                cA, (cH, cV, cD) = coeffs[0], coeffs[1]
                
                # Resize all coefficients to the size of the approximation coefficient (cA)
                # This is a simplification to create a fixed-size feature map.
                # A more robust implementation might pad or use a different reconstruction method.
                
                h, w = cA.shape
                
                # Resize detail coefficients to match cA size
                cH_resized = cv2.resize(cH, (w, h), interpolation=cv2.INTER_LINEAR)
                cV_resized = cv2.resize(cV, (w, h), interpolation=cv2.INTER_LINEAR)
                cD_resized = cv2.resize(cD, (w, h), interpolation=cv2.INTER_LINEAR)
                
                # Concatenate the coefficients along the channel dimension
                # This creates a feature map of size (H/2^level, W/2^level, 4)
                concatenated_coeffs = np.stack([cA, cH_resized, cV_resized, cD_resized], axis=-1)
                processed_channels.append(concatenated_coeffs)
            
            # Stack the processed channels back together
            # Output shape: (H/2^level, W/2^level, C * 4)
            return np.concatenate(processed_channels, axis=-1)
        
        elif frame.ndim == 2:
            coeffs = pywt.wavedec2(frame, self.wavelet, level=self.level)
            cA, (cH, cV, cD) = coeffs[0], coeffs[1]
            
            h, w = cA.shape
            cH_resized = cv2.resize(cH, (w, h), interpolation=cv2.INTER_LINEAR)
            cV_resized = cv2.resize(cV, (w, h), interpolation=cv2.INTER_LINEAR)
            cD_resized = cv2.resize(cD, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Output shape: (H/2^level, W/2^level, 4)
            return np.stack([cA, cH_resized, cV_resized, cD_resized], axis=-1)
        
        else:
            raise ValueError("Frame must be 2D (grayscale) or 3D (color).")

# Note: For a real-world application, the resizing of detail coefficients is a simplification.
# The model would typically be designed to handle the multi-resolution output of DWT.
# This implementation provides a fixed-size output for easy integration into existing models.
