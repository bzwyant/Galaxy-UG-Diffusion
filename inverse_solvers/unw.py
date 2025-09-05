"""
Uncertainty noise weighting classes
"""
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
import numpy as np
# from models import get_model


_UNWS = {}


def register_unw(cls=None, *, name=None):
    """A decorator fore registering UNW classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _UNWS:
            raise ValueError(f"Already registered UNW with name: {local_name}")
        _UNWS[local_name] = name
        return cls
    
    if cls is None:
        return _register
    else:
        return _register(cls)
    

def get_unw(name: str, **kwargs):
    if _UNWS.get(name) is None:
        raise NameError(f"Name {name} is not defined.")
    return _UNWS[name](**kwargs)


class UNW(ABC):
    """UNW abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, config):
        """Construct a UNW
        
        Args:
            config: config file for the specific UNW
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def uncertainty_mask(self, y):
        """Compute the uncertainty mask for each image in the minibatch"""
        w_u = self.compute_uncertainty(y)
        return self._u_func(w_u)

    @abstractmethod
    def compute_uncertainty(self, y):
        pass

    @abstractmethod
    def _u_func(self, w_u):
        pass


class LaplacianPyramid(UNW):
    def __init__(self):
        super().__init__()
        self.psi_max = self.config.get('psi_max', 0.05)
        self.un_offset = self.config.get('un_offset', 0.4)
        self.levels = self.config.get('levels', 2)
        self.kernel_size = self.config.get('kernel_size', 5)
        self.sigma = self.config.get('sigma', 1.0)
        self.level_weights = self.config.get('level_weights', [1.0, 0.5])
        self.pad_mode = self.config.get('pad_mode', 'reflect')

        assert len(self.level_weights) == self.levels, "Length of level_weights must match levels"


    def compute_uncertainty(self, y):
        """Compute the uncertainty mask for each image in the minibatch"""
        laplacian_pyramid = self._build_laplacian_pyramid(y)

        # use variance as uncertainty measure
        variance_masks = self._compute_local_variance(laplacian_pyramid)

        # interpolate to original size and combine levels
        combined_mask = torch.zeros_like(y)
        for i, var_mask in enumerate(variance_masks):
            upsampled = F.interpolate(var_mask, size=y.shape[2:], mode='bilinear', align_corners=False)
            combined_mask += self.level_weights[i] * upsampled

        return combined_mask / sum(self.level_weights)
    


    def _u_func(self, psi):
        """Uncertainty weighting function from UPSR paper."""
        slope = (1 - self.un_offset) / self.psi_max
        linear_weights = slope * psi + self.un_offset

        return torch.where(psi < self.psi_max, linear_weights, 1.0)
        
        
    def _build_laplacian_pyramid(self, x):
        """Build a Laplacian pyramid for a batch of images with shape (B,C,H,W)"""
        gaussian_pyramid = [x]

        for i in range(self.levels):
            blurred = gaussian_blur2d(x, (self.kernel_size, self.kernel_size), (self.sigma, self.sigma))
            downsampled = F.avg_pool2d(blurred, kernel_size=2, stride=2)
            gaussian_pyramid.append(downsampled)

        laplacian_pyramid = []
        for i in range(self.levels):
            upsampled = F.interpolate(gaussian_pyramid[i + 1], 
                                      scale_factor=2, mode='bilinear')
            laplacian = gaussian_pyramid[i] - upsampled
            laplacian_pyramid.append(laplacian)
        
        return laplacian_pyramid
    
    
    def _compute_local_variance(self, x, kernel_size=3):

        pad_size = kernel_size // 2
        padded_image = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode=self.pad_mode)
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=x.device) / kernel_size**2

        mean = F.conv2d(padded_image, kernel, padding=0)
        mean_squared = F.conv2d(padded_image**2, kernel, padding=0)

        variance = mean_squared - mean**2
        return variance


# class DiffUNW(UNW):
#     def __init__(self, config):
#         super().__init__()
#         self.net_g = get_model(config['net_g'])

#         self.psi_max = config.get('psi_max', 0.05)
#         self.un_offset = config.get('un_offset', 0.4)

    
#     def compute_uncertainty(self, y):
#         x_est = (self.net_g(y * 0.5 + 0.5) - 0.5) / 0.5
#         uncertainty = 0.5 * np.abs(x_est - y)
#         w_u = self._u_func(uncertainty)
#         return w_u
        

#     def _u_func(self, psi):
#         """Uncertainty weighting function from UPSR paper."""

#         slope = (1 - self.un_offset) / self.psi_max
#         linear_weights = slope * psi + self.un_offset

#         return torch.where(psi < self.psi_max, linear_weights, 1.0)