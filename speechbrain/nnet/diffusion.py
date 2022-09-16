"""A wrapper for the HuggingFace Diffusers library providing
support for Denoising Diffusion Models


Authors
 * Artem Ploujnikov 2022
"""

from torch import nn

_ERR_DIFUSERS_REQUIRED = (
    "The optional dependency diffusers must be installed to run this recipe. "
    "Please run pip install diffusers"
)

class UNetWrapper(nn.Module):
    """A UNet wrapper
    
    Arguments
    ---------
    impl: nn.Module
        the underlying implementation
    """
    def __init__(self, impl):
        super().__init__()
        self.impl = impl

    def __getattr__(self, name):
        """Passes through function calls to the underlying implementation"""
        return (
            self._modules[name]
            if name in self._modules
            else getattr(self.impl, name))            
    
    def forward(self, *args, **kwargs):
        """Runs the model and returns the sample as a tensor"""
        result = self.impl(*args, **kwargs)
        return result["sample"]


def unet_2d(*args, **kwargs):
    """Creates a HuggingFace Diffusers UNet2D model. All
    arguments are passed through.

    See: https://huggingface.co/docs/diffusers/api/models#diffusers.UNet2DModel    
    """
    try:
        from diffusers.models.unet_2d import UNet2DModel
    except ImportError:
        raise ImportError(_ERR_DIFUSERS_REQUIRED)
    return UNetWrapper(impl=UNet2DModel(*args, **kwargs))


_SCHEDULER_MAP = {
    "ddim": "DDIMScheduler",
    "ddpm": "DDPMScheduler",
    "pndm": "PNDMScheduler",
    "score_sde_ve": "ScoreSdeVeScheduler",
    "score_sde_vp": "ScoreSdeVpScheduler",
    "karras_ve": "KarrasVeScheduler",
}

_PIPELINE_MAP = {
    "ddim": "DDIMPipeline",
    "ddpm": "DDPMPipeline",
    "ldm": "LDMPipeline",
    "pndm": "PNDMPipeline",
    "score_sde_ve": "ScoreSdeVePipeline",
    "karras_ve": "KarrasVePipeline"
}

class Scheduler(nn.Module):
    """A noise scheduler wrapper
    
    Arguments
    ---------
    impl: nn.Module
        the underlying implementation (from Diffusers)
    """

    def __init__(self, impl):
        self.impl = impl

    def step(self, *args, **kwargs):
        """Performs a step and returns the previous sample"""
        output = self.impl.step(*args, **kwargs)
        return output.prev_sample

    def __getattr__(self, name):
        """Passes through function calls to the underlying implementation"""
        return getattr(self.impl, name)


def scheduler(name, *args, **kwargs):
    """
    Instantiates a denoising scheduler from HuggingFace
    Diffusers

    Arguments
    ---------
    name: str
        the schedduler name. 
        Supported schedulers:
        - ddpm
        - lms_discrete
        - pndms
        - score_sde_ve
        - score_sde_vp
    """
    class_name = _SCHEDULER_MAP.get(name)
    if not class_name:
        raise ValueError(f"Scheduler {name} is not supported")

    try:
        import diffusers
    except:
        raise ImportError(_ERR_DIFUSERS_REQUIRED)

    scheduler_class = getattr(diffusers, class_name)
    impl = scheduler_class(*args, **kwargs)
    return Scheduler(impl)


def pipeline(name, *args, **kwargs):
    """
    Instantiates a generation pipeline via HuggingFace Diffusers

    Arguments
    ---------
    name: str
        the schedduler name. 
        Supported pipelines:
        - ddim
        - ddpm
        - ldm
        - pndm
        - score_sde_ve
        - karras_ve
       
    """
    class_name = _PIPELINE_MAP.get(name)
    if not class_name:
        raise ValueError(f"Scheduler {name} is not supported")

    try:
        import diffusers
    except:
        raise ImportError(_ERR_DIFUSERS_REQUIRED)

    pipeline_class = getattr(diffusers, class_name)
    unet = kwargs.get("unet")
    if isinstance(unet, UNetWrapper):
        kwargs = dict(kwargs, unet=unet.impl)
    return pipeline_class(*args, **kwargs)
