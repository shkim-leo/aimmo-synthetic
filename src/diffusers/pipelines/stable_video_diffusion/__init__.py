from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    BaseOutput,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)


_dummy_objects = {}
_import_structure = {}

try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure.update(
        {
            "pipeline_stable_video_diffusion": [
                "StableVideoDiffusionPipeline",
                "StableVideoDiffusionPipelineOutput",
            ],
            "pipeline_stable_video_diffusion_img2vid_lcm": ["StableVideoDiffusionLCMPipeline"],
            "pipeline_stable_video_diffusion_vid2vid": ["StableVideoDiffusionVid2VidPipeline"],
            "pipeline_stable_video_diffusion_controlnet": ["StableVideoDiffusionControlNetPipeline"],
        }
    )


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        from .pipeline_stable_video_diffusion import (
            StableVideoDiffusionPipeline,
            StableVideoDiffusionPipelineOutput,
        )
        from .pipeline_stable_video_diffusion_img2vid_lcm import StableVideoDiffusionLCMPipeline
        from .pipeline_stable_video_diffusion_vid2vid import StableVideoDiffusionVid2VidPipeline
        from .pipeline_stable_video_diffusion_controlnet import StableVideoDiffusionControlNetPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
