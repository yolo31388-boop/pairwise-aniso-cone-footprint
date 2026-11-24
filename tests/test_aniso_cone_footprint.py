import pytest
import graphics.aniso_cone_footprint as mod

CASES = [
    ("seam_continuity", (1,)),
    ("ellipse_axis", (1,)),
    ("lod_bias_aniso", (1,)),
    ("np2_mip_chain", (1,)),
    ("upload_barrier", (1,)),
    ("version_cache", (1,)),
    ("footprint", (1,)),
    ("sample", (1,)),
    ("choose_lod", (1,)),
    ("upload", (1,)),
    ("validate_sharpness", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
