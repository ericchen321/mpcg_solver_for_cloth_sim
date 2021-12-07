# %%
import numpy as np
import dhutils as dhu
from pathlib import Path

# %% specify gltf file path and load the .gltf and .bin
gltf_file_path = Path("../../lectures/assets/ken/gltf/low res") / \
    "ken_model_rigged_low_res.gltf"

# %%
human = dhu.load_SkinnedMesh_from_mixamo(gltf_file_path)
viewer = dhu.viewer(human)
viewer

# %%
viewer2 = dhu.viewer_skeleton(human)
viewer2

# %%
