# %%[markdown]
# ## <center>Skinning Assignment - Part II</center>
#
# In part I : Using the "ken" mesh constructed from scanned data,
# get a rigged gltf file using mixamo.
#
# Objective of part II:
# Implement linear blend skinning in python, using the bones and skinning weights
# from the glTF file. You can do this in either of two ways:
#
# Method A: Directly from the glTF file, using the glTF parsing functions provided
# in dhutils
#
# Method B: Using the pythreejs SkinnedMesh generated by the utility function `load_SkinnedMesh_from_mixamo`. Example usage:
#
#  `human = dhu.load_SkinnedMesh_from_mixamo(gltf_file_path)`
#
# [pythreejs](https://pythreejs.readthedocs.io) provides python bindings for the
# popular [Three.js](https://threejs.org/) graphics library.
# The advantage is that it may be easier to use, and I plan to add useful visualization
# features during the term.
# The caveat is that we're still refining this, since pythreejs does not currently
# implement a glTF loader.
# For full credit you must implement the skinning yourself, and not use the Three.js
# implementation.

# %%
import numpy as np
import dhutils as dhu
from pathlib import Path


# %% specify gltf file path and load the .gltf and .bin
# Replace path with the path to your glTF file from Part 1
gltf_file_path = Path("../../lectures/assets/ken/gltf/low res") / \
    "ken_model_rigged_low_res.gltf"

# %%[markdown]
# ### Method A: Steps required
# ## 1 : Understand nodes important for skinning in the gltf file :
# * To get access to all nodes in the gltf file - gltf.model.nodes
# * The nodes required for skinning are :
#   * The mesh node
#   * The skin node and
#   * The armature node
# * The armature node references the mesh node and the
#  root bone of the skeleton. The armature node also has any
#  base transformations on the skeleton.
#  * The mesh node contains a reference to the skin node.
#
#
# ## 2 : Using the information in the skin node.
# * The skin node has the list of joints that make up the skeleton.
#   * Each joint is an index to a node in the global list of nodes(gltf.model.nodes)
#   * Each joint node only stores its local transform matrix.
#   * TO DO : Traverse through each joint node and calculate its global transformation matrix.
#     You will need to access the children of each joint node.
#
# * The skin node also has a list of inverseBindMatrices for each joint.
#   * TO DO : get access to these inverse bind matrices.
#
#
# ## 3 : Using the information in the mesh node.
#  * TO DO : From the mesh node get per-vertex attributes :
#    * joints influencing each vertex.
#    * weights of each joints' influence on the vertex.
#
#
# ## 4 : Write the main skinning function :
# * For each vertex, get the joints infleuncing it and
#   their weights.
# * TO DO : linear blend skinning
#
# ### Method B: Steps required
# This is similar to Method A, except that you can access the [vertex geometry data](https://threejs.org/docs/#api/en/core/BufferGeometry)
# and [skeleton](https://threejs.org/docs/#api/en/objects/Skeleton) using `human.geometry` and `human.skeleton`. See


# %%
