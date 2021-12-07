"""Simple Utilities for Digital Humans

    >>> import dhutils as dhu
    >>> human = dhu.load_SkinnedMesh (glTF_file_path)
    >>> viewer = dhu.viewer(human)
    >>> viewer
    """

# %%
import math
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from IPython.display import display
import ipywidgets
import pythreejs as THREE
from gltflib import GLTF
from .gltf_parsing_helper_functions import *


# %%
def load_glTF(glTFpath):
    """Construct plain glTF THREE.Mesh.

    Args:
        glTFpath (Path): [TODO]

    Returns:
        THREE.Mesh
    """
    gltf, bin_data = load_gltf_and_bin(glTFpath)
    # get all the THREE.Buffers, accessors and THREE.Buffer views (each is a list)
    accessors, bufferViews, _buffers = get_accessors_bufferViews_buffers(
        gltf)
    # get the mesh and the attached skin
    mesh, _skin = get_mesh_and_skin_nodes(gltf)

    # get all the attribute accessors as a dictionary
    attribute_accessor_ids_dict = get_mesh_attributes_accessor_ids(mesh)

    acc_position_id = attribute_accessor_ids_dict['position_accessor_id']
    acc_normal_id = attribute_accessor_ids_dict['normal_accessor_id']
    acc_faces_id = attribute_accessor_ids_dict['faces_accessor_id']

    # get vertices
    points = get_data_from_accessor(
        accessors[acc_position_id], bin_data, bufferViews)

    # get vertices
    normals = get_data_from_accessor(
        accessors[acc_normal_id], bin_data, bufferViews)

    # get faces as a list of integers
    faces_as_list_of_scalars = get_data_from_accessor(
        accessors[acc_faces_id], bin_data, bufferViews)

    geometry = THREE.BufferGeometry(
        attributes={
            'position': THREE.BufferAttribute(np.array(points, dtype=np.float32), normalized=False),
            'normal': THREE.BufferAttribute(np.array(normals, dtype=np.float32), normalized=False),
            'index': THREE.BufferAttribute(np.array(faces_as_list_of_scalars, dtype=np.uint16)),
        }
    )

    # TODO get material from gltf
    mesh = THREE.Mesh(geometry, THREE.MeshStandardMaterial())

    return mesh


def load_SkinnedMesh_from_mixamo(glTFpath):
    """Construct ThreeJS SkinnedTHREE.Mesh from glTF with Armature node.

    Args:
        glTFpath (Path): [TODO]

    Returns:
        SkinnedMesh: pythreejs skinned mesh
    """
    gltf, bin_data = load_gltf_and_bin(glTFpath)
    # get all the THREE.Buffers, accessors and THREE.Buffer views (each is a list)
    accessors, bufferViews, _buffers = get_accessors_bufferViews_buffers(
        gltf)
    # get the mesh and the attached skin
    mesh, skin = get_mesh_and_skin_nodes(gltf)
    joints = skin.joints

    # get all the attribute accessors as a dictionary
    attribute_accessor_ids_dict = get_mesh_attributes_accessor_ids(mesh)

    acc_position_id = attribute_accessor_ids_dict['position_accessor_id']
    acc_normal_id = attribute_accessor_ids_dict['normal_accessor_id']
    acc_faces_id = attribute_accessor_ids_dict['faces_accessor_id']
    acc_weights_id = attribute_accessor_ids_dict['weights_accessor_id']
    acc_joints_id = attribute_accessor_ids_dict['joints_accessor_id']

    # get vertices
    points = get_data_from_accessor(
        accessors[acc_position_id], bin_data, bufferViews)

    # get vertices
    normals = get_data_from_accessor(
        accessors[acc_normal_id], bin_data, bufferViews)

    # get faces as a list of integers
    faces_as_list_of_scalars = get_data_from_accessor(
        accessors[acc_faces_id], bin_data, bufferViews)

    # get joints and weights per vertex, mesh primitives - JOINTS_0 and WEIGHTS_0
    influence_weights = get_data_from_accessor(
        accessors[acc_weights_id], bin_data, bufferViews)

    joint_influences = get_data_from_accessor(
        accessors[acc_joints_id], bin_data, bufferViews)

    # TODO test skinIndex
    geometry = THREE.BufferGeometry(
        attributes={
            'position': THREE.BufferAttribute(np.array(points, dtype=np.float32), normalized=False),
            'normal': THREE.BufferAttribute(np.array(normals, dtype=np.float32), normalized=False),
            'index': THREE.BufferAttribute(np.array(faces_as_list_of_scalars, dtype=np.uint16)),
            'skinIndex': THREE.BufferAttribute(np.array(joint_influences, dtype=np.uint16)),
            'skinWeight': THREE.BufferAttribute(np.array(influence_weights, dtype=np.float32), normalized=False),
        }
    )

    # TODO get material from gltf
    skinned_mesh = THREE.SkinnedMesh(geometry, THREE.MeshStandardMaterial(
        side='DoubleSide', skinning=True))

    # make all the bones
    bones = []
    for joint in joints:
        # glTF joints correspond to ThreeJS bones
        bone_node = gltf.model.nodes[joint]
        bone = THREE.Bone()
        if bone_node.translation != None:
            bone.position = bone_node.translation
        if bone_node.rotation != None:
            bone.quaternion = bone_node.rotation
        if bone_node.scale != None:
            bone.scale = bone_node.scale
        bones.append(bone)

    # construct skeleton graph
    # relying on the bones and joints having same order. Is there a cleaner way?
    for b in range(len(bones)):
        children = gltf.model.nodes[joints[b]].children
        if children != None:
            for child_node in children:
                bones[b].add(bones[joints.index(child_node)])

    skeleton = THREE.Skeleton(bones)
    skinned_mesh.add(skeleton.bones[0])
    skinned_mesh.skeleton = skeleton

    return skinned_mesh


def viewer(human):
    view_width = 600
    view_height = 400
    camera = THREE.PerspectiveCamera(
        position=[2, 1, 2], aspect=view_width/view_height)
    key_light = THREE.DirectionalLight(position=[0, 10, 10])
    ambient_light = THREE.AmbientLight()
    axes_helper = THREE.AxesHelper(1)
    skeleton_helper = THREE.SkeletonHelper(human)

    scene = THREE.Scene(children=[human,
                                  axes_helper, skeleton_helper,
                                  camera, key_light, ambient_light])
    controller = THREE.OrbitControls(controlling=camera)
    renderer = THREE.Renderer(camera=camera, scene=scene, controls=[controller],
                              width=view_width, height=view_height)

    return renderer


def viewer_skeleton(human):
    view_width = 600
    view_height = 400
    camera = THREE.PerspectiveCamera(
        position=[2, 1, 2], aspect=view_width/view_height)
    key_light = THREE.DirectionalLight(position=[0, 10, 10])
    ambient_light = THREE.AmbientLight()
    axes_helper = THREE.AxesHelper(1)
    skeleton_helper = THREE.SkeletonHelper(human)

    scene = THREE.Scene(children=[axes_helper, skeleton_helper,
                                  camera, key_light, ambient_light])
    controller = THREE.OrbitControls(controlling=camera)
    renderer = THREE.Renderer(camera=camera, scene=scene, controls=[controller],
                              width=view_width, height=view_height)

    return renderer

# %%


def standard_rectangle(Lx, Ly, Nx, Ny):
    ''' Conveniece function to construct a rectangle in the XY plane, extending from the X axis downards (along -Y)

    Args:
        Lx (float): length of rectangle in X
        Ly (float): length of rectangle in Y
        Nx (int): number of segments in X
        Ny (int): number of segments in Y

    Returns:
        positions: array of vertex positions
        faces: array of faces (consistently oriented vertex loop for each face)
    '''

    dx = Lx / Nx
    dy = Ly / Ny

    # vertex positions

    positions = np.zeros(((Nx + 1) * (Ny + 1), 3), dtype=np.float32)
    j = 0  # vertex index
    #  with x to right and y up, j's look like this
    #  0 - 3 - 6
    #  | \ | \ |
    #  1 - 4 - 7
    #  | \ | \ |
    #  2 - 5 - 8

    for jx in range(Nx+1):
        for jy in range(Ny + 1):
            positions[j, 0] = jx * dx
            positions[j, 1] = jy * - dy  # negative sign to make it hang down
            j += 1

    # faces (elements)
    faces = np.zeros((Nx * Ny * 2, 3), dtype=np.uint16)
    i = 0  # face index
    j = 0  # vertex index
    for jx in range(Nx):
        for jy in range(Ny):
            faces[i] = [j, j + 1, j + Ny + 2]
            faces[i + 1] = [j, j + Ny + 2, j + Ny + 1]
            i += 2
            j += 1
        j += 1

    return positions, faces


# %%

def mesh_animation(times, xt, faces):
    """ Animate a mesh from a sequence of mesh vertex positions

        Args:
        times   - a list of time values t_i at which the configuration x is specified
        xt      -   i.e., x(t). A list of arrays representing mesh vertex positions at times t_i.
                    Dimensions of each array should be the same as that of mesh.geometry.array
        TODO nt - n(t) vertex normals
        faces    - array of faces, with vertex loop for each face

        Side effects:
            displays rendering of mesh, with animation action

        Returns: None
        TODO optionally return
        renderer - THREE.Render to show the default scene
        position_action - THREE.AnimationAction IPython widget
    """

    position_morph_attrs = []
    for pos in xt[1:]:  # xt[0] uses as the Mesh's default/initial vertex position
        position_morph_attrs.append(
            THREE.BufferAttribute(pos, normalized=False))

    # Testing mesh.geometry.morphAttributes = {'position': position_morph_attrs}
    geom = THREE.BufferGeometry(
        attributes={
            'position': THREE.BufferAttribute(xt[0], normalized=False),
            'index': THREE.BufferAttribute(faces.ravel())
        },
        morphAttributes={
            'position': position_morph_attrs
        }
    )
    matl = THREE.MeshStandardMaterial(
        side='DoubleSide', color='red', wireframe=True, morphTargets=True)

    mesh = THREE.Mesh(geom, matl)

    # create key frames
    position_track = THREE.NumberKeyframeTrack(
        name='.morphTargetInfluences', times=times, values=np.identity(len(times)).tolist())
    # create animation clip from the morph targets
    position_clip = THREE.AnimationClip(tracks=[position_track])
    # create animation action
    position_action = THREE.AnimationAction(
        THREE.AnimationMixer(mesh), position_clip, mesh)

    # TESTING
    camera = THREE.PerspectiveCamera(position=[5, 3, 5], aspect=600/400)
    scene = THREE.Scene(children=[mesh,
                                  camera,
                                  THREE.AxesHelper(1),
                                  THREE.DirectionalLight(
                                      position=[3, 5, 1], intensity=0.6),
                                  THREE.AmbientLight(intensity=0.5)])
    renderer = THREE.Renderer(camera=camera, scene=scene,
                              controls=[THREE.OrbitControls(
                                  controlling=camera)],
                              width=600, height=400)

    display(renderer, position_action)

    # return renderer, position_action


def mesh_display(x, faces):
    """ Display a simulation a single mesh specified by vertex positios

        Args: 
        x      -   vertex positions, same dimensions as mesh.geometry.array
        faces    - array of faces, with vertex loop for each face

        Side effects:
            displays rendering of mesh

        Returns: None
        TODO optionally return
        renderer - THREE.Render to show the default scene
        position_action - THREE.AnimationAction IPython widget
    """

    geom = THREE.BufferGeometry(
        attributes={
            'position': THREE.BufferAttribute(x, normalized=False),
            'index': THREE.BufferAttribute(faces.ravel())
        }
    )
    matl = THREE.MeshStandardMaterial(
        side='DoubleSide', color='red', wireframe=True, morphTargets=True)

    mesh = THREE.Mesh(geom, matl)

    camera = THREE.PerspectiveCamera(position=[5, 3, 5], aspect=600/400)
    scene = THREE.Scene(children=[mesh,
                                  camera,
                                  THREE.AxesHelper(1),
                                  THREE.DirectionalLight(
                                      position=[3, 5, 1], intensity=0.6),
                                  THREE.AmbientLight(intensity=0.5)])
    renderer = THREE.Renderer(camera=camera, scene=scene,
                              controls=[THREE.OrbitControls(
                                  controlling=camera)],
                              width=600, height=400)

    display(renderer)
