"""Basic functions for parsing glTF files to construct pythreejs BufferGeometry objects

Currently limited to reading geometry information

>>> import gltf_parsing_helper_functions as gp
>>> gp.load


"""

import numpy as np
import struct

from gltflib import GLTF


def get_format_char_byte_len(accessor):
    format_character = ''
    byte_len = 0

    if accessor.componentType == 5120:
        # the encoding is signed char
        format_character = 'b'
        byte_len = 1
    if accessor.componentType == 5121:
        # unsigned char
        format_character = 'B'
        byte_len = 2
    if accessor.componentType == 5122:
        # short
        format_character = 'h'
        byte_len = 2
    if accessor.componentType == 5123:
        # unsigned short - 'H'
        format_character = 'H'
        byte_len = 2
    if accessor.componentType == 5125:
        # unsigned int
        format_character = 'I'
        byte_len = 4
    if accessor.componentType == 5126:
        # float
        format_character = 'f'
        byte_len = 4

    return format_character, byte_len


def get_scalar_data_from_bin(bin_data, num_components, accessor, bufferViews):

    format_character, byte_len = get_format_char_byte_len(accessor)

    data = []

    bufferViewIndex = accessor.bufferView
    #numIndices = accessor.count
    bufferView = bufferViews[bufferViewIndex]

    # it can be assumed that num components is 1
    for i in range(bufferView.byteOffset, bufferView.byteOffset+bufferView.byteLength, num_components*byte_len):
        val_1 = struct.unpack(format_character, bin_data[i:i+byte_len])[0]
        data.append(val_1)

    return data


def get_vec_data_from_bin(bin_data, num_components, accessor, bufferViews):
    format_character, byte_len = get_format_char_byte_len(accessor)

    data = []

    bufferViewIndex = accessor.bufferView
    #numIndices = accessor.count
    bufferView = bufferViews[bufferViewIndex]

    # it can be assumed that num components is 1
    for i in range(bufferView.byteOffset, bufferView.byteOffset+bufferView.byteLength, num_components*byte_len):
        temp_data = []
        val_1 = struct.unpack(format_character, bin_data[i:i+byte_len])[0]
        val_2 = struct.unpack(
            format_character, bin_data[i+byte_len:i+2*byte_len])[0]
        temp_data.append(val_1)
        temp_data.append(val_2)

        if num_components > 2:
            val_3 = struct.unpack(
                format_character, bin_data[i+2*byte_len:i+3*byte_len])[0]
            temp_data.append(val_3)
            if num_components > 3:
                val_4 = struct.unpack(
                    format_character, bin_data[i+2*byte_len:i+3*byte_len])[0]
                temp_data.append(val_4)
            if num_components > 4:
                print("ERROR : There is no VEC5 component")
                return

        data.append(temp_data)

    return data


def get_mat_data_from_bin(bin_data, num_components, accessor, bufferViews):

    format_character, byte_len = get_format_char_byte_len(accessor)

    data = []

    bufferViewIndex = accessor.bufferView
    #numIndices = accessor.count
    bufferView = bufferViews[bufferViewIndex]

    # it can be assumed that num components is 1
    for i in range(bufferView.byteOffset, bufferView.byteOffset+bufferView.byteLength, num_components*byte_len):

        c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15 = [
            0]*16

        c0 = struct.unpack(format_character, bin_data[i:i+byte_len])[0]
        c1 = struct.unpack(
            format_character, bin_data[i+byte_len:i+2*byte_len])[0]
        c2 = struct.unpack(
            format_character, bin_data[i+2*byte_len:i+3*byte_len])[0]
        c3 = struct.unpack(
            format_character, bin_data[i+3*byte_len:i+4*byte_len])[0]

        if num_components > 4:  # (MAT3, num_components = 9)
            c4 = struct.unpack(
                format_character, bin_data[i+4*byte_len:i+5*byte_len])[0]
            c5 = struct.unpack(
                format_character, bin_data[i+5*byte_len:i+6*byte_len])[0]
            c6 = struct.unpack(
                format_character, bin_data[i+6*byte_len:i+7*byte_len])[0]
            c7 = struct.unpack(
                format_character, bin_data[i+7*byte_len:i+8*byte_len])[0]
            c8 = struct.unpack(
                format_character, bin_data[i+8*byte_len:i+9*byte_len])[0]

            if num_components > 9:  # (MAT4, num_components = 16)
                c9 = struct.unpack(
                    format_character, bin_data[i+9*byte_len:i+10*byte_len])[0]
                c10 = struct.unpack(
                    format_character, bin_data[i+10*byte_len:i+11*byte_len])[0]
                c11 = struct.unpack(
                    format_character, bin_data[i+11*byte_len:i+12*byte_len])[0]
                c12 = struct.unpack(
                    format_character, bin_data[i+12*byte_len:i+13*byte_len])[0]
                c13 = struct.unpack(
                    format_character, bin_data[i+13*byte_len:i+14*byte_len])[0]
                c14 = struct.unpack(
                    format_character, bin_data[i+14*byte_len:i+15*byte_len])[0]
                c15 = struct.unpack(
                    format_character, bin_data[i+15*byte_len:i+16*byte_len])[0]

                if num_components > 16:
                    print("ERROR : There is no MAT5 component")
                    return

        if num_components == 4:
            matrix = np.matrix([[c0, c2],
                                [c1, c3]])
            data.append(matrix)

        if num_components == 9:
            matrix = np.matrix([[c0, c3, c6],
                                [c1, c4, c7],
                                [c2, c5, c8]])
            data.append(matrix)

        if num_components == 16:
            matrix = np.matrix([[c0, c4, c8, c12],
                                [c1, c5, c9, c13],
                                [c2, c6, c10, c14],
                                [c3, c7, c11, c15]])
            data.append(matrix)

    return data


def get_data_from_accessor(accessor, bin_data, bufferViews):

    accessor_data_type = accessor.type

    if accessor_data_type == "SCALAR":
        data = get_scalar_data_from_bin(bin_data, 1, accessor, bufferViews)

    if accessor_data_type == "VEC2":
        data = get_vec_data_from_bin(bin_data, 2, accessor, bufferViews)

    if accessor_data_type == "VEC3":
        data = get_vec_data_from_bin(bin_data, 3, accessor, bufferViews)

    if accessor_data_type == "VEC4":
        data = get_vec_data_from_bin(bin_data, 4, accessor, bufferViews)

    if accessor_data_type == "MAT2":
        data = get_mat_data_from_bin(bin_data, 4, accessor, bufferViews)

    if accessor_data_type == "MAT3":
        data = get_mat_data_from_bin(bin_data, 9, accessor, bufferViews)

    if accessor_data_type == "MAT4":
        data = get_mat_data_from_bin(bin_data, 16, accessor, bufferViews)

    return data


def load_gltf_and_bin(gltf_filename):
    gltf = GLTF.load(gltf_filename, load_file_resources=True)
    resource = gltf.resources[0]
    # get binary data from the .bin file
    bin_data = resource.data
    return gltf, bin_data


def get_accessors_bufferViews_buffers(gltf):
    accessors = gltf.model.accessors
    bufferViews = gltf.model.bufferViews
    buffers = gltf.model.buffers

    return accessors, bufferViews, buffers


def get_mesh_and_skin_nodes(gltf):
    """ returns a mesh and skin if they exist, or None otherwise"""
    skin = None
    mesh = None
    for node in gltf.model.nodes:
        if node.mesh != None:
            mesh = gltf.model.meshes[node.mesh]
        if node.skin != None:
            skin = gltf.model.skins[node.skin]

    return mesh, skin


def get_mesh_attributes_accessor_ids(mesh):
    """ returns a dictionary with the accessor_ids of all the attributes"""
    attribute_accessor_ids = {}

    attribute_accessor_ids['position_accessor_id'] = mesh.primitives[0].attributes.POSITION
    attribute_accessor_ids['normal_accessor_id'] = mesh.primitives[0].attributes.NORMAL
    attribute_accessor_ids['tangent_accessor_id'] = mesh.primitives[0].attributes.TANGENT
    attribute_accessor_ids['texcoord0_accessor_id'] = mesh.primitives[0].attributes.TEXCOORD_0
    attribute_accessor_ids['texcoord1_accessor_id'] = mesh.primitives[0].attributes.TEXCOORD_1
    attribute_accessor_ids['color_accessor_id'] = mesh.primitives[0].attributes.COLOR_0

    attribute_accessor_ids['faces_accessor_id'] = mesh.primitives[0].indices
    attribute_accessor_ids['joints_accessor_id'] = mesh.primitives[0].attributes.JOINTS_0
    attribute_accessor_ids['weights_accessor_id'] = mesh.primitives[0].attributes.WEIGHTS_0

    return attribute_accessor_ids
