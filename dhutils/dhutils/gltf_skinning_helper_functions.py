import numpy as np
from scipy.spatial.transform import Rotation as R

def construct_transform_matrix(translation, rotation, scale):
    """ constructions the combined transformation matrix, 
        given translation, rotation and scale seperately """

    Tr = np.matrix([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])
    Ro = np.matrix([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])
    Sc = np.matrix([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])
        
    if translation:
        Tr[0,3] = translation[0]
        Tr[1,3] = translation[1]
        Tr[2,3] = translation[2]
        # print("T :", Tr)
            
    if rotation:
        r = rotation
        Ro = R.from_quat(r)
        Ro = Ro.as_matrix()
        Ro = np.r_[Ro, [[0, 0, 0]]]
        Ro = np.c_[Ro, [0, 0, 0, 1]]
        # print("R :", Ro)
            
    if scale:
        Sc[0,0] = scale[0]
        Sc[1,1] = scale[1]
        Sc[2,2] = scale[2]

    M = Tr @  Ro @ Sc
        
    return M

def calculate_global_joint_transforms(root, gltf):
    """ the transformation matrix for each joint is 
        given in the local space, this function multiplies
        transformations in the hierarchy and assigns 
        a global transformation matrix to every joint """

    if root.children:
        for child in root.children:
            childNode = gltf[child] 
            childLocalMat = construct_transform_matrix(childNode.translation, \
                                             childNode.rotation, childNode.scale)
            # assign global_joint_transform property to every joint
            childNode.global_joint_transform = root.global_joint_transform @ childLocalMat 
            # recursive call
            calculate_global_joint_transforms(childNode, gltf)
    else:
        return
