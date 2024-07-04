import numpy as np
import mediapipe as mp
import json

mp_hands = mp.solutions.hands
hand_connections = mp_hands.HAND_CONNECTIONS
print(hand_connections)

# we can use the hierarchy of the hand connections on the internet
connections_hierarchy = {
            0: [1, 5, 17],
            1: [2],
            2: [3],
            3: [4],
            4: [],
            5: [6],
            6: [7],
            7: [8],
            8: [],
            9: [10],
            10: [11],
            11: [12],
            12: [],
            13: [14],
            14: [15],
            15: [16],
            16: [],
            17: [18],
            18: [19],
            19: [20],
            20: []

}

def get_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = b - a
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return float(np.degrees(angle))


def get_joint_angles(landmarks):
    angles = {}

    # the angle is calculated wrt the parent (A) -> child (B) -> grandchild (C) connection. 
    # A, B, C are the points specified in the angle formula in the provided ggogle doc

    for parent, children in connections_hierarchy.items():
        if len(children) > 0:
            for child in children:
                for grandchild in connections_hierarchy[child]:
                    angles[child] = get_angle(landmarks[parent], landmarks[child], landmarks[grandchild])

    return angles


all_landmarks = np.load('data/hand_landmarks.npy', allow_pickle=True)


dict_angles = {}        

# save the angles to a json file
with open('data/hand_angles.json', 'w') as f:
    json.dump(dict_angles, f, indent=4)


# rotation matrices in each joint's local coordinate system
# The rotation matrix is calculated this way:
# 1. Find the normalized vector from the parent to the child joint and set it as z-axis (v_z)
# 2. Find the normalized vector from the child to the grandchild joint (v_child_grandchild)
# 3. Find the cross product of v_z and v_child_grandchild to get the x-axis (v_x)
# 4. Find the cross product of v_x and v_z to get the y-axis (v_y)
# 5. Create a 3x3 matrix with v_x, v_y, and v_z as columns

def get_rotation_matrix(parent, child, grandchild, landmarks):

    # joint angle
    theta = get_angle(landmarks[parent], landmarks[child], landmarks[grandchild])
    
    vec_parent_to_child = np.array([
        landmarks[child][0] - landmarks[parent][0],
        landmarks[child][1] - landmarks[parent][1],
        landmarks[child][2] - landmarks[parent][2]
    ])
    vec_z = vec_parent_to_child / np.linalg.norm(vec_parent_to_child)

    vec_child_to_grandchild = np.array([
        landmarks[grandchild][0] - landmarks[child][0],
        landmarks[grandchild][1] - landmarks[child][1],
        landmarks[grandchild][2] - landmarks[child][2]
    ])

    #####
    # from here on I am only calculating the cross product with theta as the question asks if we can use the angle to find the rotation matrix.
    # We can also simply use the cross product which would implicitly use the angle anyway.

    normalized_vec_child_to_grandchild = vec_child_to_grandchild / np.linalg.norm(vec_child_to_grandchild)

    # unit vector perpendicular to the plane defined by vec_z and vec_child_to_grandchild
    n = np.cross(vec_z, normalized_vec_child_to_grandchild)
    vec_y = np.linalg.norm(vec_z) * np.linalg.norm(vec_child_to_grandchild) * np.sin(np.radians(theta)) * n
    #####

    #####
    # we could simply do this instead of the above 3 lines
    # vec_y = np.cross(vec_z, normalized_vec_child_to_grandchild)
    #####

    vec_y /= np.linalg.norm(vec_y)

    vec_x = np.cross(vec_y, vec_z)
    vec_x /= np.linalg.norm(vec_x)

    # Create rotation matrix with vec_x, vec_y, vec_z as column vectors
    rotation_matrix = np.array([vec_x, vec_y, vec_z]).T
    return rotation_matrix
                


def get_local_rotation_matrices(landmarks):
    dict_local_rotation_matrices = {}
    for parent, children in connections_hierarchy.items():
        if len(children) > 0:
            for child in children:
                for grandchild in connections_hierarchy[child]:
                    dict_local_rotation_matrices[child] = get_rotation_matrix(parent, child, grandchild, landmarks).tolist()
    
    return dict_local_rotation_matrices



# Rotation matrices in global coordinate system are calculated this way:
# 1. Defined the global coordinate system as a 3x3 identity matrix defined at wrist joint (0)
# 2. For each joint, traverse the hierarchy from parent to child and keep multiplying the rotation matrices in the local coordinate system 
#    to get the rotation matrix in the global coordinate system.
# 3. For example, to get the rotation matrix at joint 3, we multiply the rotation matrix at joint 0 with the local rotation matrices at joint 1,2, and 3.

def get_global_rotation_matrices(landmarks):
    global_rotation_matrices = {0: np.eye(3)}  # Start with the wrist joint as the global reference
    
    def traverse_hierarchy(parent):
        for child in connections_hierarchy.get(parent, []):
            if child in connections_hierarchy:
                for grandchild in connections_hierarchy[child]:
                    
                    local_rotation_matrix = get_rotation_matrix(parent, child, grandchild, landmarks)
                    
                    global_rotation_matrix = np.array(global_rotation_matrices[parent]).dot(local_rotation_matrix)
                    
                    global_rotation_matrices[child] = global_rotation_matrix.tolist()
                    traverse_hierarchy(child)
    
    traverse_hierarchy(0)
    global_rotation_matrices[0] = global_rotation_matrices[0].tolist()
    return global_rotation_matrices


dict_angles = {}
dict_local_rotation_matrices = {}
dict_global_rotation_matrices = {}

for frame, frame_landmarks in enumerate(all_landmarks):
    dict_angles[f"Frame {frame}"] = {}
    dict_local_rotation_matrices[f"Frame {frame}"] = {}
    dict_global_rotation_matrices[f"Frame {frame}"] = {}

    if len(frame_landmarks) > 0:    # check if there are landmarks in the frame
        dict_angles[f"Frame {frame}"] = get_joint_angles(frame_landmarks[0])
        dict_local_rotation_matrices[f"Frame {frame}"] = get_local_rotation_matrices(frame_landmarks[0])
        dict_global_rotation_matrices[f"Frame {frame}"] = get_global_rotation_matrices(frame_landmarks[0])


with open('data/joint_angles.json', 'w') as f:
    json.dump(dict_angles, f, indent=4)


# save the rotation matrices to a json file
with open('data/hand_local_rotation_matrices.json', 'w') as f:
    json.dump(dict_local_rotation_matrices, f, indent=4)

with open('data/hand_global_rotation_matrices.json', 'w') as f:
    json.dump(dict_global_rotation_matrices, f, indent=4)



    





