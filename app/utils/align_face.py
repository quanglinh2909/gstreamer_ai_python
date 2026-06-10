"""
Face alignment using YOLO keypoints - matches MTCNN alignment exactly.
Uses the same similarity transform algorithm as MTCNN for consistent results.

Vendored into the source tree (was previously loaded at runtime from an
external /home/orangepi/Documents/AdaFace/align_face.py) so registration does
not depend on a path outside this repository.
"""
import cv2
import numpy as np
from numpy.linalg import inv, lstsq
from numpy.linalg import matrix_rank as rank


# MTCNN reference facial points for default crop_size (96, 112)
REFERENCE_FACIAL_POINTS = np.array([
    [30.29459953,  51.69630051],  # left eye
    [65.53179932,  51.50139999],  # right eye
    [48.02519989,  71.73660278],  # nose tip
    [33.54930115,  92.3655014],   # left mouth corner
    [62.72990036,  92.20410156]   # right mouth corner
], dtype=np.float32)


def get_reference_facial_points(output_size=(112, 112), default_square=True):
    """
    Get reference facial points adjusted for output size.
    Same logic as MTCNN's get_reference_facial_points.
    """
    tmp_5pts = REFERENCE_FACIAL_POINTS.copy()
    tmp_crop_size = np.array([96, 112])

    # Make the inner region a square (for 112x112)
    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff

    return tmp_5pts


def findNonreflectiveSimilarity(uv, xy):
    """
    Find non-reflective similarity transform.
    Same as MTCNN's matlab_cp2tform.findNonreflectiveSimilarity.
    """
    K = 2
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))
    y = xy[:, 1].reshape((-1, 1))

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))
    v = uv[:, 1].reshape((-1, 1))
    U = np.vstack((u, v))

    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U, rcond=None)
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([
        [sc, -ss, 0],
        [ss,  sc, 0],
        [tx,  ty, 1]
    ])

    T = inv(Tinv)
    T[:, 2] = np.array([0, 0, 1])

    return T, Tinv


def get_similarity_transform_for_cv2(src_pts, dst_pts):
    """
    Get similarity transform matrix for cv2.warpAffine.
    Same as MTCNN's get_similarity_transform_for_cv2.
    """
    trans, trans_inv = findNonreflectiveSimilarity(src_pts, dst_pts)
    cv2_trans = trans[:, 0:2].T
    return cv2_trans


def align_face(image, keypoints, target_size=(112, 112)):
    """
    Align face using facial keypoints - SAME algorithm as MTCNN.

    Args:
        image: Input image (BGR from OpenCV)
        keypoints: Facial keypoints from YOLO model (5x2 or 5x3 array)
                  Order: [left_eye, right_eye, nose, left_mouth, right_mouth]
        target_size: Output face size (width, height)

    Returns:
        Aligned face image (112x112 BGR)
    """
    # Get reference points (same as MTCNN)
    ref_pts = get_reference_facial_points(target_size, default_square=(target_size[0] == target_size[1]))

    # Extract keypoints (handle both 5x2 and 5x3 arrays)
    if len(keypoints) < 5:
        print("Not enough keypoints for face alignment")
        return None

    # Get first 5 keypoints, only x,y coordinates
    src_pts = keypoints[:5, :2].astype(np.float32)

    # Get similarity transform (SAME as MTCNN)
    tfm = get_similarity_transform_for_cv2(src_pts, ref_pts)

    # Apply transformation
    aligned_face = cv2.warpAffine(image, tfm, target_size, borderValue=0.0)

    return aligned_face
