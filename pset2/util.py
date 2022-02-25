import cv2
import scipy.io as sio
import numpy as np
import numpy.matlib


eps = 2 ** -52


# flatten to Nx1
def flatten(x):
    return x.reshape([-1, 1])


def load_images_and_params(scale):
    # define projector-camera intrinsic parameters
    # these are from http://vision.middlebury.edu/stereo/data/scenes2014/
    f0 = 3979.911  # focal length in (pixels)
    d = 193.001  # baseline in (mm)
    # doffs0 = 124.343  # x-difference of principal points (pixels)
    # (just used for overlaying viz on rgb camera
    #  since the data came this way; elsewhere,
    #  we treat doffs as zero)
    cx_c, cy_c = (1244.772, 1019.507)  # principal point coordinates for the camera
    cx_p, cy_p = (1369.115, 1019.507)  # principal point coordinates for the projector

    # load images
    Z_p_img = sio.loadmat("./Motorcycle-perfect/z_projector_view.mat")["z_p_img"]
    Z_c_img = sio.loadmat("./Motorcycle-perfect/z_camera_view.mat")["z_c_img"]
    rgb_c_img = cv2.imread("./Motorcycle-perfect/rgb_camera_view.png")

    # blur before downsample for anti-aliasing
    ksize = int(1.0 / scale) * 2 + 1
    Z_p_img = cv2.GaussianBlur(Z_p_img, (ksize, ksize), 0)
    Z_c_img = cv2.GaussianBlur(Z_c_img, (ksize, ksize), 0)
    rgb_c_img = cv2.GaussianBlur(rgb_c_img, (ksize, ksize), 0)

    # downsample to make things faster
    Z_p_img = cv2.resize(Z_p_img, (0, 0), fx=scale, fy=scale)
    Z_c_img = cv2.resize(Z_c_img, (0, 0), fx=scale, fy=scale)
    rgb_c_img = cv2.resize(rgb_c_img, (0, 0), fx=scale, fy=scale)
    f = f0 * scale
    # doffs = doffs0 * scale
    cx_c = cx_c * scale
    cy_c = cy_c * scale
    cx_p = cx_p * scale
    cy_p = cy_p * scale
    img_size = Z_p_img.shape

    return Z_p_img, Z_c_img, rgb_c_img, f, d, cx_c, cy_c, cx_p, cy_p, img_size


# assumes x is a NxD dimensional stack of N points with D coordinates
def to_homogeneous(x):
    return np.hstack([x, np.ones([x.shape[0], 1])])


# assumes x is a NxD dimensional stack of N points with D coordinates
def to_heterogeneous(x):
    return x[:, :-1] / x[:, [-1]]


# uv_p is Nx2
# z is Nx1
def transform_uv_p_to_uv_c(uv_p, Z, T):

    x = np.hstack([uv_p, 1.0 / (Z + eps)])  # eps is used to avoid dividing by zero
    x = to_homogeneous(x)  # to homogeneous
    y = (T.dot(x.T)).T  # apply coordinate transform
    uv_c = to_heterogeneous(y)  # back to heterogeneous

    return uv_c


# uv_p is Nx2
# z is Nx1
def transform_uv_p_to_XYZ_p(uv_p, Z, T):

    x = np.hstack([uv_p, 1.0 / (Z + eps)])  # eps is used to avoid dividing by zero
    x = to_homogeneous(x)  # to homogeneous
    y = (T.dot(x.T)).T  # apply coordinate transform
    XYZ_p = to_heterogeneous(y)  # back to heterogeneous

    return XYZ_p


# uv_p is Nx2
# z is Nx1
def transform_uv_p_to_XYZ_c(uv_p, Z, T):

    x = np.hstack([uv_p, 1.0 / (Z + eps)])  # eps is used to avoid dividing by zero
    x = to_homogeneous(x)  # to homogeneous
    y = (T.dot(x.T)).T  # apply coordinate transform
    XYZ_c = to_heterogeneous(y)  # back to heterogeneous

    return XYZ_c


## render with occlusion
# if a single xy_c coord is projected to twice, remove the one that is
# farther away in z (it is occluded)


def render(xy_c, xy_p, Z_c, L_p_img):

    xy_c = np.round(xy_c).astype(np.int)
    L_c_img = np.zeros_like(L_p_img) * np.nan
    depth_buffer = np.zeros_like(L_c_img) * np.nan

    for i in range(xy_p.shape[0]):  # iterate through every projected light ray
        if (
            xy_c[i, 0] >= 0
            and xy_c[i, 0] < depth_buffer.shape[1]
            and xy_c[i, 1] >= 0
            and xy_c[i, 1] < depth_buffer.shape[0]
        ):
            if (
                np.isnan(depth_buffer[xy_c[i, 1], xy_c[i, 0]])
                or depth_buffer[xy_c[i, 1], xy_c[i, 0]] > Z_c[i]
            ):

                L_c_img[xy_c[i, 1], xy_c[i, 0]] = L_p_img[xy_p[i, 1], xy_p[i, 0]]
                depth_buffer[xy_c[i, 1], xy_c[i, 0]] = Z_c[i]

    return L_c_img


def estimate_normals_from_z(Z_img):

    Z_blurred = cv2.GaussianBlur(Z_img, (3, 3), 1)
    n_x = cv2.filter2D(Z_img, -1, np.array([[-1, 0, 1]]) / 2.0)
    n_y = cv2.filter2D(Z_img, -1, np.array([[1], [0], [-1]]) / 2.0)
    n_z = np.ones_like(n_x)
    Z = np.sqrt(n_x ** 2 + n_y ** 2 + n_z ** 2)

    n_x = n_x / Z
    n_y = n_y / Z
    n_z = n_z / Z
    n_img = np.dstack([n_x, n_y, n_z])

    return n_img


# render with occlusion and Lambertian reflectance
def render_Lambertian(xy_c, xy_p, Z_c, XYZ_p, L_p_img):

    Z_p = XYZ_p[:, 2]

    # estimate normal map
    Z_p_img = np.reshape(Z_p, [L_p_img.shape[0], L_p_img.shape[1]])
    n_p_img = estimate_normals_from_z(Z_p_img)

    # calculate angle of each projected light ray
    L_angle_p_img = np.reshape(XYZ_p, [L_p_img.shape[0], L_p_img.shape[1], 3])
    L_angle_p_img = (
        L_angle_p_img / np.sqrt(np.sum(L_angle_p_img ** 2, 2))[:, :, np.newaxis]
    )

    ## Display surface normal
    # import matplotlib.pyplot as plt

    # plt.subplot(121)
    # plt.imshow(n_p_img / 2 + 0.5)
    # plt.subplot(122)
    # plt.imshow(L_angle_p_img / 2 + 0.5)
    # plt.show()

    xy_c = np.round(xy_c).astype(np.int)
    L_c_img = np.zeros_like(L_p_img) * np.nan
    depth_buffer = np.zeros((L_c_img.shape[0], L_c_img.shape[1])) * np.nan

    for i in range(xy_p.shape[0]):  # iterate through every projected light ray
        if (
            xy_c[i, 0] >= 0
            and xy_c[i, 0] < depth_buffer.shape[1]
            and xy_c[i, 1] >= 0
            and xy_c[i, 1] < depth_buffer.shape[0]
        ):
            if (
                np.isnan(depth_buffer[xy_c[i, 1], xy_c[i, 0]])
                or depth_buffer[xy_c[i, 1], xy_c[i, 0]] > Z_c[i]
            ):

                NdotL = flatten(n_p_img[xy_p[i, 1], xy_p[i, 0], :]).T.dot(
                    flatten(L_angle_p_img[xy_p[i, 1], xy_p[i, 0], :])
                )
                NdotL = np.maximum(NdotL, 0)

                for c in range(L_c_img.shape[2]):
                    L_c_img[xy_c[i, 1], xy_c[i, 0], c] = (
                        L_p_img[xy_p[i, 1], xy_p[i, 0], c] * NdotL
                    )

                depth_buffer[xy_c[i, 1], xy_c[i, 0]] = Z_c[i]

    return L_c_img
