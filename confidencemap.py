import numpy as np
import cv2
import matplotlib.pyplot as plt
import cupy as cp
import math
from numba import cuda, float32, int32

def read_pfm(file):
    with open(file, 'rb') as f:
        # Read the header
        header = f.readline().rstrip().decode('utf-8')
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        # Read the dimensions
        dim_line = f.readline().decode('utf-8')
        width, height = map(int, dim_line.split())
        
        # Read the scale (endianness)
        scale_line = f.readline().decode('utf-8')
        scale = float(scale_line.strip())
        endian = '<' if scale < 0 else '>'
        
        # Read the data
        data = np.fromfile(f, endian + 'f')
        if color:
            data = data.reshape((height, width, 3))
        else:
            data = data.reshape((height, width))

        # Flip the data vertically (as PFM format stores from bottom to top)
        data = np.flipud(data)

    return data

# @njit
def compute_confidence_map(ldisp, rdisp, lrgb, rrgb):
    h, w = ldisp.shape
    conf_map = np.zeros((h, w), dtype=np.float32)

    rdisp_shifted = np.zeros_like(ldisp, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            ld = ldisp[y, x]
            if ld > 0 and np.isfinite(ld):
                r_x = int(x - ld)
                if 0 <= r_x < w:
                    rdisp_shifted[y, x] = rdisp[y, r_x]

    consistency_mask = np.abs(ldisp - rdisp_shifted) < 1
    conf_map[consistency_mask] = 1.0

    max_conf = np.max(conf_map)
    if max_conf > 0:
        conf_map /= max_conf

    # Using color information to enhance the confidence map
    color_similarity_threshold = 30  # Adjust based on your needs

    lrgb = lrgb.astype(np.float32)
    rrgb = rrgb.astype(np.float32)

    for y in range(h):
        for x in range(w):
            lc = lrgb[y, x]
            ld = ldisp[y, x]
            if np.isfinite(ld) and ld > 0:
                r_x = int(x - ld)
                if 0 <= r_x < w:
                    rc = rrgb[y, r_x]
                    color_diff = np.linalg.norm(lc - rc)
                    if color_diff < color_similarity_threshold:
                        conf_map[y, x] = min(
                            conf_map[y, x] + (1 - color_diff / color_similarity_threshold),
                            1.0,
                        )

    return conf_map

def plot_four_figures(left_img, right_img, left_disp, right_disp):
    # Plot images
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(left_img)
    axes[0, 0].set_title('Left Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(right_img)
    axes[0, 1].set_title('Right Image')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(left_disp, cmap='gray')
    axes[1, 0].set_title('Left Disparity Map')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(right_disp, cmap='gray')
    axes[1, 1].set_title('Right Disparity Map')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

# Kernel function to shift the right disparity map
@cuda.jit
def shift_rdisp_kernel(ldisp, rdisp, rdisp_shifted):
    y, x = cuda.grid(2)
    h, w = ldisp.shape
    if y < h and x < w:
        ld = ldisp[y, x]
        if ld > 0 and math.isfinite(ld):
            r_x = int(x - ld)
            if 0 <= r_x < w:
                rdisp_shifted[y, x] = rdisp[y, r_x]

# Kernel function to compute the confidence map
@cuda.jit
def compute_conf_map_kernel(ldisp, rdisp_shifted, conf_map):
    y, x = cuda.grid(2)
    h, w = ldisp.shape
    if y < h and x < w:
        if abs(ldisp[y, x] - rdisp_shifted[y, x]) < 0.5:
            conf_map[y, x] = 1.0

# Kernel function to enhance the confidence map using color information
@cuda.jit
def enhance_conf_map_kernel(ldisp, lrgb, rrgb, conf_map, color_similarity_threshold):
    y, x = cuda.grid(2)
    h, w = ldisp.shape
    if y < h and x < w:
        lc = lrgb[y, x]
        ld = ldisp[y, x]
        if math.isfinite(ld) and ld > 0:
            r_x = int(x - ld)
            if 0 <= r_x < w:
                rc = rrgb[y, r_x]
                color_diff = abs(lc[0] - rc[0]) + abs(lc[1] - rc[1]) + abs(lc[2] - rc[2])
                if color_diff < color_similarity_threshold:
                    conf_map[y, x] = min(conf_map[y, x] + (1 - color_diff / color_similarity_threshold), 1.0)

def compute_confidence_map_cuda(ldisp, rdisp, lrgb, rrgb):
    ldisp = np.ascontiguousarray(ldisp)
    rdisp = np.ascontiguousarray(rdisp)
    lrgb = np.ascontiguousarray(lrgb)
    rrgb = np.ascontiguousarray(rrgb)

    h, w = ldisp.shape
    conf_map = np.zeros((h, w), dtype=np.float32)
    rdisp_shifted = np.zeros_like(ldisp, dtype=np.float32)

    # Define CUDA grid and block dimensions
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(w / threadsperblock[1]))
    blockspergrid_y = int(np.ceil(h / threadsperblock[0]))
    blockspergrid = (blockspergrid_y, blockspergrid_x)

    # Allocate device memory
    ldisp_device = cuda.to_device(ldisp)
    rdisp_device = cuda.to_device(rdisp)
    rdisp_shifted_device = cuda.to_device(rdisp_shifted)
    conf_map_device = cuda.to_device(conf_map)
    lrgb_device = cuda.to_device(lrgb)
    rrgb_device = cuda.to_device(rrgb)

    # Launch CUDA kernels
    shift_rdisp_kernel[blockspergrid, threadsperblock](ldisp_device, rdisp_device, rdisp_shifted_device)
    compute_conf_map_kernel[blockspergrid, threadsperblock](ldisp_device, rdisp_shifted_device, conf_map_device)
    color_similarity_threshold = 15  # Adjust based on your needs
    enhance_conf_map_kernel[blockspergrid, threadsperblock](ldisp_device, lrgb_device, rrgb_device, conf_map_device, color_similarity_threshold)

    # Copy the result back to the host
    conf_map = conf_map_device.copy_to_host()

    # Normalize the confidence map
    max_conf = np.max(conf_map)
    if max_conf > 0:
        conf_map /= max_conf

    return conf_map

if __name__ == "__main__":
    # sample_dir = "/home/william/extdisk/data/middlebury/middlebury2014/Motorcycle-perfect/"
    base_dir = "/home/william/extdisk/data/middlebury/middlebury2014"

    left_img_name = "im0.png"
    right_img_name = "im1.png"
    left_disp_name = "disp0.pfm"
    right_disp_name = "disp1.pfm"
    confimap_name = "confimap.npy"

    import os
    scenes = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    for scene in scenes:
        path = os.path.join(base_dir, scene)
        confimap_path = os.path.join(path, confimap_name)
        left_img = cv2.imread(path+"/"+left_img_name)[:,:,::-1]
        right_img = cv2.imread(path+"/"+right_img_name)[:,:,::-1]
        left_disp = read_pfm(path+"/"+left_disp_name)
        right_disp = read_pfm(path+"/"+right_disp_name)

        # plot_four_figures(left_img, right_img, left_disp, right_disp)
        confimap = compute_confidence_map_cuda(left_disp, right_disp, left_img, right_img)
        np.save(confimap_path, confimap)
        # print(confimap)
        # plt.figure()
        # plt.imshow(confimap, cmap='gray')
        # plt.show()
    print("done!")