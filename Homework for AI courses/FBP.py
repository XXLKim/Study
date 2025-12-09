import h5py
import numpy as np
from skimage.transform import iradon
import matplotlib.pyplot as plt
import os

# -----------------------------
# 配置路径（云服务器）
# -----------------------------
# 输入 hdf5 文件在云服务器上的路径
input_file = "/home/ubuntu/FBP_data/observation_test_000.hdf5"

# 输出文件夹也在云服务器
output_folder = "/home/ubuntu/FBP_data/observation_FBP"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# 1. 读取 sinogram
# -----------------------------
with h5py.File(input_file, "r") as f:
    key = list(f.keys())[0]
    sinograms = f[key][:].astype('float32')  # shape: (128, 1000, 513)

num_slices, num_angles, num_detectors = sinograms.shape
print(f"读取完成: {num_slices} slices, {num_angles} angles, {num_detectors} detectors")

# -----------------------------
# 2. 定义投影角度
# -----------------------------
theta = np.linspace(0., 180., num_angles, endpoint=False)

# -----------------------------
# 3. 批量 FBP 重建并保存
# -----------------------------
recon_slices = []
for i in range(num_slices):
    sinogram = sinograms[i]
    recon = iradon(sinogram.T, theta=theta, circle=True)
    recon_slices.append(recon)

recon_slices = np.stack(recon_slices, axis=0)  # shape: (num_slices, H, W)

# 保存为 numpy 文件
output_file = os.path.join(output_folder, "observation_test_000_FBP.npy")
np.save(output_file, recon_slices)
print(f"全部重建 slice 保存完成，shape = {recon_slices.shape}, 文件路径: {output_file}")

# -----------------------------
# 4. 挑选一张 slice 可视化并单独保存
# -----------------------------
slice_idx = 0  # 可以改为想要展示的 slice
slice_img = recon_slices[slice_idx]

plt.figure(figsize=(6,6))
plt.imshow(slice_img, cmap='gray')
plt.axis('off')
plt.title(f"FBP Low-dose CT Slice {slice_idx}")

# 保存为图片
slice_filename = os.path.join(output_folder, f"slice_{slice_idx}.png")
plt.savefig(slice_filename, bbox_inches='tight', pad_inches=0)
plt.close()
print(f"单张 slice 保存为图片: {slice_filename}")

