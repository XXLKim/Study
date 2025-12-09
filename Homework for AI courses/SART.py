import h5py
import numpy as np
import tomopy
import matplotlib.pyplot as plt
import os

# -----------------------------
# 配置路径（云服务器）
# -----------------------------
input_file = "/home/ubuntu/FBP_data/observation_test_000.hdf5"
output_folder = "/home/ubuntu/FBP_data/observation_FBP_SART_AS"  # ASTRA SART 输出文件夹
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
# 2. 定义投影角度（radians）
# -----------------------------
theta = np.linspace(0., np.pi, num_angles, endpoint=False)

# -----------------------------
# 3. 批量 ASTRA SART 重建并保存（带进度）
# -----------------------------
recon_slices = []
num_iter = 10  # 每个 slice 的迭代次数，可以根据需要调节

for i in range(num_slices):
    # tomopy 要求输入 shape = (1, num_angles, num_detectors)
    sinogram = sinograms[i:i+1]

    # ASTRA 后端 SART 重建
    recon = tomopy.recon(sinogram, theta, algorithm='sart', num_iter=num_iter, backend='astra')
    
    recon_slices.append(recon[0])

    # 输出进度
    print(f"已完成 {i+1}/{num_slices} slices")

recon_slices = np.stack(recon_slices, axis=0)  # shape = (num_slices, H, W)

# -----------------------------
# 4. 保存全部 slice 为 numpy 文件
# -----------------------------
output_file = os.path.join(output_folder, "observation_test_000_SART_AS.npy")
np.save(output_file, recon_slices)
print(f"全部重建 slice 保存完成, shape = {recon_slices.shape}, 文件路径: {output_file}")

# -----------------------------
# 5. 挑选一张 slice 可视化并保存
# -----------------------------
slice_idx = 0  # 可修改展示其它 slice
slice_img = recon_slices[slice_idx]

plt.figure(figsize=(6,6))
plt.imshow(slice_img, cmap='gray')
plt.axis('off')
plt.title(f"ASTRA SART Low-dose CT Slice {slice_idx}")

# 保存图片
slice_filename = os.path.join(output_folder, f"slice_{slice_idx}.png")
plt.savefig(slice_filename, bbox_inches='tight', pad_inches=0)
plt.close()
print(f"单张 slice 保存为图片: {slice_filename}")
