import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# 选择 GPU / CPU
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# --------------------------------------------------
# MNIST 数据集
# --------------------------------------------------
mnist = datasets.MNIST(
    root='./others/',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
)

dataloader = DataLoader(
    dataset=mnist,
    batch_size=64,
    shuffle=True,
    num_workers=4,          # 多线程加载数据，提高速度
    pin_memory=True         # host→GPU 拷贝更快
)

# --------------------------------------------------
# 生成图片，保存而不是 show（不会阻塞训练）
# --------------------------------------------------
def gen_img_plot(model, epoch, text_input):
    model.eval()
    with torch.no_grad():
        prediction = np.squeeze(model(text_input.to(device)).detach().cpu().numpy()[:16])

    plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow((prediction[i] + 1) / 2, cmap='gray')
        plt.axis('off')

    # 保存图片
    os.makedirs("generated_images", exist_ok=True)
    plt.savefig(f"generated_images/epoch_{epoch:03d}.png")
    plt.close()


# --------------------------------------------------
# 生成器
# --------------------------------------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.mean = nn.Sequential(
            *block(100, 256, normalize=False),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        imgs = self.mean(x)
        imgs = imgs.view(-1, 1, 28, 28)
        return imgs


# --------------------------------------------------
# 判别器
# --------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.mean = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        img = self.mean(x)
        return img


# --------------------------------------------------
# 实例化模型（移到 GPU）
# --------------------------------------------------
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 优化器
G_optim = torch.optim.Adam(generator.parameters(), lr=0.0001)
D_optim = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

criterion = torch.nn.BCELoss()

# --------------------------------------------------
# 训练
# --------------------------------------------------
epoch_num = 100
G_loss_save = []
D_loss_save = []

for epoch in range(epoch_num):
    G_epoch_loss = 0
    D_epoch_loss = 0
    count = len(dataloader)

    generator.train()
    discriminator.train()

    for i, (img, _) in enumerate(dataloader):
        img = img.to(device, non_blocking=True)  # 开启 GPU 拷贝优化
        size = img.size(0)

        # -----------------------------
        # 训练 Discriminator
        # -----------------------------
        fake_img = torch.randn(size, 100, device=device)   # 在 GPU 上生成噪声

        output_fake = generator(fake_img)
        fake_score = discriminator(output_fake.detach())
        D_fake_loss = criterion(fake_score, torch.zeros_like(fake_score, device=device))

        real_score = discriminator(img)
        D_real_loss = criterion(real_score, torch.ones_like(real_score, device=device))

        D_loss = D_fake_loss + D_real_loss
        D_optim.zero_grad()
        D_loss.backward()
        D_optim.step()

        # -----------------------------
        # 训练 Generator
        # -----------------------------
        fake_score_forG = discriminator(output_fake)
        G_fake_loss = criterion(fake_score_forG, torch.ones_like(fake_score_forG, device=device))

        G_optim.zero_grad()
        G_fake_loss.backward()
        G_optim.step()

        with torch.no_grad():
            G_epoch_loss += G_fake_loss
            D_epoch_loss += D_loss

    with torch.no_grad():
        G_epoch_loss /= count
        D_epoch_loss /= count
        G_loss_save.append(G_epoch_loss.item())
        D_loss_save.append(D_epoch_loss.item())

        print(f"Epoch [{epoch+1}/{epoch_num}]  G_loss: {G_epoch_loss:.4f}  D_loss: {D_epoch_loss:.4f}")

        # 每个 epoch 生成图片保存
        text_input = torch.randn(64, 100, device=device)
        gen_img_plot(generator, epoch + 1, text_input)


# --------------------------------------------------
# 绘制 loss 曲线并保存
# --------------------------------------------------
plt.figure()
x = [epoch + 1 for epoch in range(epoch_num)]
plt.plot(x, G_loss_save, 'r')
plt.plot(x, D_loss_save, 'b')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['G_loss','D_loss'])
plt.savefig("loss_curve.png")
plt.close()
