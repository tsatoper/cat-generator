
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torchvision.models import vgg16

from PIL import Image
import os
import json
from datetime import datetime
import argparse


def parse_and_save_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--beta', type=float, default=4.0)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--job_id', type=str, default=datetime.now().strftime('%Y%m%d_%H%M'))
    
    args = parser.parse_args()

    # Save config
    config_dir = os.path.join(args.save_dir)
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "config.json")

    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    print("Saved config to")
    print(config_path)
    config = vars(args)
    print(
        f"JOB_ID={config['job_id']}\n"
        f"python vaetrain.py \\\n"
        f"    --batch_size {config['batch_size']} \\\n"
        f"    --epochs {config['epochs']} \\\n"
        f"    --lr {config['lr']:.0e} \\\n"
        f"    --latent_dim {config['latent_dim']} \\\n"
        f"    --beta {config['beta']} \\\n"
        f"    --data_path {config['data_path']} \\\n"
        f"    --save_dir /hb/home/tsatoper/cat-generator/results/${{JOB_ID}} \\\n"
        f"    --resume {config['resume']} \\\n"
        f"    --job_id ${{JOB_ID}}"
    )
    
    if args.resume == "null":
        args.resume = None

    return args

def load_cluster_checkpoint(checkpoint_path, model, optimizer=None):
    """Load checkpoint for resuming training"""
    checkpoint = torch.load(
        checkpoint_path, 
        map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return

class CatDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
        return image

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(), ResidualBlock(32),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(), ResidualBlock(64),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(), ResidualBlock(128),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(), ResidualBlock(256)
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(), ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(), ResidualBlock(64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(), ResidualBlock(32),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = self.fc_decode(z).view(-1, 256, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class MultiLayerPerceptualLoss(nn.Module):
    def __init__(self, layers=("relu1_2", "relu2_2", "relu3_3", "relu4_3"), resize=True):
        super().__init__()
        vgg = vgg16(weights="IMAGENET1K_V1").features
        layer_map = {"relu1_2": 4, "relu2_2": 9, "relu3_3": 16, "relu4_3": 23}
        self.vgg_slices = nn.ModuleList()
        prev_idx = 0
        for name in layers:
            idx = layer_map[name]
            self.vgg_slices.append(nn.Sequential(*vgg[prev_idx:idx]))
            prev_idx = idx
        for p in self.parameters():
            p.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.upsample = nn.Upsample((224, 224), mode="bilinear", align_corners=False) if resize else nn.Identity()

    def normalize(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, x, y):
        x, y = self.normalize(self.upsample(x)), self.normalize(self.upsample(y))
        loss = 0
        for slice in self.vgg_slices:
            x, y = slice(x), slice(y)
            loss += F.mse_loss(x, y)
        return loss

def vae_loss(recon_x, x, mu, logvar, beta, perceptual_loss_fn):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon_loss + beta * kl

def main():
    args = parse_and_save_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Model
    model = VAE(latent_dim=args.latent_dim).to(device)
    perceptual_loss_fn = MultiLayerPerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)


    # Load Data
    transform = transforms.Compose([
        transforms.Resize(64), 
        transforms.CenterCrop(64), 
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor()
    ])
    dataset = CatDataset(args.data_path, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Resume
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        load_cluster_checkpoint(args.resume, model, optimizer)

    # Create directories
    os.makedirs(args.save_dir , exist_ok=True)
    models_path = os.path.join(args.save_dir, "models")
    os.makedirs(models_path, exist_ok=True)

    # Train
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss = vae_loss(recon, data, mu, logvar, args.beta, perceptual_loss_fn)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader.dataset):.4f}")

        if epoch%10 == 0 or epoch == 1:
            print(f"Epoch {epoch}, LR: {scheduler.get_last_lr()[0]}")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
            },   os.path.join(args.save_dir, "models", f"epoch_{epoch}.pth"))

    # Save Latest
    latest_path = os.path.join(args.save_dir, "latest.pth")
    torch.save({
                "epoch": args.epochs,
                "model_state_dict": model.state_dict(),
            }, latest_path)
    print('Saved latest model in:')
    print(latest_path)


    # Generate and Save sample
    model.eval()
    with torch.no_grad():
        z = torch.randn(16, args.latent_dim).to(device) 
        samples = model.decode(z)
    grid = make_grid(samples, nrow=4, normalize=True, padding=2)
    sample_path = os.path.join(args.save_dir, "samples.png")
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    save_image(grid, sample_path)
    print("Saved sample images to")
    print(sample_path)

if __name__ == "__main__":
    main()
