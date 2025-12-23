# Copyright © 2025107441785
# This code is protected by copyright law.

# Regarding DINOv2
# We hereby declare that we have only modified line 58 of the source code in `dinov2/hub/backbone.py`, and the modified content is as follows:
# state_dict = torch.hub.load_state_dict_from_url(url=url,model_dir="/NSFF-main/dinov2/weight", map_location="cpu")

# Patent Declaration
# Our method is patented (2025107441785) and protected by copyright law. This patent primarily targets software or computer devices developed based on our method, and its content is completely independent of DINOv2.
# This patent covers only one method for generating image detection. While this method is inspired by DINOv2, it does not include the DINOv2 algorithm, technology, or implementation details.

# License Notice
# Our license (GPL-3.0) applies only to our own method and does not involve the DINOv2 repository.
# You are free to:
# View, download, and use our code for personal study, research, or evaluation.
# Modify the source code and distribute your modified versions, but must retain this statement.

# Without express written permission, you may not:
# Use our code for any commercial purpose, including but not limited to sale, rental, or provision as part of a commercial product.
# Provide technical support or services based on our code to third parties in any form for compensation.

# This code is provided "as is," and the authors assume no responsibility.

import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import cv2
from PIL import Image, ImageFilter
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import random
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, brier_score_loss
from dinov2.hub.backbones import dinov2_vitl14_reg
from AIGIDetection_custom_preprocessing import NFE_GHPF

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global random seed set to {seed}")

# Random seed (42, 2025, 3407)
set_global_seed(42)


def random_data_augmentation(image):
    if random.random() < 0.2:
        sigma = random.uniform(0.1, 3.0)
        image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    if random.random() < 0.2:
        image_array = np.array(image)
        noise = np.random.normal(0, 3, image_array.shape)
        noisy_image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(noisy_image_array)
    return image


def random_postprocessing_perturbation(image, mode=None):
    if mode in ["all", "jpeg"] and random.random() < 0.3:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=75)
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")

    if mode in ["all", "noise"] and random.random() < 0.3:
        image_array = np.array(image, dtype=np.float32)
        noise = np.random.normal(0, 20, image_array.shape)
        noisy_image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(noisy_image_array)

    if mode in ["all", "blur"] and random.random() < 0.3:
        image = image.filter(ImageFilter.GaussianBlur(radius=1.5))

    if mode in ["all", "denoise"] and random.random() < 0.3:
        img_np = np.array(image)
        den = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)
        image = Image.fromarray(den)

    if mode in ["all", "resample"] and random.random() < 0.3:
        resample_methods = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
        method = random.choice(resample_methods)
        w, h = image.size
        image = image.resize((w, h), resample=method)

    if mode in ["all", "resize256"] and random.random() < 0.3:
        image = image.resize((256, 256), Image.BICUBIC) 

    if mode in ["all", "resize512"] and random.random() < 0.3:
        image = image.resize((512, 512), Image.BICUBIC)

    if mode in ["all", "resize1024"] and random.random() < 0.3:
        image = image.resize((1024, 1024), Image.BICUBIC)

    return image


def dynamic_resize(img):
    MAX_SIZE = 980  # Please set it to a multiple of 14.
    w, h = img.size
    original_ratio = w / h

    if w > h:
        new_w = min(w, MAX_SIZE)
        new_w = (new_w // 14) * 14
        new_h = int(new_w / original_ratio)
        new_h = (new_h // 14) * 14
        new_h = min(new_h, MAX_SIZE)
    else:
        new_h = min(h, MAX_SIZE)
        new_h = (new_h // 14) * 14
        new_w = int(new_h * original_ratio)
        new_w = (new_w // 14) * 14
        new_w = min(new_w, MAX_SIZE)

    new_w = min(new_w, w)
    new_h = min(new_h, h)

    new_w = (new_w // 14) * 14
    new_h = (new_h // 14) * 14
    return img.resize((new_w, new_h), Image.BICUBIC)


# Image preprocessing
high_transform = transforms.Compose([
    transforms.Lambda(dynamic_resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

low_transform = transforms.Compose([
    transforms.Lambda(NFE_GHPF),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# data loader
class DualBranchDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.split = split
        self.paths = []
        self.labels = []
        self.models = []

        if split == "train":
            for label_dir, label in [("0_real", 0), ("1_fake", 1)]:
                dir_path = os.path.join(root_dir, label_dir)
                if os.path.exists(dir_path):
                    for fn in os.listdir(dir_path):
                        self.paths.append(os.path.join(dir_path, fn))
                        self.labels.append(label)
                        self.models.append("train")

        elif split == "test":
            for model_dir in os.listdir(root_dir):
                model_path = os.path.join(root_dir, model_dir)
                if os.path.isdir(model_path):
                    for label_dir, label in [("0_real", 0), ("1_fake", 1)]:
                        label_path = os.path.join(model_path, label_dir)
                        if os.path.exists(label_path):
                            for fn in os.listdir(label_path):
                                self.paths.append(os.path.join(label_path, fn))
                                self.labels.append(label)
                                self.models.append(model_dir)

        else:
            raise ValueError("split should be 'train' or 'test'")

        print(f"Loaded {len(self.paths)} samples from '{split}' split.")
        print(f"Labels distribution: {np.unique(self.labels, return_counts=True)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        pil = Image.open(self.paths[idx]).convert("RGB")
        if self.split == "train":
            pil = random_data_augmentation(pil)

        elif self.split == "test":
            # perturbation_mode = "all"
            # perturbation_mode = "jpeg"
            # perturbation_mode = "noise"
            # perturbation_mode = "blur"
            # perturbation_mode = "denoise"
            # perturbation_mode = "resample"
            # perturbation_mode = "resize256"
            # perturbation_mode = "resize512"
            # perturbation_mode = "resize1024"
            perturbation_mode = None
            pil = random_postprocessing_perturbation(pil, mode=perturbation_mode)

        high = high_transform(pil)
        low = low_transform(pil)
        return high, low, self.labels[idx], self.models[idx]


# DINOv2 backbone (SFE)
class DINOv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = dinov2_vitl14_reg(pretrained=True)
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.backbone.forward_features(x)
        tokens = out["x_prenorm"]
        return tokens


def extract_features_and_fuse(high_imgs, low_imgs, model):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        with torch.no_grad():
            all_tokens = model(high_imgs)
            cls_token = all_tokens[:, :1, :]
            patch_tokens = all_tokens[:, 1:-4, :]
            mean_patch_tokens = torch.mean(patch_tokens, dim=1, keepdim=True)
            selected_tokens = torch.cat([cls_token, mean_patch_tokens], dim=1)

        B, C, H, W = low_imgs.shape
        blocks = low_imgs.view(B, C, 12, 32, 12, 32) # n = 12 * 12 = 144, M = 32
        blocks = blocks.permute(0, 1, 3, 5, 2, 4)
        avg_blocks = torch.mean(blocks, dim=[-2, -1])
        low_token = avg_blocks.view(B, C, -1)  # [B, 3, 1024]
        fused = torch.cat([selected_tokens, low_token], dim=1)  # [B, 5, 1024]
        fused_normed = F.normalize(fused, p=2, dim=-1)
        return fused_normed


# linear Classifier
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1),
        )
    def forward(self, x):
        return self.fc(x).squeeze(1)


# train
def train_classifier(train_loader, model, input_dim, device, num_epochs=15):
    model.eval()
    train_features, train_labels = [], []
    for high, low, label, _ in train_loader:
        high, low = high.to(device), low.to(device)
        with torch.no_grad():
            fused = extract_features_and_fuse(high, low, model)
        train_features.append(fused.view(fused.size(0), -1).cpu().numpy())
        train_labels.append(label.numpy())

    train_features = np.concatenate(train_features)
    train_labels = np.concatenate(train_labels)

    classifier = SimpleClassifier(input_dim).to(device)
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.005,
        amsgrad=False
    )
    criterion = nn.BCEWithLogitsLoss()
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_features, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(num_epochs):
        classifier.train()
        total_loss, correct = 0.0, 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()

        avg_loss = total_loss / len(dataset)
        acc = correct / len(dataset)
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    return classifier


# Compute Expected Calibration Error (ECE)
def compute_ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if np.any(mask):
            bin_acc = labels[mask].mean()
            bin_conf = probs[mask].mean()
            ece += np.abs(bin_conf - bin_acc) * mask.sum() / len(probs)
    return ece


# test
def test_classifier(classifier, test_loader, model, device):
    model.eval()
    test_features, test_labels, test_models = [], [], []

    for batch in test_loader:
        high, low, label, model_name = batch
        high, low = high.to(device), low.to(device)
        with torch.no_grad():
            fused = extract_features_and_fuse(high, low, model)
        test_features.append(fused.view(fused.size(0), -1).cpu().numpy())
        test_labels.append(label.numpy())
        test_models.extend(model_name)

    test_features = np.concatenate(test_features)
    test_labels = np.concatenate(test_labels)
    test_models = np.array(test_models)

    # eval
    classifier.eval()
    with torch.no_grad():
        logits = classifier(torch.tensor(test_features, dtype=torch.float32).to(device))
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)

    # result
    results = {}
    per_model_metrics = []
    for model_name in np.unique(test_models):
        mask = test_models == model_name
        model_probs = probs[mask]
        model_labels = test_labels[mask]
        model_preds = preds[mask]
        tp = np.sum((model_preds == 1) & (model_labels == 1))
        tn = np.sum((model_preds == 0) & (model_labels == 0))
        fp = np.sum((model_preds == 1) & (model_labels == 0))
        fn = np.sum((model_preds == 0) & (model_labels == 1))
        roc_auc = roc_auc_score(model_labels, model_probs)
        ece = compute_ece(model_probs, model_labels)
        brier = brier_score_loss(model_labels, model_probs)

        metrics = {
            "accuracy": accuracy_score(model_labels, model_preds),
            "ap": average_precision_score(model_labels, model_probs),
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "roc_auc": roc_auc,
            "ece": ece,
            "brier": brier
        }
        results[model_name] = metrics
        per_model_metrics.append(metrics)

    # Avg result
    avg_metrics = {}
    for key in per_model_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in per_model_metrics])

    results["Average_models"] = avg_metrics

    return results


# main
if __name__ == "__main__":

    TRAIN_DIR = "D:/NSFF_datasets/train_ProGAN_ADM"  # your training set path
    TEST_DIR = "D:/NSFF_datasets/test"               # your testing set path
    MODEL_PATH = "weight/NSFF.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dinov2_model = DINOv2().to(DEVICE)
    input_dim = (1 + 1 + 3) * 1024

    # train (If the corresponding model already exists under MODEL_PATH, skip training.)
    if not os.path.exists(MODEL_PATH):
        train_dataset = DualBranchDataset(TRAIN_DIR, split="train")
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        classifier = train_classifier(train_loader, dinov2_model, input_dim, DEVICE)

        # save the trained model to MODEL_PATH
        torch.save(classifier.state_dict(), MODEL_PATH)
        print(f"\nModel saved to {MODEL_PATH}")
    else:
        print(f"\nFound existing model at {MODEL_PATH}, skip training")

    # test
    classifier = SimpleClassifier(input_dim).to(DEVICE)
    
    try:
        classifier.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("\nSuccessfully loaded pretrained model")
    except Exception as e:
        print(f"\nError loading model: {e}")
        exit()

    test_dataset = DualBranchDataset(TEST_DIR, split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    results = test_classifier(classifier, test_loader, dinov2_model, DEVICE)


    # print result
    print("\nTest Results:")
    for model, metrics in results.items():
        if model == "Average_models":
            continue
        print(f"\nModel: {model}")
        print(f"Acc: {metrics['accuracy']:.4f}, AP(PR-AUC): {metrics['ap']:.4f}")
        print(f"TP: {metrics['TP']}, FP: {metrics['FP']}, TN: {metrics['TN']}, FN: {metrics['FN']}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}, ECE: {metrics['ece']:.4f}, Brier: {metrics['brier']:.4f}")

    avg_metrics = results["Average_models"]
    print("\n===== Average Generators =====")
    print(f"Acc: {avg_metrics['accuracy']:.4f}, AP(PR-AUC): {avg_metrics['ap']:.4f}")
    print(
        f"TP: {avg_metrics['TP']:.1f}, FP: {avg_metrics['FP']:.1f}, TN: {avg_metrics['TN']:.1f}, FN: {avg_metrics['FN']:.1f}")
    print(f"ROC-AUC: {avg_metrics['roc_auc']:.4f}, ECE: {avg_metrics['ece']:.4f}, Brier: {avg_metrics['brier']:.4f}")
