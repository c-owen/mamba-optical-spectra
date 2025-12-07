import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import Data and the NEW Model
from src.data.datasets_coherent import get_coherent_dataloaders
from src.models.bimamba import CoherentBiMamba

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def calculate_accuracy(outputs, targets):
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).sum().item()
    total = targets.numel()
    return correct, total

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        c, t = calculate_accuracy(outputs, targets)
        correct += c
        total += t

    return total_loss / len(dataloader.dataset), correct / total

def evaluate(model, dataloader, criterion, device, split="val"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            c, t = calculate_accuracy(outputs, targets)
            correct += c
            total += t

    acc = correct / total
    logging.info(f"{split} | Loss: {total_loss/len(dataloader.dataset):.4f} | Acc: {acc*100:.2f}%")
    return total_loss / len(dataloader.dataset), acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run-dir", type=str, default="experiments/logs/bimamba")
    args = parser.parse_args()

    setup_logging()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load Data
    loaders = get_coherent_dataloaders(args.data_dir, args.batch_size)
    
    # Initialize BiMamba
    logging.info("Initializing CoherentBiMamba...")
    model = CoherentBiMamba(num_classes=4, in_channels=2, d_model=64, num_layers=6)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    run_path = Path(args.run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    best_path = run_path / "best_bimamba.pt"
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device, "val")
        scheduler.step()
        
        logging.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            logging.info(f"New Best Model: {best_acc*100:.2f}%")

    # Final Test
    model.load_state_dict(torch.load(best_path))
    test_loss, test_acc = evaluate(model, loaders["test"], criterion, device, "test")
    logging.info(f"Final Test Accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()