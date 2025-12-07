import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import our new modules
from src.data.datasets_coherent import get_coherent_dataloaders
from src.models.coherent_models import CoherentCNN, CoherentMamba

def setup_logging(verbosity: int = 1) -> None:
    level = logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_accuracy(outputs, targets):
    """
    Calculates accuracy for sequence data.
    outputs: [Batch, Classes, Seq_Len]
    targets: [Batch, Seq_Len]
    """
    # Get the class with max score
    _, predicted = outputs.max(1) # [Batch, Seq_Len]
    
    # Flatten both to calculate global symbol accuracy
    total_symbols = targets.numel()
    correct_symbols = predicted.eq(targets).sum().item()
    
    return correct_symbols, total_symbols

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct_symbols = 0
    total_symbols = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs) # [B, 4, L]
        
        # CrossEntropyLoss expects [B, C, L] logits and [B, L] targets
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Clip gradients for Mamba stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        
        # Accuracy Calc
        corr, tot = calculate_accuracy(outputs, targets)
        correct_symbols += corr
        total_symbols += tot

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_symbols / total_symbols
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device, split_name="val"):
    model.eval()
    running_loss = 0.0
    correct_symbols = 0
    total_symbols = 0

    with torch.inference_mode():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            
            corr, tot = calculate_accuracy(outputs, targets)
            correct_symbols += corr
            total_symbols += tot

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_symbols / total_symbols
    
    logging.info(
        "%s | Loss: %.4f | Symbol Acc: %.2f%%",
        split_name, epoch_loss, 100.0 * epoch_acc,
    )
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description="Train Coherent Optical Decoders (CNN vs Mamba)")
    parser.add_argument("--model", type=str, choices=["cnn", "mamba"], required=True, help="Which architecture to train")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data (e.g., data/coherent_low)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run-dir", type=str, default="experiments/logs/coherent")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Load Data
    loaders = get_coherent_dataloaders(args.data_dir, args.batch_size)
    
    # 2. Initialize Model
    logging.info(f"Initializing {args.model.upper()} model...")
    if args.model == "cnn":
        model = CoherentCNN(num_classes=4, in_channels=2, hidden_dim=64, num_layers=6)
    else:
        model = CoherentMamba(num_classes=4, in_channels=2, d_model=64, num_layers=6)
        
    model.to(device)
    
    # 3. Setup Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    run_dir = Path(args.run_dir) / f"{args.model}_{Path(args.data_dir).name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best_model.pt"
    best_acc = 0.0

    # 4. Train Loop
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device, "val")
        scheduler.step()
        
        logging.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            logging.info(f"New best model saved! ({best_acc*100:.2f}%)")

    # 5. Final Test
    logging.info(f"Training finished in {time.time()-start_time:.1f}s")
    model.load_state_dict(torch.load(best_path))
    test_loss, test_acc = evaluate(model, loaders["test"], criterion, device, "test")
    logging.info(f"Final Test Accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()