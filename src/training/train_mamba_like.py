import argparse
import logging
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.datasets import get_dataloaders, get_class_mapping
from src.models.mamba_like_1d import SpectraMamba


def setup_logging(verbosity: int = 1) -> None:
    level = logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 50 == 0:
            logging.info(
                "Epoch %d | Batch %d/%d | Loss: %.4f | Acc: %.2f%%",
                epoch,
                batch_idx,
                len(dataloader),
                loss.item(),
                100.0 * correct / max(1, total),
            )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model,
    dataloader,
    criterion,
    device,
    split_name: str = "val",
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    logging.info(
        "%s | Loss: %.4f | Acc: %.2f%%",
        split_name,
        epoch_loss,
        100.0 * epoch_acc,
    )
    return epoch_loss, epoch_acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 1D CNN baseline on synthetic optical spectra.",
    )
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="mps")  # 'cuda', 'cpu', or 'mps' on Mac
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=str, default="experiments/logs/mamba")
    parser.add_argument("-v", "--verbose", action="count", default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    set_seed(args.seed)

    # Device selection: mps (Apple GPU), cuda (NVIDIA), or cpu
    if args.device == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("MPS device detected. Using Apple GPU via Metal Performance Shaders.")
        else:
            logging.warning("MPS requested but not available. Falling back to CPU.")
            device = torch.device("cpu")

    elif args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info("CUDA GPU detected. Using NVIDIA GPU.")
        else:
            logging.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")

    else:
        device = torch.device("cpu")
        logging.info("Using CPU.")
        logging.info("Using device: %s", device)

    data_dir = Path(args.data_dir)
    loaders = get_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        normalize=True,
        pin_memory=False,  # on MPS pin_memory is not useful
    )

    class_map = get_class_mapping(data_dir)
    num_classes = len(class_map)

    model = SpectraMamba(
        num_classes=num_classes,
        in_channels=1,
        d_model=128,
        num_layers=6,
        kernel_size=7,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    best_model_path = run_dir / "best_cnn.pt"

    logging.info("Starting training for %d epochs", args.epochs)
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            epoch,
        )
        logging.info(
            "Epoch %d | Train Loss: %.4f | Train Acc: %.2f%%",
            epoch,
            train_loss,
            100.0 * train_acc,
        )

        val_loss, val_acc = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
            split_name="val",
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "class_mapping": class_map,
                },
                best_model_path,
            )
            logging.info(
                "New best model saved to %s (val_acc=%.2f%%)",
                best_model_path,
                100.0 * val_acc,
            )

    total_time = time.time() - start_time
    logging.info("Training completed in %.1f seconds", total_time)

    # Final test evaluation with best model
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info(
            "Loaded best model from %s (epoch=%d, val_acc=%.2f%%)",
            best_model_path,
            checkpoint["epoch"],
            100.0 * checkpoint["val_acc"],
        )
    else:
        logging.warning("Best model checkpoint not found, evaluating last model.")

    test_loss, test_acc = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        split_name="test",
    )
    logging.info("Final test accuracy: %.2f%%", 100.0 * test_acc)


if __name__ == "__main__":
    main()
