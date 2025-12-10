import sys
import logging
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

# --- PATH SETUP ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

# --- PROJECT IMPORTS ---
from src.data.datasets_depricated.datasets_curriculum_w_pn import get_stage_dataloader
from src.models.bimamba import CoherentBiMamba

# --- LOGGING SETUP ---
LOG_DIR = project_root / "experiments" / "logs" / "bimamba_curriculum_pn"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LR = 1e-3
NUM_EPOCHS = 20  

# 5-Stage Schedule
SCHEDULE = {
    0: "stage_1",   # Static CD (Foundation)
    4: "stage_2",   # CD + 100k PN (Wake up)
    8: "stage_3",   # CD + 500k PN (Tracking)
    12: "stage_4",  # CD + 1M PN (High Speed)
    16: "stage_5"   # CD + Random PN (Robustness)
}

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    loop = tqdm(loader, leave=False, desc="Train")
    
    for data, targets in loop:
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        
        # Forward Pass
        outputs = model(data) 
        loss = criterion(outputs, targets)
        
        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = outputs.argmax(dim=1) 
        correct += (preds == targets).sum().item()
        total += targets.numel()
        
        loop.set_description(f"Loss: {loss.item():.4f}")

    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
            
    return total_loss / len(loader), correct / total

def main():
    logger.info(f"Starting Phase Noise Curriculum Training on {DEVICE}")
    logger.info(f"Logs will be saved to: {LOG_DIR}")
    
    # 1. Initialize Model
    model = CoherentBiMamba(
        num_classes=4, 
        in_channels=2, 
        d_model=64, 
        num_layers=4
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    current_stage = None
    train_loader = None
    val_loader = None
    best_val_loss = float('inf')
    
    # Create checkpoints directory
    CHECKPOINT_DIR = LOG_DIR / "checkpoints"
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    for epoch in range(NUM_EPOCHS):
        
        # --- CURRICULUM SCHEDULER ---
        if epoch in SCHEDULE:
            new_stage = SCHEDULE[epoch]
            logger.info(f"\n{'='*40}")
            logger.info(f"🚀 LEVEL UP! Entering {new_stage} at Epoch {epoch}")
            logger.info(f"{'='*40}")
            
            try:
                # Load Data
                train_loader = get_stage_dataloader(new_stage, "train", batch_size=BATCH_SIZE, num_workers=4)
                val_loader = get_stage_dataloader(new_stage, "val", batch_size=BATCH_SIZE, num_workers=4)
                current_stage = new_stage
            except FileNotFoundError as e:
                logger.error(f"CRITICAL: Could not load data for {new_stage}")
                logger.error(f"Error details: {e}")
                return

        # --- TRAIN ---
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        
        # --- VALIDATE ---
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        logger.info(
            f"Epoch {epoch+1:02d}/{NUM_EPOCHS} [{current_stage}] | "
            f"Tr Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )
        
        # --- SAVE BEST MODEL ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), LOG_DIR / "best_model.pt")
            logger.info(f"  --> New Best Model Saved (Loss: {best_val_loss:.4f})")

        # --- SAVE STAGE CHECKPOINT ---
        next_epoch = epoch + 1
        if next_epoch in SCHEDULE:
            ckpt_path = CHECKPOINT_DIR / f"checkpoint_{current_stage}_done.pt"
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  --> Stage Complete. Checkpoint saved: {ckpt_path.name}")

if __name__ == "__main__":
    main()