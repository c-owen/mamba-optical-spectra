import sys
import logging
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

# --- PATH SETUP ---
# Calculate project root: src/training/ -> src/ -> root
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

# --- PROJECT IMPORTS ---
# Now we can import from the other folders
from src.data.datasets_curriculum import get_stage_dataloader
from src.models.bimamba import CoherentBiMamba

# --- LOGGING SETUP ---
# Ensure logs directory exists
LOG_DIR = project_root / "experiments" / "logs" / "bimamba_curriculum"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LR = 1e-3
NUM_EPOCHS = 15

# The Curriculum Schedule: {Epoch_Threshold: Stage_Name}
# Epoch 0-4: Stage 1
# Epoch 5-9: Stage 2
# Epoch 10+: Stage 3
SCHEDULE = {
    0: "stage_1",
    5: "stage_2",
    10: "stage_3"
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    loop = tqdm(loader, leave=False)
    for batch_idx, (data, targets) in enumerate(loop):
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        
        # Forward
        # Dataset produces [B, 2, L]
        # Model expects [B, 2, L] (it handles the transpose internally)
        outputs = model(data)  # Output: [B, 4, L] (Channels First)
        
        # Loss Calculation
        # CrossEntropyLoss handles (Batch, Class, d1...) vs (Batch, d1...)
        loss = criterion(outputs, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy Calculation
        # Argmax over dimension 1 (Classes) because shape is [B, 4, L]
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
            
            outputs = model(data) # [B, 4, L]
            
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
            
    return total_loss / len(loader), correct / total

def main():
    logger.info(f"Starting Curriculum Training on {DEVICE}")
    
    # 1. Initialize Model
    # Input dim: 2 (Real, Imag), Output dim: 4 (QPSK symbols)
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
    
    for epoch in range(NUM_EPOCHS):
        # 2. Check Curriculum Schedule
        if epoch in SCHEDULE:
            new_stage = SCHEDULE[epoch]
            logger.info(f"--- LEVEL UP! Entering {new_stage} ---")
            
            train_loader = get_stage_dataloader(new_stage, "train", batch_size=BATCH_SIZE)
            val_loader = get_stage_dataloader(new_stage, "val", batch_size=BATCH_SIZE)
            current_stage = new_stage
            
        # 3. Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        
        # 4. Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        logger.info(
            f"Epoch {epoch+1}/{NUM_EPOCHS} [{current_stage}] | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )
        
        # Save checkpoint
        save_path = LOG_DIR / f"model_epoch_{epoch}.pt"
        torch.save(model.state_dict(), save_path)
        logger.info(f"Saved checkpoint: {save_path}")

if __name__ == "__main__":
    Path("checkpoints").mkdir(exist_ok=True)
    main()