import logging
from src.data.datasets import get_dataloaders, get_class_mapping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

loaders = get_dataloaders(data_dir="data/processed", batch_size=32, num_workers=0)
class_map = get_class_mapping("data/processed")

train_loader = loaders["train"]

for batch_idx, (x, y) in enumerate(train_loader):
    print("Batch", batch_idx, "x shape:", x.shape, "y shape:", y.shape)
    print("Example labels:", y[:10])
    break

print("Class mapping:", class_map)