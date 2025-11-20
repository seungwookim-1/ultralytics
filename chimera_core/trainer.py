import torch
from torch.utils.data import DataLoader

def train_loop(model, dataloader, optimizer, device):
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    model.to(device)
    model.train()

    for imgs, targets_per_head in dataloader:
        imgs = imgs.to(device)
        targets = {k: v.to(device) for k, v in targets_per_head.items()}

        optimizer.zero_grad()
        out = model(imgs, targets=targets)
        loss = out["total_loss"]
        loss.backward()
        optimizer.step()

