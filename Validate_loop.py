import torch


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, use_amp: bool = False):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_sum += loss.item() * x.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)

    return loss_sum / max(1, total), 100.0 * correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp: bool = False):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)

        loss_sum += loss.item() * x.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)

    return loss_sum / max(1, total), 100.0 * correct / max(1, total)
