import torch
import os
import time


# ===========================================================================================
def train(epoch, trainloader, device, criterion, net, scaler, optimizer, progress_bar):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.cuda.amp.autocast(enabled=False):
            outputs = net(inputs)  # [512, 10]
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return train_loss / (batch_idx + 1)


# ===========================================================================================
def test(epoch, net, device, testloader, criterion, progress_bar, optimizer, save_root):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    acc = 100. * correct / total

    content = time.ctime() + ', ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.15f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)

    with open(os.path.join(save_root, 'log.txt'), 'a') as appender:
        appender.write(content + "\n")

    return test_loss, acc


# ===========================================================================================
def save_model(net, save_root, epoch, best_acc):
    print('Saving..')
    state = {"model": net.state_dict(),
             "epoch": epoch,
             "best_acc": best_acc
             }

    torch.save(state, os.path.join(save_root, 'weight.t7'))
