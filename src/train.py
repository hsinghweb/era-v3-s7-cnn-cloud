import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model import Model3
from utils import get_data_loaders
from torchsummary import summary

train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

  return 100*correct/processed  # Return the final training accuracy

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    accuracy = 100. * correct / len(test_loader.dataset)
    test_acc.append(accuracy)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    return test_loss, accuracy  # Return both test loss and accuracy


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    model = Model3().to(device)
    summary(model, input_size=(1, 28, 28))
    BATCH_SIZE = 128

    model = Model3().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train_loader, test_loader = get_data_loaders(BATCH_SIZE)

    best_train_acc = 0
    best_test_acc = 0
    best_test_epoch = 0
    
    EPOCHS = 15
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train_acc = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader)
        
        # Update best accuracies
        best_train_acc = max(best_train_acc, train_acc)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_epoch = epoch + 1

    # Print final summary
    total_params = sum(p.numel() for p in model.parameters())
    print("\n===================")
    print("Results:")
    print(f"Parameters: {total_params/1000:.1f}k")
    print(f"Best Train Accuracy: {best_train_acc:.2f}")
    print(f"Best Test Accuracy: {best_test_acc:.2f} ({best_test_epoch}th Epoch)")
    print("===================")




if __name__ == '__main__':
    main() 