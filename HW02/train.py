import gc
from model import *
from utils import *
from data_loader import *
from config import *
from torch.utils.data import DataLoader


if __name__ == '__main__':

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = "cuda"
    print(f'DEVICE: {device}')

    """## Prepare dataset and model"""
    # preprocess data
    train_X, train_y = preprocess_data(split='train', feat_dir='./libriphone/feat', phone_path='./libriphone',
                                       concat_nframes=concat_nframes, train_ratio=train_ratio)
    val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat', phone_path='./libriphone',
                                   concat_nframes=concat_nframes, train_ratio=train_ratio)

    # get dataset
    train_set = LibriDataset(train_X, train_y)
    val_set = LibriDataset(val_X, val_y)

    # remove raw feature to save memory
    del train_X, train_y, val_X, val_y
    gc.collect()

    # get dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # fix random seed
    same_seeds(seed)

    # create model, define a loss function, and optimizer
    model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    """## Training"""

    best_acc = 0.0
    for epoch in range(num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # training
        model.train() # set the model to training mode
        for i, batch in enumerate(tqdm(train_loader)):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            features = features.view(-1, concat_nframes, input_dim_lstm)

            optimizer.zero_grad()
            outputs = model(features)

            output_flatten = outputs.view(-1, 41)
            labels_flatten = labels.view(-1)

            loss = criterion(output_flatten, labels_flatten)
            loss.backward()
            optimizer.step()

            # new
            outputs = outputs[:, concat_nframes // 2, :].view(outputs.shape[0], -1)
            labels = labels[:, concat_nframes // 2]

            _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
            train_loss += loss.item()

        # validation
        if len(val_set) > 0:
            model.eval() # set the model to evaluation mode
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_loader)):
                    features, labels = batch
                    features = features.to(device)
                    labels = labels.to(device)

                    features = features.view(-1, concat_nframes, input_dim_lstm)

                    outputs = model(features)

                    output_flatten = outputs.view(-1, 41)
                    labels_flatten = labels.view(-1)

                    loss = criterion(output_flatten, labels_flatten)

                    # new
                    outputs = outputs[:, concat_nframes // 2, :].view(outputs.shape[0], -1)
                    labels = labels[:, concat_nframes // 2]

                    _, val_pred = torch.max(outputs, 1)
                    val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                    val_loss += loss.item()

                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                    epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
                ))

                # if the model improves, save a checkpoint at this epoch
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), model_path)
                    print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
        else:
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader)
            ))

    # if not validating, save the last epoch
    if len(val_set) == 0:
        torch.save(model.state_dict(), model_path)
        print('saving model at last epoch')

    del train_loader, val_loader
    gc.collect()
