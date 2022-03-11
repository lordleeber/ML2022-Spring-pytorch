"""## Testing
Create a testing dataset, and load model from the saved checkpoint.
"""

from torch.utils.data import DataLoader
from model import *
from utils import *
from data_loader import *
from config import *


if __name__ == '__main__':

    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(f'DEVICE: {device}')

    # load data
    test_X = preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone',
                             concat_nframes=concat_nframes)
    test_set = LibriDataset(test_X, None)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # load model
    model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path))

    """Make prediction."""
    test_acc = 0.0
    test_lengths = 0
    pred = np.array([], dtype=np.int32)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            features = batch
            features = features.to(device)

            features = features.view(-1, concat_nframes, input_dim_lstm)

            outputs = model(features)

            outputs = outputs[:, concat_nframes // 2, :].view(outputs.shape[0], -1)

            _, test_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
            pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

    """Write prediction to a CSV file.
    
    After finish running this block, download the file `prediction.csv` from the files section on the left-hand side and submit it to Kaggle.
    """

    with open('prediction.csv', 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(pred):
            f.write('{},{}\n'.format(i, y))
