import argparse
import os
import pickle
import numpy as np
from typing import Dict

import torch
from torch import nn, optim

from train_tft import DictDataSet, get_set_and_loaders, QueueAggregator, EarlyStopping

class VanillaLSTM(nn.Module):
    """
    Basic LSTM model for benchmarking. Currently treats all numeric features as a single vector input.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super(VanillaLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]  
        output = self.fc(last_out)     # [batch_size, output_size]
        return output


def process_batch_lstm(batch: Dict[str, torch.Tensor],
                       model: nn.Module,
                       device: torch.device,
                       seq_len: int):

    for k in list(batch.keys()):
        batch[k] = batch[k].to(device)

    input_feats = batch['input_feats'].view(-1, seq_len, batch['input_feats'].shape[-1])
    predictions = model(input_feats)
    labels = batch['target'].float()
    if len(labels.shape) == 1:
        labels = labels.view(-1, 1)

    criterion = nn.MSELoss()
    loss = criterion(predictions, labels)
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM benchmark")
    parser.add_argument("filename", type=str, help="data filename")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to pre-trained model")
    args = parser.parse_args()

    filename = args.filename

    # copied from TFT
    configuration = {
        'optimization': {
            'batch_size': {
                'training': 128,
                'inference': 128
            },
            'learning_rate': 0.001,
            'max_grad_norm': 100
        },
        'model': {
            'input_size': 10, #TODO   
            'hidden_size': 64,  
            'num_layers': 2,    
            'output_size': 1,   
            'dropout': 0.1
        },
        'training': {
            'max_epochs': 50,
            'patience': 5,      
            'seq_len': 365 #TODO        
        }
    }

    data_directory = "path/to/data"
    weights_directory = "path/to/save/weights"
    data_path = os.path.join(data_directory, filename)
    output_path = os.path.join(weights_directory, 'weights_lstm_' + filename.split('.')[0] + '.pth')

    print(f"Loading data from: {data_path}")
    print(f"Weights will be saved to: {output_path}")

    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)

    # Not sure if we need 'feature_map' and 'cardinalities_map' 
    # feature_map = data['feature_map']
    # cardinalities_map = data['categorical_cardinalities']

    meta_keys = ['time', 'location', 'soil_x', 'soil_y', 'id']  
    shuffled_loader_config = {
        'batch_size': configuration['optimization']['batch_size']['training'],
        'drop_last': True,
        'shuffle': True
    }
    serial_loader_config = {
        'batch_size': configuration['optimization']['batch_size']['inference'],
        'drop_last': False,
        'shuffle': False
    }

    train_set, train_loader, train_serial_loader = get_set_and_loaders(
        data_dict=data['data_sets']['train'],
        shuffled_loader_config=shuffled_loader_config,
        serial_loader_config=serial_loader_config,
        ignore_keys=meta_keys
    )
    validation_set, validation_loader, validation_serial_loader = get_set_and_loaders(
        data_dict=data['data_sets']['validation'],
        shuffled_loader_config=shuffled_loader_config,
        serial_loader_config=serial_loader_config,
        ignore_keys=meta_keys
    )
    test_set, test_loader, test_serial_loader = get_set_and_loaders(
        data_dict=data['data_sets']['test'],
        shuffled_loader_config=serial_loader_config, 
        serial_loader_config=serial_loader_config,
        ignore_keys=meta_keys
    )

    config = configuration['model']
    model = VanillaLSTM(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=config['output_size'],
        dropout=config['dropout']
    )
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    model.to(device)

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        pretrained_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(pretrained_dict)
        print("Checkpoint loaded!")

    opt = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=configuration['optimization']['learning_rate']
    )
    max_epochs = configuration['training']['max_epochs']
    patience = configuration['training']['patience']
    seq_len = configuration['training']['seq_len']
    epoch_iters = len(data['data_sets']['train']['time_index']) // configuration['optimization']['batch_size']['training']
    eval_iters = len(data['data_sets']['validation']['time_index']) // configuration['optimization']['batch_size']['inference']
    loss_aggregator = QueueAggregator(max_size=50)  
    es = EarlyStopping(patience=patience, mode='min')

    batch_idx = 0
    epoch_idx = 0

    while epoch_idx < max_epochs:
        print(f"Starting Epoch {epoch_idx}")
        
        model.eval()
        with torch.no_grad():
            for subset_name, subset_loader, subset_iters in zip(
                ['train', 'validation', 'test'],
                [train_loader, validation_loader, test_loader],
                [eval_iters, eval_iters, eval_iters]  # or define different eval iters for test
            ):
                print(f"Evaluating {subset_name} set...")
                losses = []
                for _ in range(subset_iters):
                    batch = next(subset_loader)
                    loss = process_batch_lstm(
                        batch=batch,
                        model=model,
                        device=device,
                        seq_len=seq_len
                    )
                    losses.append(loss.item())
                eval_loss = np.mean(losses)
                print(f"Epoch: {epoch_idx}, {subset_name} Loss: {eval_loss:.5f}")

                if subset_name == 'validation':
                    validation_loss = eval_loss
        
        if es.step(torch.tensor(validation_loss)):
            print("Early stopping triggered.")
            break

        model.train()
        for _ in range(epoch_iters):
            batch = next(train_loader)

            opt.zero_grad()
            loss = process_batch_lstm(
                batch=batch,
                model=model,
                device=device,
                seq_len=seq_len
            )
            loss.backward()

            if configuration['optimization']['max_grad_norm'] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), configuration['optimization']['max_grad_norm'])
            opt.step()

            loss_aggregator.append(loss.item())

            if batch_idx % 20 == 0:
                avg_loss = np.mean(loss_aggregator.get())
                print(f"Epoch: {epoch_idx}, Batch Index: {batch_idx}, Train Loss: {avg_loss:.5f}")

            batch_idx += 1

        epoch_idx += 1

    print("Training complete.")
    torch.save(model.state_dict(), output_path)
    print(f"Model weights saved to {output_path}")
