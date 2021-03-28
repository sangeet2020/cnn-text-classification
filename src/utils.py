import time
import os
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from arguments import parse_arguments

args = parse_arguments()

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")


class DatasetMaper(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Run:
    '''Training, evaluation and metrics calculation'''

    @staticmethod
    def train(model, data, params):
        # Initialize dataset maper
        train = DatasetMaper(data['x_train'], data['y_train'])
        test = DatasetMaper(data['x_test'], data['y_test'])
        valid = DatasetMaper(data['x_valid'], data['y_valid'])

        # Initialize loaders
        # import pdb; pdb.set_trace()
        loader_train = DataLoader(train, batch_size=params.batch_size)
        loader_test = DataLoader(test, batch_size=params.batch_size)
        loader_valid = DataLoader(valid, batch_size=params.batch_size)

        # Define optimizer
        optimizer = optim.RMSprop(model.parameters(), lr=params.learning_rate)
        # optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

        # Tracking best validation accuracy
        best_accuracy = 0
        
        print("\nStart training...")
        # print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^9} | {'Test Acc':^9} | {'Elapsed':^9}")
        print("-"*80)
    
        # Starts training phase
        for epoch in range(params.epochs):
            # =======================================
            #               Training
            # =======================================

            # Tracking time and loss
            t0_epoch = time.time()
            total_loss = 0
        
            # Put the model into training mode
            model.train()
        
            # Starts batch training
            for x_batch, y_batch in loader_train:
                # Load batch to GPU
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.type(torch.FloatTensor)
                y_batch = y_batch.to(DEVICE)
                
                # Feed the model
                y_pred = model(x_batch.to(torch.int64))

                # Compute loss and accumulate the loss values
                # import pdb; pdb.set_trace()
                loss = F.cross_entropy(y_pred, y_batch.long(), reduction='sum')
                total_loss += loss.item()

                # Clean gradients
                optimizer.zero_grad()

                # Gradients calculation
                loss.backward()

                # Gradients update
                optimizer.step()
            
                # break

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(loader_train)
            
            # =======================================
            #               Evaluation
            # =======================================
            
            # After the completion of each training epoch, measure the model's
            # performance on our validation set.
            
            # Training metrics
            corrects = (torch.max(y_pred, 1)[1].view(y_batch.size()).data == y_batch.data).sum()
            train_accuracy = 100.0 * corrects/loader_train.batch_size
            
            # Validation metrics
            val_loss, val_accuracy = evaluation(model, loader_valid)
            _, test_accuracy = evaluation(model, loader_test)
            
            # Track the best accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch + 1:^7} | {avg_train_loss:^12.6f} | {train_accuracy:^10.2f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {test_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            
            
        print("\n")
        print(f"Training complete! \nBest accuracy: {best_accuracy:.2f} %.")
        
        os.makedirs(args.out_dir, exist_ok=True)
        if args.save_model:
            PATH = args.out_dir + "/model_epochs_" + str(params.epochs) +"_batch_size_" + str(params.batch_size) + ".pth"
            torch.save(model.state_dict(), PATH)
            print("Model saved: ",PATH)


def evaluation(model, loader_valid):

    # Set the model in evaluation mode

    model.eval()
    corrects, avg_loss = 0, 0
    
    # Starst evaluation phase   
    with torch.no_grad():
        for x_batch, y_batch in loader_valid:
            x_batch = x_batch.to(DEVICE); y_batch = y_batch.to(DEVICE)
            x_batch = x_batch.to(torch.int64)
            
            y_pred = model(x_batch)
            
            loss = F.cross_entropy(y_pred, y_batch.long(), reduction='sum')
            
            avg_loss += loss.item()
            corrects += (torch.max(y_pred, 1) [1].view(y_batch.size()).data == y_batch.data).sum()
            
    size = len(loader_valid.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    
    return avg_loss, accuracy

