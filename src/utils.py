import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


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

        # Initialize loaders
        # import pdb; pdb.set_trace()
        loader_train = DataLoader(train, batch_size=params.batch_size)
        loader_test = DataLoader(test, batch_size=params.batch_size)

        # Define optimizer
        optimizer = optim.RMSprop(model.parameters(), lr=params.learning_rate)

        # Starts training phase
        for epoch in range(params.epochs):
            # Set model in training model
            model.train()
            predictions = []
            # Starts batch training
            for x_batch, y_batch in loader_train:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.type(torch.FloatTensor)
                y_batch = y_batch.to(DEVICE)
                
                

                # Feed the model
                # import pdb
                # pdb.set_trace()
                y_pred = model(x_batch.to(torch.int64))

                # Loss calculation
                # import pdb
                # pdb.set_trace()
                loss = F.binary_cross_entropy(y_pred, y_batch)

                # Clean gradients
                optimizer.zero_grad()

                # Gradients calculation
                loss.backward()

                # Gradients update
                optimizer.step()

                # Save predictions
                predictions += list(y_pred.detach())

            # Evaluation phase
            test_predictions = evaluation(model, loader_test)

            # Metrics calculation
            train_accuary = calculate_accuray(data['y_train'], predictions)
            test_accuracy = calculate_accuray(
                data['y_test'], test_predictions)
            print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (
                epoch+1, loss.item(), train_accuary, test_accuracy))
        
        PATH = "model_epochs_" + str(params.epochs) +"_batch_size_" + str(params.batch_size) + ".pth"
        torch.save(model.state_dict(), PATH)


def evaluation(model, loader_test):

    # Set the model in evaluation mode
    # print(DEVICE)
    model.eval()
    predictions = []
    # import pdb; pdb.set_trace()
    # Starst evaluation phase   
    with torch.no_grad():
        for x_batch, _ in loader_test:
            x_batch = x_batch.to(DEVICE)
            x_batch = x_batch.to(torch.int64)
            y_pred = model(x_batch)
            predictions += list(y_pred.detach())
    return predictions


def calculate_accuray(grand_truth, predictions):
    # Metrics calculation
    true_positives = 0
    true_negatives = 0
    for true, pred in zip(grand_truth, predictions):
        if (pred >= 0.5) and (true == 1):
            true_positives += 1
        elif (pred < 0.5) and (true == 0):
            true_negatives += 1
        else:
            pass
    # Return accuracy
    return (true_positives+true_negatives) / len(grand_truth)
