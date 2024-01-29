# import torch
# import torch.nn as nn
# # Assuming your data tensor has dimensions (10, 35, 300)
# # data_tensor = torch.randn(10, 35, 300)
# # print(data_tensor)

# # Apply mean pooling along the temporal dimension (axis 1)
# # mean_pooled_tensor = nn.AdaptiveAvgPool1d(1)(data_tensor.permute(0,2,1)).squeeze()

# # # The resulting tensor will have dimensions (10, 300)
# # print(mean_pooled_tensor.shape)


# embeddings = torch.empty(10, 10).normal_(mean=0, std=1)
# print(embeddings)

# mbeddings = torch.randn(10, 10)
# print(embeddings)

# ANSI escape codes for text colors
# class Colors:
#     RED = '\033[91m'
#     GREEN = '\033[92m'
#     YELLOW = '\033[93m'
#     BLUE = '\033[94m'
#     END = '\033[0m'

# # Example usage
# print(Colors.RED + "This text is red." + Colors.END)
# print(Colors.GREEN + "This text is green." + Colors.END)
# print(Colors.YELLOW + "This text is yellow." + Colors.END)
# print(Colors.BLUE + "This text is blue." + Colors.END)


import torch
import torch.nn as nn
import numpy as np
import pdb

class TESTRNN(nn.Module):
    def __init__(self):
        super(TESTRNN, self).__init__()
        self.rnn = nn.RNN(1, 20, 1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(40, 1)
    
    def forward(self, x):
        _, hn = self.rnn(x)
        hn = hn.view(-1, 40)
        output = self.fc(hn)
        return output

def train(model, X, t, epochs, optimizer, loss_fn):
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output.view(-1), t)
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}, Loss: {loss.item()}')
        accuracy, out = evaluate(model, X, t)
        print(f'Epoch: {epoch}, Accuracy: {accuracy}')
        if epoch == epochs-1:
            print(out)

        
def evaluate(model, X, t):
    model.eval()
    output = model(X)
    output = output.view(-1)
    # pdb.set_trace()
    output = np.around(output.detach().numpy()).astype(int)
    accuracy = 0
    for i in range(len(output)):
        if output[i] == t[i]:
            accuracy += 1
    accuracy /= len(output)
    return accuracy, output



if __name__ == '__main__':
    # Create dataset
    nb_of_samples = 20
    sequence_len = 10
    # Create the sequences
    X = np.zeros((nb_of_samples, sequence_len))
    for row_idx in range(nb_of_samples):
        X[row_idx,:] = np.around(np.random.rand(sequence_len)).astype(int)
    # Create the targets for each sequence
    t = np.sum(X, axis=1)

    # Create the RNN model
    model = TESTRNN()
    # Set the parameters
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # Set the input and target tensors
    X_tensor = torch.from_numpy(X).type(torch.FloatTensor)
    t_tensor = torch.from_numpy(t).type(torch.FloatTensor).view(-1)

    train(model, X_tensor.view(-1, sequence_len, 1), t_tensor, 1000, optimizer, nn.MSELoss())
    print(t_tensor)
    # test
    model.eval()
    test_data = torch.tensor([[[1.0], [0.0], [1.0], [1.0], [0.0], [1.0], [1.0], [0.0], [1.0], [1.0]]])

    print(model(test_data).detach().numpy())

    named_params = model.named_parameters()

    for name, param in named_params:
        print(name, param)


