import numpy as np
from sklearn.metrics import accuracy_score
from TensorList import HopperTensor, decompose_tensor
import torch
# from keras.src.layers import Dense

class OptimizeRank:
    def __init__(self, tensor, model, y, delta=0.1):
        self.tensor = tensor
        self.model = model
        self.y = y
        self.delta = delta

        if hasattr(model, 'predict'):  # Keras model
            original_predictions = np.argmax(model.predict(tensor), axis=-1)
            self.original_accuracy = accuracy_score(y, original_predictions)

            weights_biases = [(layer.get_weights()[0], layer.get_weights()[1])
                              for layer in model.layers if isinstance(layer, Dense)]

            self.first_layer_weight = [wb[0] for wb in weights_biases][0]
            self.first_layer_bias = [wb[1] for wb in weights_biases][0]
            self.max_rank = int(HopperTensor.product_of_dimensions(self.first_layer_weight) / HopperTensor.sum_of_dimensions(self.first_layer_weight))
        elif hasattr(model, 'forward'):  # PyTorch model
            self.tensor = torch.tensor(tensor).float()
            original_predictions = torch.argmax(model(self.tensor), dim=-1).detach().numpy()
            self.original_accuracy = accuracy_score(y, original_predictions)

            # Extracting weights and biases
            weights_biases = [(layer.weight.detach().numpy(), layer.bias.detach().numpy())
                              for layer in model.modules() if isinstance(layer, torch.nn.Linear)]

            # Extract just the weights of the first layer
            self.first_layer_weight = [wb[0] for wb in weights_biases][0]
            self.first_layer_bias = [wb[1] for wb in weights_biases][0]
            self.max_rank = int(HopperTensor.product_of_dimensions(self.first_layer_weight) / HopperTensor.sum_of_dimensions(self.first_layer_weight))
        else:
            raise ValueError("Unsupported model type. Supported frameworks: Keras, PyTorch")

    def calculate_accuracy_space_saved(self, rank):
        if hasattr(self.model, 'predict'):  # Keras model
            ht = decompose_tensor(self.first_layer_weight, rank=rank)
            first_layer_new_weight = ht.reconstructed_tensor
            # Update only the first layer's weights
            dense_layers = [layer for layer in self.model.layers if isinstance(layer, Dense)]
            dense_layers[0].set_weights([first_layer_new_weight, self.first_layer_bias])
            new_predictions = np.argmax(self.model.predict(self.tensor), axis=-1)
        elif hasattr(self.model, 'forward'):  # PyTorch model
            ht = decompose_tensor(self.first_layer_weight, rank=rank)
            first_layer_new_weight = ht.reconstructed_tensor
            # Update only the first layer's weights
            self.model[0].weight.data = torch.tensor(first_layer_new_weight)
            self.model[0].bias.data = torch.tensor(self.first_layer_bias)
            new_predictions = torch.argmax(self.model(self.tensor), dim=-1).detach().numpy()
        else:
            raise ValueError("Unsupported model type. Supported frameworks: Keras, PyTorch")

        space_saved = ht.space_saved_percentage
        accuracy = accuracy_score(self.y, new_predictions)
        return accuracy, space_saved

    def optimize_rank_brute_force(self):
        accuracy = None
        space_saved = None
        # Optimize the rank of the tensor
        for rank in range(1, int(self.max_rank)):
            accuracy, space_saved = self.calculate_accuracy_space_saved(rank)
            if accuracy > (1 - self.delta) * self.original_accuracy:
                return rank, accuracy, space_saved
        return self.max_rank, accuracy, space_saved

    def optimize_rank_binary_search(self):
        # Optimize the rank of the tensor
        accuracy = None
        space_saved = None
        left, right = 1, int(self.max_rank)
        #print the max_rank
        print(f"Max rank: {self.max_rank}")
        while left < right:
            mid = int((left + right) // 2)
            accuracy, space_saved = self.calculate_accuracy_space_saved(mid)
            if accuracy > (1 - self.delta) * self.original_accuracy:
                right = mid
            else:
                left = mid + 1
        return left, np.round(self.original_accuracy - accuracy, 2), space_saved

if __name__ == '__main__':
    # Example usage
    tensor = np.random.rand(100, 10)
    model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.Linear(5, 3))
    y = np.random.randint(0, 3, 100)
    optimizer = OptimizeRank(tensor, model, y)
    rank, accuracy, space_saved = optimizer.optimize_rank_binary_search()
    print(f"Optimized rank: {rank}, Accuracy: {accuracy}, Space saved: {space_saved}")

    import torch
    import torch.nn as nn
    # example with pytorch class
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(4, 10)  # 4 input features, 10 neurons in the hidden layer
            self.fc2 = nn.Linear(10, 3)  # 10 neurons in the hidden layer, 3 output classes

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

        def __getitem__(self, idx):
            if idx == 0:
                return self.fc1
            elif idx == 1:
                return self.fc2
            else:
                raise IndexError("Index out of range")

    # example on iris data
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #convert to tensor
    X_train_tensor = torch.tensor(X_train).float()
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
    X_test_tensor = torch.tensor(X_test).float()
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)

    model = Net()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    #train the model
    epochs = 100
    for epoch in range(epochs):  # loop over the dataset multiple times
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Example usage
    tensor = X_train
    y = y_train
    opt_rank = OptimizeRank(tensor, model, y)
    rank, accuracy, space_saved = opt_rank.optimize_rank_binary_search()