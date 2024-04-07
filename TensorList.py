from functools import reduce
import tensorly as tly
import numpy as np

def decompose_tensor(tensor, rank, type='tucker'):
    if type == 'parafac':
        ht = HopperParafac(tensor, rank)
    elif type == 'tucker':
        ht = HopperTucker(tensor, rank)
    else:
        raise ValueError("Decompose method must be either parafac or tucker")
    return ht

class HopperTensor:
    def __init__(self, tensor,rank = None):
        self.tensor = tensor
        self.output = None # stores the output from the function call.
        self.decomposed_rank = rank
        self.reconstructed_tensor = None

    @staticmethod
    def product_of_dimensions(tensor):
        """Return the number of elements in a tensor or matrix.
        """
        return reduce(lambda x, y: x * y, tensor.shape)

    @staticmethod
    def error_norm(tensor, reconstructed_tensor):
        """Return the error between the original tensor and the reconstructed tensor.
        """
        if tensor is None or reconstructed_tensor is None:
            return None
        return tly.norm(reconstructed_tensor - tensor)/tly.norm(tensor)
    @staticmethod
    def sum_of_dimensions(tensor):
        """
        Returns the sum of dimensions of the tensor
        """
        return sum(tensor.shape)

    def decompose(self):
        raise NotImplementedError("Decompose method must be implemented in the subclass.")

    def calculate_total_space_saved(self):
        raise NotImplementedError("Space saving method must be implemented in the subclass.")

    def space_saving_percentage(self):
        np.round(self.calculate_total_space_saved() / self.product_of_dimensions(self.tensor))

class HopperParafac(HopperTensor):
    """Accepts a tensor or matrix and decomposes it into the relevant factors."""
    def __init__(self, tensor, rank):
        super().__init__(tensor, rank)
        self.output = tly.decomposition.parafac(self.tensor, self.decomposed_rank)
        self.reconstructed_tensor = tly.cp_to_tensor(self.output)
        self.space_saved = self.calculate_total_space_saved()
        self.space_saved_percentage = np.round(self.space_saved / self.product_of_dimensions(self.tensor),2)
        self.reconstruction_error = self.error_norm(self.tensor, self.reconstructed_tensor)


    def calculate_total_space_saved(self):
        # calculate the space saved by the decomposition
        # For parafac the final size is given by the formula
        # (I1 * I2 * I3 * ... * In) - (I1 * R1 + I2 * R2 + I3 * R3 + ... + In * Rn)
        # here we hold the dimensional rank constant, you can change that by specifying a list of modes in tly.decomposition.parafac(tensor, rank)
        # in other words : the final size is the sum of all the dimensions times rank
        # since this is the  general case of an SVD, you can see this more clearly by an example lets say you have a matrix of size 87 x 43
        # the final size of the matrix is 87 * 43 = 3741
        # if you decompose it into a rank 2 matrix, the final size is 87 * 2 + 43 * 2 = 260
        # so the space saved is 3741 - 260 = 3481
        # consider a tensor example of size 8192 x 128 x 64
        # the final size of the tensor is 8192 * 128 * 64 = 67108864
        # if you decompose it into a rank 2 tensor, the final size is 8192 * 2 + 128 * 2 + 64 * 2 = 332
        # so the space saved is 67108864 - 332 = 67108532
        error = (self.product_of_dimensions(self.tensor) - (self.sum_of_dimensions(self.tensor) * self.decomposed_rank))
        return np.round(error, 2)


class HopperTucker(HopperTensor):
    """Accepts a tensor or matrix and decomposes it into the relevant factors."""
    def __init__(self, tensor, rank):
        super().__init__(tensor, rank)
        self.output = tly.decomposition.tucker(self.tensor, self.decomposed_rank)
        self.reconstructed_tensor = tly.tucker_to_tensor(self.output)
        self.space_saved = self.calculate_total_space_saved()
        self.space_saved_percentage = np.round(self.space_saved / self.product_of_dimensions(self.tensor),2)
        self.reconstruction_error = self.error_norm(self.tensor, self.reconstructed_tensor)

    def space_saving_percentage(self):
        np.round(self.calculate_total_space_saved() / self.product_of_dimensions(self.tensor))

    def calculate_total_space_saved(self):
        # calculate the space saved by the decomposition
        # For tucker the final size is given by the formula
        # (I1 * I2 * I3 * ... * In) - (I1 * R1 + I2 * R2 + I3 * R3 + ... + In * Rn) + R1 * R2 * R3 * ... * Rn
        # here we hold the dimensional rank constant, you can change that by specifying a list of modes in tly.decomposition.tucker(tensor, rank)
        # in other words : the final size is the sum of all the dimensions times rank plus the rank to the power of dimensions
        # since this is the  general case of an SVD, you can see this more clearly by an example lets say you have a matrix of size 87 x 43
        # the final size of the matrix is 87 * 43 = 3741
        # if you decompose it into a rank 2 matrix, the final size is 87 * 2 + 43 * 2 + 2 * 2 = 174
        # so the space saved is 3741 - 174 = 3567
        # consider a tensor example of size 8192 x 128 x 64
        # the final size of the tensor is 8192 * 128 * 64 = 67108864
        # if you decompose it into a rank 2 tensor, the final size is 8192 * 2 + 128 * 2 + 64 * 2 + 2 * 2 * 2  = 16776
        # so the space saved is 67108864 - 16776 = 67092088
        number_of_tensor_dimensions = len(self.tensor.shape) # the number of dimensions of the tensor
        error = (self.product_of_dimensions(self.tensor) - (self.sum_of_dimensions(self.tensor) * self.decomposed_rank) + self.decomposed_rank ** number_of_tensor_dimensions)
        return  np.round(error, 2)



class TensorList:
    """Accepts a list of tensors or matrices as outputted by a neural network and provides methods for
    manipulating them.
    """
    def __init__(self, tensor_list, ranks, decompose_method='parafac'):
        self.tensor_list = tensor_list
        self.factors = []
        self.recomposed_tensors = []
        self.error = []
        self.space_saved = []
        self.decompose_method = decompose_method
        # rank can be either a list of ranks or a single rank, if its a single rank create a list of ranks of the same length as the tensor_list
        if isinstance(ranks, int):
            self.ranks = [ranks] * len(tensor_list)
        else:
            assert len(ranks) == len(tensor_list), "The rank list must be the same length as the tensor list"
            self.ranks = ranks

    def size(self):
        """Return the number of elements in the list of tensors.
        """
        return sum([HopperTensor.product_of_dimensions(x) for x in self.tensor_list])

    def calculate_percentage_space_saved(self):
        """Return the number of elements saved by the low rank approximation.
        """
        total_space_saved_in_reconstruction = sum(self.space_saved)
        total_size_of_tensors = sum([HopperTensor.product_of_dimensions(x) for x in self.tensor_list])
        return np.round((total_space_saved_in_reconstruction / total_size_of_tensors),2)

    def decompose(self):
        """ Perform low rank approximation of each tensor in the list.
        Returns a list of lists of matrices. Each list of matrices represents the low rank approximation of each tensor in the list.
        Such that the original tensor can be reconstructed by multiplying the matrices in the list.
        """
        # self.decomposed_tensors = [self.parafac(x, 2) for x in self.tensor_list]
        # self.factors = [self.parafac(x, r) for x, r in zip(self.tensor_list, self.rank)]
        for i in range(len(self.tensor_list)):
            tensor_to_decompose = self.tensor_list[i]
            rank = self.ranks[i]

            if self.decompose_method == 'parafac':
                decomposer = HopperParafac(tensor_to_decompose, rank)
            elif self.decompose_method == 'tucker':
                decomposer = HopperTucker(tensor_to_decompose, rank)
            else:
                raise ValueError("Decompose method must be either parafac or tucker")

            # append each thing to relevant list
            self.error.append(decomposer.reconstruction_error)
            self.space_saved.append(decomposer.space_saved)
            print(f"tensor {i} rec_error : {np.round(decomposer.reconstruction_error,2)*100}%, space saved {decomposer.space_saved_percentage}%")

    def factorize(self):
        """Return the list of parafac decomposed tensors.
        """
        self.decompose()

    def __str__(self):
        return str(self.tensor_list)


# sample usage of the TensorList class
# create a list of 3 random 4D tensors
if __name__ == "__main__":
    t1 = np.random.rand(8192, 128)
    t2 = np.random.rand(6, 4, 4, 2)
    t3 = np.random.rand(8, 6, 6, 4)

    tensor_list = [t1, t2, t3]
    # create a list of ranks for each tensor
    TensorList(tensor_list, [2, 2, 2]).factorize()
    import pickle
    with open('../../../../weights_biases.pkl', 'rb') as f:
        tensor_list_all = pickle.load(f)

    with open('../../../../X.pkl', 'rb') as f:
        X = pickle.load(f)

    first_layer_weights = tensor_list_all[0][0]
    print(first_layer_weights.shape)

    original_prediction = np.dot(X, first_layer_weights).flatten()

    #now we have the original prediction, we can compare it with the prediction from the decomposed tensor
    ht = HopperTucker(first_layer_weights, 2)
    decomposed_prediction = np.dot(X, ht.reconstructed_tensor).flatten()

    #calculate the error between the original prediction and decomposed prediction using mae, mape and r2
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
    mae = mean_absolute_error(original_prediction, decomposed_prediction)
    mape = mean_absolute_percentage_error(original_prediction, decomposed_prediction)
    r2 = r2_score(original_prediction, decomposed_prediction)
    norm_error = np.linalg.norm(first_layer_weights - ht.reconstructed_tensor)

    print(f"MAE: {mae}, MAPE: {mape}, R2: {r2}, Norm error: {norm_error}")






    #
    #
    # tensor_list = [tensor_list_all[0][0]]
    # # create a list of ranks for each tensor
    # max_r = int((t1.shape[0] * t1.shape[1])/(t1.shape[0] + t1.shape[0]))
    # min_r = 2
    #
    # import tensorly as tl
    # from tensorly import random
    # from tensorly.decomposition import tucker
    #
    # core, factors = tucker(t1, rank=2)
    #
    #
    # # create a TensorList object
    # tl = TensorList(tensor_list, max_r)
    # # perform low rank approximation of each tensor
    # tl.factorize()
    # # print the reconstructed tensors
    # # print(tl.recomposed_tensors)
    # # # print the reconstruction error
    # # # print the original tensors
    # # print(tl.tensor_list)
    # # # print the decomposed tensors
    # # print(tl.factors)
    #
    #
    # # add a tensor to the list
    # # t4 = np.random.rand(2, 2, 2, 2)
    # # tl.add(t4)
    # # # print the updated tensor list
    # # print(tl.tensor_list)
    # # # subtract a tensor from the list
    # # tl.subtract(t4)
    # # # print the updated tensor list
    # # print(tl.tensor_list)
    # # # multiply a tensor with the list
    # # tl.multiply(t4)
    # # # print the updated tensor list
    # # print(tl.tensor_list)
    # # # divide a tensor by the list
    # # tl.divide(t4)
    # # # print the updated tensor list
    # # print(tl.tensor_list)
    #
    # # Define your loss function L(r) here
    # # Define your loss function L(r) here
    # t1 = np.random.rand(8192, 128)
    #
    #
    # LEARNING_RATE = 1
    # def loss_function(r, weight_factor=0.05):
    #
    #
    #
    #     tl = TensorList([t1], r)
    #     tl.factorize()
    #     recomposed_tensor = tl.recomposed_tensors[0]
    #     return -(np.linalg.norm(t1 - recomposed_tensor) + r/max_r)
    #
    #
    #
    # # Numerical approximation of the derivative of the loss function
    # def numerical_derivative_loss_function(r, h=LEARNING_RATE):
    #     return (loss_function(r + h) - loss_function(r - h)) / (2 * h)
    #
    #
    # # Newton-Raphson method implementation with numerical derivative
    # def newton_raphson(loss_function, numerical_derivative_loss_function, initial_guess, tolerance=1, max_iterations=100):
    #     x = int(initial_guess)
    #     iterations = 0
    #     while True:
    #         f_x = loss_function(x)
    #         f_prime_x = numerical_derivative_loss_function(x)
    #
    #         if f_prime_x == 0:
    #             print("Derivative is zero, cannot proceed.")
    #             break
    #
    #         x_new = int(x - f_x / f_prime_x)
    #
    #         if abs(x_new - x) < tolerance:
    #             break
    #
    #         x = x_new
    #         iterations += 1
    #
    #         if iterations >= max_iterations:
    #             print("Newton-Raphson method did not converge within maximum iterations.")
    #             break
    #
    #     return x, loss_function(x)
    #
    #
    # # Set the initial guess and call the Newton-Raphson method
    # initial_guess = max_r/2  # Starting from the lower bound of the range
    # minimum_r, minimum_loss = newton_raphson(loss_function, numerical_derivative_loss_function, initial_guess)
    #
    # print("Minimum value of r:", minimum_r)
    # print("Minimum loss:", minimum_loss)
    #
    # # Set the initial guess and call the Newton-Raphson method
    # initial_guess = 2  # Starting from the lower bound of the range
    # minimum_r, minimum_loss = newton_raphson(loss_function, numerical_derivative_loss_function, initial_guess)
    #
    # print("Minimum value of r:", minimum_r)
    # print("Minimum loss:", minimum_loss)