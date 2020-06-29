import logging

from tensorflow import keras

if False:
    from program_arguments import ProgramArguments


class Dataset:

    @staticmethod
    def make_mnist_dataset():
        # logging.info("note: the mnist data set is getting truncated for the purpose of debugging.")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")
        
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        y_train = keras.utils.to_categorical(y_train)
        y_test = keras.utils.to_categorical(y_test)

        return Dataset(x_train, y_train, x_test, y_test, 28, 28, 1)
    

    @staticmethod
    def make_debug_dataset():
        """
        A truncated version of mnist that shouldn't take a lot of time to train for, useful for debugging
        """
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")
        
        truncated_at = 1000
        x_train = x_train.reshape(-1, 28, 28, 1)[:truncated_at]
        x_test = x_test.reshape(-1, 28, 28, 1)[:truncated_at]

        y_train = keras.utils.to_categorical(y_train)[:truncated_at]
        y_test = keras.utils.to_categorical(y_test)[:truncated_at]

        return Dataset(x_train, y_train, x_test, y_test, 28, 28, 1)


    @staticmethod
    def dataset_from_arguments(program_arguments: 'ProgramArguments'):
        dataset_str = program_arguments.args.dataset

        if dataset_str == "mnist":
            return Dataset.make_mnist_dataset()
        elif dataset_str == "debug":
            return Dataset.make_debug_dataset()
        else:
            raise NotImplementedError(f"dataset '{dataset_str}' has not been implemented")


    def __init__(self, x_train, y_train, x_test, y_test, width: int, height: int, channels: int):
        self.x_train = x_train
        self.y_train = y_train
        
        self.x_test = x_test
        self.y_test = y_test

        self.width: int = width
        self.height: int = height
        self.channels: int = channels



