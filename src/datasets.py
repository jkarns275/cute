from tensorflow import keras

class Dataset:
    

    def dataset_from_arguments(program_arguments: 'ProgramArguments'):
        dataset_str = program_arguments.args.dataset

        if dataset_str == "mnist":
            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")
            
            x_train = x_train.reshape(-1, 28, 28, 1)
            x_test = x_test.reshape(-1, 28, 28, 1)

            y_train = keras.utils.to_categorical(y_train)
            y_test = keras.utils.to_categorical(y_test)

            return Dataset(x_train, y_train, x_test, y_test, 28, 28, 1)
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



