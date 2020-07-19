# cute
cute is an evolutionary algorithm to evolve convolutional neural networks for image classification tasks, and eventually other tasks like object detection.
cute stands for **C**NNs through asynchrono**U**s **T**raining and **E**volution


# TODO:
A list of things that need to be done:
- [x] master and worker process logic
- [x] design and implementation of the cnn genome
- [x] implementation of island speciation strategy
- [x] mutations and crossover operations (dependent on [2])
- [x] handling program arguments
- [ ] dataset processing - currently only support MNIST and a debug dataset which is just a truncated MNIST
- [ ] intelligently handle multiple GPUS, train multiple networks on the same gpu if possible
