## Creating a new edge
When a new edge is created, the input volume size must be greater than the output volume size.
The input and output layer must also be able to be connected without creating a cycle.
All possible strides will be calculated and a random one will be selected.
The input volume size, output volume size, and stride will determine the filter size.
A random number of filters is selected (the possible numbers of filters is a hyper parameter).

# split edge
1. Pick any random edge which connects layer A to layer B
2. Disable it
3. Create a new with a random volume size that is less than the size of layer A and greater than the size of layer B
    ->  It may be the case that layer B is an output layer (i.e. the randomly selected edge is a DenseEdge).
        Then the total size of the new layer should be less than the total size of layer B

# split layer
1. todo: write this

# add layer
1. Pick two random layers, that when connected do not create a cycle
2. Create an intermediate layer with volume size between the two selected layers (if the output layer is the output layer,
    then it need only be less than the input layer size). If the two selected layers are two close in size an intermediate layer may not be possible?
3. connect the selected input and output layer to the new intermediate layer

# add edge
1. Pick two random layers that when connected do not create a cycle
2. Ensure that the volume size of the selected input layer is less than the volume size of the output layer, unless the output layer is the output layer.

# disable layer
1. todo: write this

# disable edge
1. Pick a random edge. The edge must be allowed to be disabled without preventing the input from reaching the output.
2. Disable the edge.

# enable edge
1. Pick a random disabled edge
2. Enable that edge.

# enable layer
1. todo: write this

# clone
1. clone


# Some other possibilities
- add / remove padding
- exotic edges
- 
