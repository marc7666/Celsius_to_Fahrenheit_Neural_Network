import tensorflow as tf
import numpy as np

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float) # Inputs
farenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float) # Outputs

cape = tf.keras.layers.Dense(units=1, input_shape=[1]) # A dense layer is connected to each neuron in the next layer 
# units = number of neurons in the layer
# input_shape = 1 input with 1 neuron
model = tf.keras.Sequential([cape])

model.compile(
    optimizer = tf.keras.optimizers.Adam(0.1), # Allows the layer to adjust biases and weights efficiently for it to learn 
    # The number tells the optimizer how much to adjust for weights and biases. 
    loss = 'mean_squared_error' # This function considers that a small number of large errors is worse than a large number of small errors 
    
)

print("Starting training...\n")
historyal = model.fit(celsius, farenheit, epochs=1000, verbose=False)
# model.fit(inputs, outputs, epochs => 1 lap means reviewing the data only once)
print("Model trained!\n")

# Result of the loss function.
# This function tells us how bad are the results of the network in each epoch
import matplotlib.pyplot as plt
plt.xlabel("# Epoch")
plt.ylabel("Loss magnitude")
plt.plot(historyal.history["loss"])

print("Let's make a prediction!")
result = model.predict([100.0])
print("The result of converting 100.0 Celsius degrees in Farenheit is: " + str(result))
# 100 Celsius degrees = 212 Fahrenheit

print("Internal variables of the model \n")
print(cape.get_weights())

"""
Input                   Outout
            1.798    
100ºC ----------------> 31.9ºF



100 * 1.798 = 179.8 + 31.9 = 211.74
"""