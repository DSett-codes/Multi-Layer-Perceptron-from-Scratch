import random
import numpy as np
import matplotlib.pyplot as plt


from MLP.engine import Value
from MLP.nn import Neuron, Layer, MLP
# from MLP.utils import loss # This is not needed and is incorrect for this problem
np.random.seed(1337)
random.seed(1337)


import pandas as pd
df = pd.DataFrame({
    'a':[0,0,1,1],
    'b':[0,1,0,1],
    'y':[0,1,1,0]
})
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# data conversion to list of lists/numbers
X_train = X.values.tolist()
y_train = y.tolist()

model = MLP(2, [16, 16, 1]) # 2-layer neural network
print(model)
print("number of parameters in model: ", len(model.parameters()))
print("\n--- Training ---")

# training loop
for k in range(101):
    # forward pass
    y_pred = [model(x) for x in X_train]
    
    # calculate loss (Mean Squared Error)
    total_loss = sum((yout - ygt)**2 for ygt, yout in zip(y_train, y_pred))
    
    # backward pass
    model.zero_grad()
    total_loss.backward()
    
    # update (sgd)
    learning_rate = 0.2 - 0.18*k/100 # decreasing learning rate from 0.2 to 0.02
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    
    if k % 10 == 0:
        # calculate accuracy
        predicted_labels = [1 if yout.data > 0.5 else 0 for yout in y_pred]
        correct_predictions = sum(p == a for p, a in zip(predicted_labels, y_train))
        acc = correct_predictions / len(y_train)
        print(f"step {k} loss {total_loss.data:.4f}, accuracy {acc*100:.2f}%")

print("\n--- Final Predictions ---")
for i, (x, y) in enumerate(zip(X_train, y_train)):
    pred = model(x)
    pred_class = 1 if pred.data > 0.5 else 0
    print(f"input: {x}, actual: {y}, predicted_val: {pred.data:.4f}, predicted_class: {pred_class}")
