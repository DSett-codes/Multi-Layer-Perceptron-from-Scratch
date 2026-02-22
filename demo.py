import random
from MLP.nn import MLP

random.seed(1337)


# XOR dataset:
# [0, 0] -> 0
# [0, 1] -> 1
# [1, 0] -> 1
# [1, 1] -> 0
X_train = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]
y_train = [0, 1, 1, 0]


def class_from_output(output_value):
    """Convert raw output to class 0 or 1."""
    return 1 if output_value.data > 0.5 else 0


model = MLP(2, [4, 4, 1])
print(model)
print("Number of parameters:", len(model.parameters()))
print("\n--- Training ---")

steps = 100
start_lr = 0.2
end_lr = 0.02

for step in range(steps + 1):
    # 1) Forward pass: get predictions for every training sample.
    y_pred = []
    for x in X_train:
        y_pred.append(model(x))

    # 2) Compute Mean Squared Error style total loss.
    total_loss = 0
    for actual, predicted in zip(y_train, y_pred):
        total_loss = total_loss + (predicted - actual) ** 2

    # 3) Backward pass.
    model.zero_grad()
    total_loss.backward()

    # 4) Update parameters with SGD.
    learning_rate = start_lr - (start_lr - end_lr) * (step / steps)
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if step % 10 == 0:
        correct = 0
        for actual, predicted in zip(y_train, y_pred):
            if class_from_output(predicted) == actual:
                correct += 1
        accuracy = correct / len(y_train)
        print(f"step {step:3d} | loss {total_loss.data:.4f} | accuracy {accuracy*100:.2f}%")

print("\n--- Final Predictions ---")
for x, actual in zip(X_train, y_train):
    predicted_value = model(x)
    predicted_class = class_from_output(predicted_value)
    print(
        f"input: {x}, actual: {actual}, "
        f"predicted_value: {predicted_value.data:.4f}, predicted_class: {predicted_class}"
    )
