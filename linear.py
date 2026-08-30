import numpy as np
import matplotlib.pyplot as plt

# make 50 points along a line y = 2x + 3, with a bit of random noise
np.random.seed(42)  # so you get the same "random" numbers every time you run it
x = np.linspace(0, 50, 50)
noise = np.random.randn(50) * 5
y = 2 * x + 3 + noise

# look at what you made before doing anything else
plt.scatter(x, y)
plt.title("Fake data — this is what we're fitting a line to")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

def predict(x, w, b):
    return w * x + b

# test it with random starting guesses
w = 0.0
b = 0.0
print(predict(x, w, b))  # right now this just prints all zeros — expected

def compute_cost(x, y, w, b):
    predictions = predict(x, w, b)
    errors = predictions - y
    cost = np.mean(errors ** 2)
    return cost

print(compute_cost(x, y, w, b))  # with w=0, b=0 this number will be large — that's expected

n = len(x)
w = 0.0
b = 0.0
learning_rate = 0.0005
iterations = 20000
cost_history = []

for i in range(iterations):
    predictions = predict(x, w, b)
    errors = predictions - y

    # the gradients — how much to blame w and b for the current error
    dw = (2/n) * np.sum(errors * x)
    db = (2/n) * np.sum(errors)

    # take a small step downhill
    w = w - learning_rate * dw
    b = b - learning_rate * db

    cost = compute_cost(x, y, w, b)
    cost_history.append(cost)

    if i % 100 == 0:
        print(f"iteration {i}: cost = {cost:.3f}, w = {w:.3f}, b = {b:.3f}")

print(f"\nfinal: w = {w:.3f}, b = {b:.3f}")

plt.plot(cost_history)
plt.title("Cost over time — this should go down and flatten out")
plt.xlabel("iteration")
plt.ylabel("cost")
plt.show()

plt.scatter(x, y, label="actual data")
plt.plot(x, predict(x, w, b), color="red", label="your line")
plt.title("Your line vs. the real data")
plt.legend()
plt.show()

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x.reshape(-1, 1), y)

print(f"sklearn says: w = {model.coef_[0]:.3f}, b = {model.intercept_:.3f}")
print(f"you got:      w = {w:.3f}, b = {b:.3f}")
