# Linear Regression From Scratch

No sklearn doing the work for you. Just numpy, math you can actually see, and a gradient descent loop you write yourself — line by line.

## What this project is

You have a bunch of points on a graph. You want to draw **one straight line** that best fits through them. "Best" means the line sits as close as possible to all the points at once.

A straight line is described by two numbers:

```
y = w·x + b
```

- **w** — the slope (how steep the line is)
- **b** — the intercept (where the line crosses the y-axis)

"Training the model" is nothing more than finding the right `w` and `b`. This project teaches the computer to find them by trial and error — starting from a terrible guess and slowly correcting itself, thousands of times, until the guess is good.

This one idea — **gradient descent** — is the engine behind almost every machine learning model you will ever train. Get comfortable with it here on the simplest possible case, because the next project (Multivariate Linear Regression) and eventually the Neural Network reuse this exact loop.

## Setup

```bash
pip install numpy matplotlib scikit-learn
```

Create one file, `linear_regression_scratch.py`. Build it up piece by piece as you go through this README — don't split it into separate files. Run it after every step.

## How the file is built, step by step

### 1. Imports

```python
import numpy as np
import matplotlib.pyplot as plt
```

`numpy` does fast math on lists of numbers. `matplotlib.pyplot` draws graphs. Without these two lines you have no math tools and no way to see anything.

### 2. Make fake data

```python
np.random.seed(42)
x = np.linspace(0, 50, 50)
noise = np.random.randn(50) * 5
y = 2 * x + 3 + noise
```

- `np.random.seed(42)` — locks the "random" numbers so you get the same result every time you run the file. Makes debugging possible.
- `x = np.linspace(0, 50, 50)` — 50 evenly spaced x-values from 0 to 50.
- `noise = np.random.randn(50) * 5` — 50 small random numbers, so the data isn't perfectly clean (like real-world data never is).
- `y = 2 * x + 3 + noise` — the *true* relationship is `y = 2x + 3`, plus noise. You secretly already know the right answer (w=2, b=3), so later you can check whether your code actually finds it.

Look at the data before doing anything else:

```python
plt.scatter(x, y)
plt.title("Fake data — this is what we're fitting a line to")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

You should see points that roughly go up and to the right. If not, stop and fix this before continuing — everything after depends on it.

### 3. The model — turning a guess into a line

```python
def predict(x, w, b):
    return w * x + b
```

Given an x and your current guesses for `w` and `b`, this returns a predicted y. Right now your guesses will be wrong — that's expected. The rest of the project is about making them less wrong.

### 4. The cost function — measuring how wrong you are

```python
def compute_cost(x, y, w, b):
    predictions = predict(x, w, b)
    errors = predictions - y
    cost = np.mean(errors ** 2)
    return cost
```

- `predictions` — your guessed y for every x.
- `errors` — guessed minus actual, for every point.
- `errors ** 2` — square every error. This kills negative signs (a miss is a miss either direction) and punishes big misses more than small ones.
- `np.mean(...)` — average all the squared errors into one number: the **cost**.

This is called **Mean Squared Error (MSE)**. Big cost = bad line. Small cost = good line. Everything from here is about making this number smaller.

### 5. Gradient descent — the idea, before more code

Imagine you're blindfolded on a hill, trying to reach the lowest point. You can't see the whole hill, but you can feel which direction is downhill from where you're standing. So you take a small step that way, check again, step again — until you reach the bottom.

The "hill" is the cost from step 4. "Standing on the hill" means having some current `w` and `b`. Each step downhill is a small adjustment to `w` and `b` that makes the cost a little smaller. Do this enough times and you land on the best `w` and `b`.

You don't need to derive the calculus yourself — just trust what the gradients mean:

- `dw` = "if I nudge w up a little, how much does the cost change?"
- `db` = "if I nudge b up a little, how much does the cost change?"

**Watch before writing the loop:** YouTube — StatQuest, *"Gradient Descent, Step-by-Step."* This is the part almost everyone finds confusing the first time. Watch it slowly, pause if you need to.

### 6. The gradient descent loop — where the learning happens

```python
n = len(x)
w = 0.0
b = 0.0
learning_rate = 0.0005
iterations = 2000
cost_history = []

for i in range(iterations):
    predictions = predict(x, w, b)
    errors = predictions - y

    dw = (2/n) * np.sum(errors * x)
    db = (2/n) * np.sum(errors)

    w = w - learning_rate * dw
    b = b - learning_rate * db

    cost = compute_cost(x, y, w, b)
    cost_history.append(cost)

    if i % 100 == 0:
        print(f"iteration {i}: cost = {cost:.3f}, w = {w:.3f}, b = {b:.3f}")

print(f"\nfinal: w = {w:.3f}, b = {b:.3f}")
```

- `n = len(x)` — how many data points you have (50).
- `w = 0.0`, `b = 0.0` — start with the worst possible guess: a flat, useless line. You have to start somewhere.
- `learning_rate = 0.0005` — how big a step to take downhill each time. Too big and you overshoot the bottom; too small and it takes forever.
- `iterations = 2000` — how many times you repeat the "step downhill" process.
- `cost_history = []` — a diary of the cost at every step, so you can plot it later.
- `for i in range(iterations):` — repeat everything below 2000 times.
- `dw` / `db` — the actual gradient formulas. `(2/n) * np.sum(errors * x)` and `(2/n) * np.sum(errors)` compute, in one shot across all 50 points, which direction lowers the cost.
- `w = w - learning_rate * dw` and `b = b - learning_rate * db` — the actual step downhill. Subtracting moves `w` and `b` in the direction that *reduces* cost. `learning_rate` controls how big that nudge is.
- `cost_history.append(cost)` — record today's score.
- `if i % 100 == 0:` — only print every 100th iteration (`%` gives the remainder of division), so your terminal isn't flooded with 2000 lines.

By the end, `w` should drift toward **2.0** and `b` toward **3.0** — the real relationship the fake data was built from in step 2.

### 7. Watch the cost drop

```python
plt.plot(cost_history)
plt.title("Cost over time — this should go down and flatten out")
plt.xlabel("iteration")
plt.ylabel("cost")
plt.show()
```

A healthy graph starts high and drops fast, then flattens near the bottom. If cost goes **up** or turns into `nan`, your learning rate is too big — try `0.0001` and rerun.

### 8. Plot your line against the real data

```python
plt.scatter(x, y, label="actual data")
plt.plot(x, predict(x, w, b), color="red", label="your line")
plt.title("Your line vs. the real data")
plt.legend()
plt.show()
```

Uses your `predict` function with the *final*, trained `w` and `b` to draw a red line over the scatter of real points. If the red line cuts through the middle of the dots, your gradient descent worked. This is the most satisfying way to check your work — numbers on a screen don't mean much until you see it visually.

### 9. Check yourself against sklearn

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x.reshape(-1, 1), y)

print(f"sklearn says: w = {model.coef_[0]:.3f}, b = {model.intercept_:.3f}")
print(f"you got:      w = {w:.3f}, b = {b:.3f}")
```

`sklearn`'s `LinearRegression` finds the mathematically exact best-fit line instantly, no guessing loop needed for a case this simple. `x.reshape(-1, 1)` just reformats your 1D list of x-values into the column shape sklearn expects (`-1` means "figure out the row count automatically").

If your from-scratch `w`/`b` land close to sklearn's, that's proof your code is correct. They don't need to match exactly — close within a few decimal places is a win. Wildly different means go back to the loop and try more iterations or a smaller learning rate.

## Full concept summary, one sentence each

1. **Make fake data** you already know the answer to, so you can check your work.
2. **Build a formula** (`predict`) that turns a guess (`w`, `b`) into a line.
3. **Build a way to measure wrongness** (`compute_cost`) — the average squared distance between your line and the real points.
4. **Repeatedly nudge the guess** (gradient descent loop) in the direction that lowers the wrongness.
