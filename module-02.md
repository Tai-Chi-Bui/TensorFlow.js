```markdown
# Module 2: Your First Real TensorFlow.js Model  
## From Fake Rules to Real Learning (Beginner-Friendly, Step-by-Step)

> **Target Audience**: You completed **Module 1** — you know what ML is, ran the fake house predictor, and have **basic JS + browser/Node setup**.  
> **Zero math**. **Zero prior ML**. Just **code, visuals, and fun**.

---

## Module Overview

| Section | Time | Goal |
|-------|------|------|
| 2.1 Quick Recap & What’s New | 5 min | Connect Module 1 → 2 |
| 2.2 Install TensorFlow.js | 10 min | CDN + npm setup |
| 2.3 Meet Tensors: JS Arrays on Steroids | 15 min | Understand data in TF.js |
| 2.4 Build Your First Model: `y = 2x - 1` | 20 min | Full code + live demo |
| 2.5 Train It! (`model.fit`) | 20 min | Watch the computer **learn** |
| 2.6 Predict & Test | 15 min | Use the trained model |
| 2.7 Visualize Training (Chart.js) | 15 min | See loss drop in real-time |
| 2.8 Node.js Version | 15 min | Run same model in terminal |
| 2.9 Debug Common Errors | 10 min | Fix shape mismatches |
| 2.10 Quiz & Project | 15 min | Solidify knowledge |
| **Total** | **~2.5 hours** | You’ll train a **real ML model** in JS! |

---

## 2.1 Quick Recap: From Fake to Real

| Module 1 (Fake ML) | Module 2 (Real ML) |
|--------------------|--------------------|
| You **wrote** the rule: `price = size × 120 + 5000` | The **computer learns** the rule from data |
| Hardcoded function | `model.fit()` auto-adjusts |
| No training | 250 training steps (`epochs`) |

**Today’s Goal**:  
Teach a model to learn:  
> **y = 2x − 1**  
> (e.g., x=5 → y=9)

---

## 2.2 Install TensorFlow.js (2 Ways)

### Option 1: Browser (Easiest)

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
```

> Just add this line to any HTML file.

---

### Option 2: Node.js (For APIs, CLI)

```bash
mkdir tfjs-module2
cd tfjs-module2
npm init -y
npm install @tensorflow/tfjs-node
```

> Use `tfjs-node` for CPU/GPU acceleration in Node.

---

## 2.3 Meet Tensors: Data in TensorFlow.js

### Think: **Enhanced JavaScript Arrays**

| JS Array | Tensor |
|--------|--------|
| `let arr = [1, 2, 3];` | `let t = tf.tensor([1, 2, 3]);` |
| Manual loops | Auto vectorized (fast!) |
| No shape info | Knows `.shape`, `.dtype` |

### Try in Browser Console

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
<script>
  const t = tf.tensor2d([[1, 2], [3, 4]]);
  t.print(); // [[1, 2], [3, 4]]
  console.log(t.shape); // [2, 2]
</script>
```

**Why 2D?**  
Because ML loves **tables**:
```js
xs = [[1], [2], [3]]  // Input
ys = [[1], [3], [5]]  // Output (2*1-1=1, 2*2-1=3, etc.)
```

---

## 2.4 Build Your First Model: `y = 2x - 1`

### Full Working HTML File

```html
<!DOCTYPE html>
<html>
<head>
  <title>Module 2: My First Real Model</title>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { font-family: Arial; padding: 20px; max-width: 800px; margin: auto; }
    button, input { padding: 10px; font-size: 16px; margin: 5px; }
    #output, #chartContainer { margin: 20px 0; }
    canvas { border: 1px solid #ddd; border-radius: 8px; }
    .code { background: #f4f4f4; padding: 15px; border-radius: 8px; font-family: monospace; }
  </style>
</head>
<body>

  <h1>🧠 My First ML Model: Learn y = 2x − 1</h1>
  <p><strong>Goal</strong>: Train a model to predict output from input.</p>

  <div class="code">
    <strong>Training Data (Examples):</strong><br>
    x = [1, 2, 3, 4] → y = [1, 3, 5, 7] <br>
    <small>Because: 2×1−1=1, 2×2−1=3, etc.</small>
  </div>

  <button onclick="trainModel()">Train Model (250 steps)</button>
  <button onclick="predict()">Test Prediction</button>
  <input type="number" id="testX" placeholder="Enter x (e.g. 5)" />
  <div id="output"></div>

  <h2>Training Progress (Loss)</h2>
  <div id="chartContainer"><canvas id="lossChart"></canvas></div>

  <script>
    let model;
    let lossValues = [];

    // Step 1: Create the model
    async function createModel() {
      model = tf.sequential();
      
      // One dense layer: 1 input → 1 output
      model.add(tf.layers.dense({
        units: 1,           // 1 neuron
        inputShape: [1]     // 1 input value (x)
      }));

      // Optimizer + Loss
      model.compile({
        optimizer: 'sgd',   // Stochastic Gradient Descent
        loss: 'meanSquaredError'
      });

      document.getElementById('output').innerHTML = 
        '<span style="color:green">Model created! Click "Train Model"</span>';
    }

    // Step 2: Train the model
    async function trainModel() {
      await createModel();
      document.getElementById('output').innerHTML = 'Training...';

      // Training data
      const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
      const ys = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

      lossValues = [];

      // Train for 250 epochs
      for (let i = 0; i < 250; i++) {
        const response = await model.fit(xs, ys, { epochs: 1 });
        const loss = response.history.loss[0];
        lossValues.push({ x: i + 1, y: loss });

        // Update UI every 25 steps
        if ((i + 1) % 25 === 0) {
          document.getElementById('output').innerHTML = 
            `Training... Epoch ${i + 1}/250<br>Loss: ${loss.toFixed(6)}`;
          updateChart();
        }
      }

      document.getElementById('output').innerHTML = 
        '<span style="color:green">Training Complete! Try prediction.</span>';
      updateChart();
    }

    // Step 3: Predict
    function predict() {
      if (!model) {
        alert("Train the model first!");
        return;
      }

      const input = document.getElementById('testX');
      const x = parseFloat(input.value);

      if (isNaN(x)) {
        alert("Enter a number!");
        return;
      }

      const inputTensor = tf.tensor2d([x], [1, 1]);
      const prediction = model.predict(inputTensor);
      const result = prediction.dataSync()[0];

      const correct = 2 * x - 1;
      document.getElementById('output').innerHTML = `
        <strong>Input x = ${x}</strong><br>
        Model predicts: <strong>${result.toFixed(4)}</strong><br>
        Correct answer: <strong>${correct}</strong><br>
        Error: ${Math.abs(result - correct).toFixed(4)}
      `;

      inputTensor.dispose();
      prediction.dispose();
    }

    // Step 4: Visualize loss
    let chart;
    function updateChart() {
      const ctx = document.getElementById('lossChart').getContext('2d');
      const data = lossValues.map(p => p.y);
      const labels = lossValues.map(p => p.x);

      if (chart) chart.destroy();

      chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: labels,
          datasets: [{
            label: 'Training Loss',
            data: data,
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.1)',
            fill: true,
            tension: 0.4
          }]
        },
        options: {
          responsive: true,
          scales: {
            y: { beginAtZero: true, title: { display: true, text: 'Loss' } },
            x: { title: { display: true, text: 'Epoch' } }
          }
        }
      });
    }

    // Auto-create model on load
    window.onload = () => {
      document.getElementById('output').innerHTML = 
        'Click "Train Model" to start learning!';
    };
  </script>

</body>
</html>
```

### Save as `module2-real-model.html` → Open in Chrome

---

## 2.5 What Just Happened? (Breakdown)

| Step | Code | ML Concept |
|------|------|-----------|
| 1. `tf.sequential()` | Create empty model | Like `const app = express()` |
| 2. `model.add(dense)` | Add a layer | Like `app.use(middleware)` |
| 3. `model.compile()` | Choose optimizer & loss | Set training rules |
| 4. `model.fit()` | Train with data | **The magic**: adjusts weights |
| 5. `model.predict()` | Get answer | Use learned rule |

---

## 2.6 Watch the Loss Drop

- **Loss** = How wrong the model is.
- Starts high (~10), drops to **<0.001**.
- **Visual proof** the model is learning!

---

## 2.7 Node.js Version (Same Model, No Browser)

```js
// node-model.js
import * as tf from '@tensorflow/tfjs-node';

async function run() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

  const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
  const ys = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

  console.log('Training...');
  await model.fit(xs, ys, { epochs: 250 });

  const result = model.predict(tf.tensor2d([10], [1, 1]));
  result.print(); // ~19 (2*10 - 1 = 19)
}

run();
```

Run:
```bash
node node-model.js
```

---

## 2.8 Debug Common Errors

| Error | Fix |
|------|-----|
| `Shape mismatch` | Check `inputShape: [1]` and data `[4, 1]` |
| `tf is not defined` | Add CDN or `import * as tf` |
| Memory leak | Always `tensor.dispose()` |

---

## 2.9 Quiz: Test Your Knowledge

1. What does `inputShape: [1]` mean?  
   → One input value per example.

2. What is `loss`?  
   → How wrong the prediction is.

3. What does `model.fit()` do?  
   → Adjusts the model to reduce loss.

4. Can you run this in Node.js?  
   → Yes! With `@tensorflow/tfjs-node`.

---

## 2.10 Mini Project: Predict Temperature

**Task**:  
Train a model to convert **Celsius → Fahrenheit**.

| °C | °F |
|----|----|
| 0  | 32 |
| 10 | 50 |
| 20 | 68 |
| 30 | 86 |

**Hint**:  
```js
const xs = tf.tensor2d([[0], [10], [20], [30]], [4, 1]);
const ys = tf.tensor2d([[32], [50], [68], [86]], [4, 1]);
```

**Bonus**: Add a slider + live prediction.

---

## Your Module 2 Checklist

- [ ] Run `module2-real-model.html`
- [ ] Train model → loss drops below 0.01
- [ ] Predict `x=5` → ~9
- [ ] Run Node.js version
- [ ] Complete temperature project
- [ ] Explain to a friend: “The model learned `y=2x-1` by itself!”

---

## Resources

| Type | Link |
|------|------|
| Official Tutorial | [tensorflow.org/js/tutorials](https://www.tensorflow.org/js/tutorials) |
| Video | [YouTube: First Model (10 min)](https://www.youtube.com/watch?v=0oW4JLP7l8c) |
| Interactive | [TensorFlow.js Playground](https://playground.tensorflow.org) |

---

**You did it!**  
You just trained a **real neural network** in JavaScript.  
No Python. No math. Just **code**.

> **Save this as `MODULE-2.md`**  
> Next: **Module 3** — Tensors, data, and real-world datasets!

---
``` 

**Save the entire content above as `MODULE-2.md`**  
Ready to teach, run, and impress — 100% beginner-friendly, fully interactive, and deeply explained.  
Let the learning continue!