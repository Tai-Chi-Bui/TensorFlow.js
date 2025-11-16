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

### Option 1: React Project (Recommended)

```bash
npx create-react-app tfjs-app
cd tfjs-app
npm install @tensorflow/tfjs
```

Then import in your components:
```jsx
import * as tf from '@tensorflow/tfjs';
```

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

```jsx
import * as tf from '@tensorflow/tfjs';

// In your React component or console
const t = tf.tensor2d([[1, 2], [3, 4]]);
t.print(); // [[1, 2], [3, 4]]
console.log(t.shape); // [2, 2]
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

```jsx
// FirstModel.jsx
import React, { useState, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

export default function FirstModel() {
  const [model, setModel] = useState(null);
  const [lossValues, setLossValues] = useState([]);
  const [output, setOutput] = useState('Click "Train Model" to start learning!');
  const [testX, setTestX] = useState('');
  const [isTraining, setIsTraining] = useState(false);
  const chartRef = useRef(null);
  const chartInstanceRef = useRef(null);
  const modelRef = useRef(null); // Use ref for immediate access

  // Step 1: Create the model
  const createModel = () => {
    const newModel = tf.sequential();
    
    // One dense layer: 1 input → 1 output
    newModel.add(tf.layers.dense({
      units: 1,           // 1 neuron
      inputShape: [1]     // 1 input value (x)
    }));

    // Optimizer + Loss
    newModel.compile({
      optimizer: 'sgd',   // Stochastic Gradient Descent
      loss: 'meanSquaredError'
    });

    modelRef.current = newModel; // Store in ref immediately
    setModel(newModel);
    setOutput('Model created! Click "Train Model"');
    return newModel;
  };

  // Step 2: Train the model
  const trainModel = async () => {
    // Get or create model
    let currentModel = modelRef.current;
    if (!currentModel) {
      currentModel = createModel();
    }
    
    setIsTraining(true);
    setOutput('Training...');

    // Training data
    const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
    const ys = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

    const newLossValues = [];

    try {
      // Train for 250 epochs
      for (let i = 0; i < 250; i++) {
        const response = await currentModel.fit(xs, ys, { epochs: 1 });
        const loss = response.history.loss[0];
        newLossValues.push({ x: i + 1, y: loss });

        // Update UI every 25 steps
        if ((i + 1) % 25 === 0) {
          setOutput(`Training... Epoch ${i + 1}/250\nLoss: ${loss.toFixed(6)}`);
          setLossValues([...newLossValues]);
          updateChart([...newLossValues]);
        }
      }

      setOutput('Training Complete! Try prediction.');
      setLossValues(newLossValues);
      updateChart(newLossValues);
    } catch (error) {
      setOutput(`Error during training: ${error.message}`);
    } finally {
      xs.dispose();
      ys.dispose();
      setIsTraining(false);
    }
  };

  // Step 3: Predict
  const handlePredict = () => {
    const currentModel = modelRef.current;
    if (!currentModel) {
      setOutput('Train the model first!');
      return;
    }

    const x = parseFloat(testX);

    if (isNaN(x)) {
      setOutput('Enter a number!');
      return;
    }

    tf.tidy(() => {
      const inputTensor = tf.tensor2d([x], [1, 1]);
      const prediction = currentModel.predict(inputTensor);
      const result = prediction.dataSync()[0];

      const correct = 2 * x - 1;
      setOutput(
        `Input x = ${x}\n` +
        `Model predicts: ${result.toFixed(4)}\n` +
        `Correct answer: ${correct}\n` +
        `Error: ${Math.abs(result - correct).toFixed(4)}`
      );
    });
  };

  // Step 4: Visualize loss
  const updateChart = (data = lossValues) => {
    if (!chartRef.current) return;

    const ctx = chartRef.current.getContext('2d');
    const chartData = data.map(p => p.y);
    const labels = data.map(p => p.x);

    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
    }

    chartInstanceRef.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Training Loss',
          data: chartData,
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
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
      }
      if (modelRef.current) {
        modelRef.current.dispose();
        modelRef.current = null;
      }
    };
  }, []);

  return (
    <div style={{ fontFamily: 'Arial', padding: '20px', maxWidth: '800px', margin: 'auto' }}>
      <h1>🧠 My First ML Model: Learn y = 2x − 1</h1>
      <p><strong>Goal</strong>: Train a model to predict output from input.</p>

      <div style={{ 
        background: '#f4f4f4', 
        padding: '15px', 
        borderRadius: '8px', 
        fontFamily: 'monospace' 
      }}>
        <strong>Training Data (Examples):</strong><br />
        x = [1, 2, 3, 4] → y = [1, 3, 5, 7] <br />
        <small>Because: 2×1−1=1, 2×2−1=3, etc.</small>
      </div>

      <button 
        onClick={trainModel}
        disabled={isTraining}
        style={{ padding: '10px', fontSize: '16px', margin: '5px' }}
      >
        {isTraining ? 'Training...' : 'Train Model (250 steps)'}
      </button>
      <button 
        onClick={handlePredict}
        style={{ padding: '10px', fontSize: '16px', margin: '5px' }}
      >
        Test Prediction
      </button>
      <input 
        type="number" 
        value={testX}
        onChange={(e) => setTestX(e.target.value)}
        placeholder="Enter x (e.g. 5)"
        style={{ padding: '10px', fontSize: '16px', margin: '5px' }}
      />
      <div style={{ margin: '20px 0', whiteSpace: 'pre-line' }}>{output}</div>

      <h2>Training Progress (Loss)</h2>
      <div style={{ margin: '20px 0' }}>
        <canvas 
          ref={chartRef}
          style={{ border: '1px solid #ddd', borderRadius: '8px' }}
        />
      </div>
    </div>
  );
}
```

### Save as `FirstModel.jsx` in your React project

**Install dependencies:**
```bash
npm install @tensorflow/tfjs chart.js
```

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

