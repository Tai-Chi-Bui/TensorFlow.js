# Module 4: Building & Training Simple Models  
## From Toy Problems to Real-World Data (Step-by-Step, Zero Math)

> **Target Audience**: You completed **Module 1–3** — you understand ML, trained a `y=2x-1` model, and know **tensors + `model.fit()`**.  
> **No math, no jargon**. Just **real datasets**, **practical code**, and **clear explanations**.

---

## Module Overview

| Section | Time | Goal |
|-------|------|------|
| 4.1 Recap & What’s New | 5 min | From toy → real |
| 4.2 The Iris Dataset (Classic ML) | 15 min | Load real data |
| 4.3 Build a Classification Model | 20 min | Multi-class prediction |
| 4.4 Train with Real Data | 20 min | `fit()` + validation |
| 4.5 Evaluate Accuracy | 15 min | How good is it? |
| 4.6 Visualize Results (Confusion Matrix) | 20 min | See mistakes |
| 4.7 Add Layers & Activation Functions | 20 min | Improve accuracy |
| 4.8 Save & Load Your Model | 15 min | Persist your brain |
| 4.9 Node.js + Express API | 20 min | Serve predictions |
| 4.10 Mini Project: Predict Your Own | 20 min | Custom dataset |
| 4.11 Quiz & Debug | 15 min | Master the flow |
| **Total** | **~3 hours** | You’ll build a **real classifier** that works! |

---

## 4.1 Quick Recap: From `y=2x-1` to Real Problems

| Module 2 | Module 4 |
|--------|--------|
| 1 input → 1 output (regression) | 4 inputs → 3 outputs (classification) |
| Fake data | **Real Iris flowers** |
| 1 dense layer | **Multiple layers + softmax** |

**Today’s Goal**:  
Predict **Iris flower species** from 4 measurements.

---

## 4.2 The Iris Dataset (Your First Real Data)

### What is Iris?
- 150 flowers
- 4 features: sepal/petal length & width
- 3 species: Setosa, Versicolor, Virginica

```js
// Example row
{
  sepalLength: 5.1,
  sepalWidth: 3.5,
  petalLength: 1.4,
  petalWidth: 0.2,
  species: "setosa"  // ← label
}
```

---

### Load Data in Browser (No Server!)

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
<script>
// Load from public CSV
const CSV_URL = 'https://storage.googleapis.com/learnjs-data/iris/iris.csv';

async function loadIrisData() {
  const csv = await tf.data.csv(CSV_URL, {
    columnConfigs: { species: { isLabel: true } }
  });

  const data = await csv.toArray();
  console.log("First 3 rows:", data.slice(0, 3));
  return data;
}
</script>
```

> **Try it**: Open DevTools → paste → see real data!

---

## 4.3 Full Working Model: Iris Classifier

```html
<!DOCTYPE html>
<html>
<head>
  <title>Module 4: Iris Flower Classifier</title>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { font-family: Arial; padding: 20px; max-width: 900px; margin: auto; }
    .container { display: flex; gap: 20px; flex-wrap: wrap; }
    .card { border: 1px solid #ddd; padding: 15px; border-radius: 8px; flex: 1; min-width: 300px; }
    button, input { padding: 10px; font-size: 16px; margin: 5px; }
    .code { background: #f8f8f8; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 14px; }
    canvas { margin-top: 10px; }
  </style>
</head>
<body>

  <h1>🌸 Iris Flower Classifier</h1>
  <p><strong>Predict species</strong> from 4 measurements.</p>

  <div class="container">
    <div class="card">
      <h3>Train Model</h3>
      <button onclick="train()">Start Training (100 epochs)</button>
      <div id="status">Ready</div>
      <canvas id="lossChart" height="150"></canvas>
    </div>

    <div class="card">
      <h3>Predict New Flower</h3>
      <input type="number" id="sl" placeholder="Sepal Length (e.g. 5.1)" /><br>
      <input type="number" id="sw" placeholder="Sepal Width (e.g. 3.5)" /><br>
      <input type="number" id="pl" placeholder="Petal Length (e.g. 1.4)" /><br>
      <input type="number" id="pw" placeholder="Petal Width (e.g. 0.2)" /><br>
      <button onclick="predict()">Predict Species</button>
      <div id="result"></div>
    </div>
  </div>

  <div class="card" style="margin-top: 20px;">
    <h3>Confusion Matrix</h3>
    <canvas id="confusionChart"></canvas>
  </div>

  <script>
    let model;
    let speciesNames = ['setosa', 'versicolor', 'virginica'];
    let lossHistory = [];
    let confusionData = [];

    // Load and preprocess data
    async function loadAndPrepareData() {
      const rawData = await tf.data.csv(
        'https://storage.googleapis.com/learnjs-data/iris/iris.csv',
        { columnConfigs: { species: { isLabel: true } } }
      ).toArray();

      // Shuffle
      tf.util.shuffle(rawData);

      const features = rawData.map(r => [
        r.sepalLength, r.sepalWidth, r.petalLength, r.petalWidth
      ]);
      const labels = rawData.map(r => speciesNames.indexOf(r.species));

      const xs = tf.tensor2d(features);
      const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), 3);

      // Normalize features (0 to 1)
      const xsMin = xs.min(0);
      const xsMax = xs.max(0);
      const xsNorm = xs.sub(xsMin).div(xsMax.sub(xsMin));

      // Split: 120 train, 30 test
      const split = 120;
      return {
        trainXs: xsNorm.slice([0, 0], [split, 4]),
        trainYs: ys.slice([0, 0], [split, 3]),
        testXs: xsNorm.slice([split, 0], [30, 4]),
        testYs: ys.slice([split, 0], [30, 3]),
        xsMin, xsMax
      };
    }

    // Build model
    function createModel() {
      model = tf.sequential();
      model.add(tf.layers.dense({
        units: 16,
        activation: 'relu',
        inputShape: [4]
      }));
      model.add(tf.layers.dense({
        units: 3,
        activation: 'softmax'
      }));

      model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });
    }

    // Train
    async function train() {
      document.getElementById('status').innerHTML = 'Loading data...';
      const data = await loadAndPrepareData();

      createModel();
      document.getElementById('status').innerHTML = 'Training...';

      lossHistory = [];
      confusionData = [];

      await model.fit(data.trainXs, data.trainYs, {
        epochs: 100,
        validationData: [data.testXs, data.testYs],
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            lossHistory.push({
              epoch: epoch + 1,
              loss: logs.val_loss,
              acc: logs.val_acc
            });
            document.getElementById('status').innerHTML = 
              `Epoch ${epoch + 1}/100<br>
               Val Loss: ${logs.val_loss.toFixed(4)} | 
               Val Acc: ${(logs.val_acc * 100).toFixed(1)}%`;
            updateCharts();
          }
        }
      });

      // Evaluate test set
      const evalResult = model.evaluate(data.testXs, data.testYs);
      const accuracy = evalResult[1].dataSync()[0];
      document.getElementById('status').innerHTML = 
        `<span style="color:green">Training Done! Test Accuracy: ${(accuracy*100).toFixed(1)}%</span>`;

      // Generate confusion matrix
      const preds = model.predict(data.testXs).argMax(-1);
      const truths = data.testYs.argMax(-1);
      const predArray = await preds.array();
      const truthArray = await truths.array();

      confusionData = Array(3).fill().map(() => Array(3).fill(0));
      for (let i = 0; i < 30; i++) {
        confusionData[truthArray[i]][predArray[i]]++;
      }

      updateCharts();
    }

    // Predict
    function predict() {
      if (!model) {
        alert("Train the model first!");
        return;
      }

      const sl = parseFloat(document.getElementById('sl').value);
      const sw = parseFloat(document.getElementById('sw').value);
      const pl = parseFloat(document.getElementById('pl').value);
      const pw = parseFloat(document.getElementById('pw').value);

      if ([sl, sw, pl, pw].some(isNaN)) {
        alert("Fill all fields!");
        return;
      }

      // Normalize input
      const input = tf.tensor2d([[sl, sw, pl, pw]]);
      const normalized = input.sub([4.3, 2.0, 1.0, 0.1]).div([3.6, 2.4, 6.8, 2.4]);
      const prediction = model.predict(normalized);
      const probs = prediction.dataSync();
      const species = speciesNames[probs.indexOf(Math.max(...probs))];

      document.getElementById('result').innerHTML = `
        <strong>Prediction: <span style="color:#d4a017">${species.toUpperCase()}</span></strong><br>
        <small>
          Setosa: ${(probs[0]*100).toFixed(1)}% | 
          Versicolor: ${(probs[1]*100).toFixed(1)}% | 
          Virginica: ${(probs[2]*100).toFixed(1)}%
        </small>
      `;
    }

    // Charts
    function updateCharts() {
      // Loss & Accuracy
      const ctx1 = document.getElementById('lossChart').getContext('2d');
      new Chart(ctx1, {
        type: 'line',
        data: {
          labels: lossHistory.map(l => l.epoch),
          datasets: [
            {
              label: 'Validation Loss',
              data: lossHistory.map(l => l.loss),
              borderColor: 'rgb(255, 99, 132)',
              fill: false
            },
            {
              label: 'Validation Accuracy',
              data: lossHistory.map(l => l.acc * 100),
              borderColor: 'rgb(75, 192, 192)',
              yAxisID: 'y2',
              fill: false
            }
          ]
        },
        options: {
          responsive: true,
          scales: {
            y: { title: { display: true, text: 'Loss' } },
            y2: { position: 'right', title: { display: true, text: 'Accuracy %' }, max: 100 }
          }
        }
      });

      // Confusion Matrix
      if (confusionData.length > 0) {
        const ctx2 = document.getElementById('confusionChart').getContext('2d');
        new Chart(ctx2, {
          type: 'matrix',
          data: {
            datasets: [{
              label: 'Confusion Matrix',
              data: confusionData.flatMap((row, i) =>
                row.map((val, j) => ({ x: speciesNames[j], y: speciesNames[i], v: val }))
              ),
              backgroundColor: (ctx) => {
                const value = ctx.dataset.data[ctx.dataIndex].v;
                const alpha = value / 10;
                return `rgba(54, 162, 235, ${alpha})`;
              },
              width: ({chart}) => (chart.chartArea.width / 3) * 0.8,
              height: ({chart}) => (chart.chartArea.height / 3) * 0.8
            }]
          },
          options: {
            scales: {
              x: { title: { display: true, text: 'Predicted' } },
              y: { title: { display: true, text: 'Actual' } }
            }
          }
        });
      }
    }
  </script>

</body>
</html>
```

### Save as `module4-iris-classifier.html` → Open in Chrome

---

## 4.4 Key Concepts Explained (No Math!)

| Concept | What It Is | JS Analogy |
|-------|-----------|----------|
| **Classification** | Pick one of many labels | `if/else` but learned |
| **One-Hot Encoding** | `[0,1,0]` = versicolor | Enum → array |
| **Softmax** | Turns scores → probabilities | `Math.max()` + normalize |
| **ReLU** | `max(0, x)` — "on/off switch" | `value > 0 ? value : 0` |
| **Normalization** | Scale 0–1 | CSS `transform: scale()` |
| **Validation** | Hold-out test set | Code review before merge |

---

## 4.5 Save & Load Model

```js
// Save
await model.save('localstorage://iris-model');

// Load later
const loaded = await tf.loadLayersModel('localstorage://iris-model');
```

> Works in browser! No server needed.

---

## 4.6 Node.js + Express API

```js
// server.js
import express from 'express';
import * as tf from '@tensorflow/tfjs-node';

const app = express();
app.use(express.json());

let model;
(async () => {
  model = await tf.loadLayersModel('file://./iris-model/model.json');
  app.listen(3000, () => console.log('API running on :3000'));
})();

app.post('/predict', (req, res) => {
  const { sepalLength, sepalWidth, petalLength, petalWidth } = req.body;
  const input = tf.tensor2d([[sepalLength, sepalWidth, petalLength, petalWidth]]);
  const norm = input.sub([4.3, 2.0, 1.0, 0.1]).div([3.6, 2.4, 6.8, 2.4]);
  const pred = model.predict(norm).argMax(-1).dataSync()[0];
  res.json({ species: ['setosa', 'versicolor', 'virginica'][pred] });
});
```

---

## 4.7 Mini Project: **Predict Your Own Dataset**

### Idea: **Student Pass/Fail Predictor**

| Hours Studied | Sleep Hours | Pass (0/1) |
|---------------|-------------|------------|
| 5             | 8           | 1          |
| 2             | 5           | 0          |

**Task**:
1. Create CSV
2. Train model
3. Build UI
4. Deploy

---

## 4.8 Debug Checklist

| Issue | Fix |
|------|-----|
| `Invalid argument` | Check CSV headers |
| Low accuracy | Add layers or epochs |
| `model is null` | Train before predict |
| Memory leak | `tensor.dispose()` |

---

## 4.9 Quiz

1. What does `softmax` do?  
   → Turns outputs into probabilities.

2. Why normalize data?  
   → Helps model learn faster.

3. What is `validationData`?  
   → Test set during training.

4. How to save a model?  
   → `model.save('localstorage://name')`

---

## Your Module 4 Checklist

- [ ] Run Iris classifier → >90% accuracy
- [ ] Predict 3 flowers
- [ ] Save model → reload
- [ ] Run Express API
- [ ] Build **your own** predictor
- [ ] Explain: “My model learned to classify from examples!”

---

## Resources

| Type | Link |
|------|------|
| Dataset | [Iris CSV](https://storage.googleapis.com/learnjs-data/iris/iris.csv) |
| Guide | [TF.js Models](https://www.tensorflow.org/js/guide/models_and_layers) |
| Video | [Iris in 10 min](https://www.youtube.com/watch?v=c8zMVPW9vzg) |

---

**You’re a real ML developer now!**  
You trained, evaluated, saved, and served a model.

> **Save as `MODULE-4.md`**  
> Next: **Module 5** — Neural network architecture deep dive!

---