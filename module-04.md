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

```jsx
import * as tf from '@tensorflow/tfjs';

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
```

> **Try it**: Open DevTools → paste → see real data!

---

## 4.3 Full Working Model: Iris Classifier

```jsx
// IrisClassifier.jsx
import React, { useState, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

export default function IrisClassifier() {
  const [model, setModel] = useState(null);
  const [status, setStatus] = useState('Ready');
  const [result, setResult] = useState('');
  const [lossHistory, setLossHistory] = useState([]);
  const [confusionData, setConfusionData] = useState([]);
  const [inputs, setInputs] = useState({ sl: '', sw: '', pl: '', pw: '' });
  const lossChartRef = useRef(null);
  const confusionChartRef = useRef(null);
  const lossChartInstanceRef = useRef(null);
  const confusionChartInstanceRef = useRef(null);
  const modelRef = useRef(null); // Use ref for immediate access
  const dataRef = useRef(null); // Store data for cleanup

  const speciesNames = ['setosa', 'versicolor', 'virginica'];

  // Load and preprocess data
  const loadAndPrepareData = async () => {
    const rawData = await tf.data.csv(
      'https://storage.googleapis.com/learnjs-data/iris/iris.csv',
      { columnConfigs: { species: { isLabel: true } } }
    ).toArray();

    tf.util.shuffle(rawData);

    const features = rawData.map(r => [
      r.sepalLength, r.sepalWidth, r.petalLength, r.petalWidth
    ]);
    const labels = rawData.map(r => speciesNames.indexOf(r.species));

    const xs = tf.tensor2d(features);
    const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), 3);

    const xsMin = xs.min(0);
    const xsMax = xs.max(0);
    const xsNorm = xs.sub(xsMin).div(xsMax.sub(xsMin));

    const split = 120;
    const data = {
      trainXs: xsNorm.slice([0, 0], [split, 4]),
      trainYs: ys.slice([0, 0], [split, 3]),
      testXs: xsNorm.slice([split, 0], [30, 4]),
      testYs: ys.slice([split, 0], [30, 3]),
      xsMin, xsMax
    };
    
    // Cleanup old data
    if (dataRef.current) {
      if (dataRef.current.trainXs) dataRef.current.trainXs.dispose();
      if (dataRef.current.trainYs) dataRef.current.trainYs.dispose();
      if (dataRef.current.testXs) dataRef.current.testXs.dispose();
      if (dataRef.current.testYs) dataRef.current.testYs.dispose();
      if (dataRef.current.xsMin) dataRef.current.xsMin.dispose();
      if (dataRef.current.xsMax) dataRef.current.xsMax.dispose();
    }
    
    dataRef.current = data;
    return data;
  };

  // Build model
  const createModel = () => {
    // Cleanup old model
    if (modelRef.current) {
      modelRef.current.dispose();
    }
    
    const newModel = tf.sequential();
    newModel.add(tf.layers.dense({
      units: 16,
      activation: 'relu',
      inputShape: [4]
    }));
    newModel.add(tf.layers.dense({
      units: 3,
      activation: 'softmax'
    }));

    newModel.compile({
      optimizer: 'adam',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    modelRef.current = newModel; // Store in ref immediately
    setModel(newModel);
    return newModel;
  };

  // Train
  const train = async () => {
    try {
      setStatus('Loading data...');
      const data = await loadAndPrepareData();

      // Get or create model
      let currentModel = modelRef.current;
      if (!currentModel) {
        currentModel = createModel();
      }
      
      setStatus('Training...');
      const newLossHistory = [];

      await currentModel.fit(data.trainXs, data.trainYs, {
        epochs: 100,
        validationData: [data.testXs, data.testYs],
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            newLossHistory.push({
              epoch: epoch + 1,
              loss: logs.val_loss,
              acc: logs.val_acc
            });
            setStatus(
              `Epoch ${epoch + 1}/100\n` +
              `Val Loss: ${logs.val_loss.toFixed(4)} | ` +
              `Val Acc: ${(logs.val_acc * 100).toFixed(1)}%`
            );
            setLossHistory([...newLossHistory]);
            updateCharts([...newLossHistory], []);
          }
        }
      });

      const evalResult = currentModel.evaluate(data.testXs, data.testYs);
      const accuracy = evalResult[1].dataSync()[0];
      setStatus(`Training Done! Test Accuracy: ${(accuracy*100).toFixed(1)}%`);

      // Cleanup eval tensors
      evalResult[0].dispose();
      evalResult[1].dispose();

      const preds = currentModel.predict(data.testXs).argMax(-1);
      const truths = data.testYs.argMax(-1);
      const predArray = await preds.array();
      const truthArray = await truths.array();

      // Cleanup prediction tensors
      preds.dispose();
      truths.dispose();

      const newConfusionData = Array(3).fill().map(() => Array(3).fill(0));
      for (let i = 0; i < 30; i++) {
        newConfusionData[truthArray[i]][predArray[i]]++;
      }

      setConfusionData(newConfusionData);
      updateCharts(newLossHistory, newConfusionData);
    } catch (error) {
      setStatus(`Error: ${error.message}`);
    }
  };

  // Predict
  const handlePredict = () => {
    const currentModel = modelRef.current;
    if (!currentModel) {
      setResult('Train the model first!');
      return;
    }

    const { sl, sw, pl, pw } = inputs;
    const values = [parseFloat(sl), parseFloat(sw), parseFloat(pl), parseFloat(pw)];

    if (values.some(isNaN)) {
      setResult('Fill all fields!');
      return;
    }

    tf.tidy(() => {
      const input = tf.tensor2d([values]);
      const normalized = input.sub([4.3, 2.0, 1.0, 0.1]).div([3.6, 2.4, 6.8, 2.4]);
      const prediction = currentModel.predict(normalized);
      const probs = prediction.dataSync();
      const species = speciesNames[probs.indexOf(Math.max(...probs))];

      setResult(
        `Prediction: ${species.toUpperCase()}\n` +
        `Setosa: ${(probs[0]*100).toFixed(1)}% | ` +
        `Versicolor: ${(probs[1]*100).toFixed(1)}% | ` +
        `Virginica: ${(probs[2]*100).toFixed(1)}%`
      );
    });
  };

  // Charts
  const updateCharts = (lossData = lossHistory, confData = confusionData) => {
    if (lossChartRef.current) {
      if (lossChartInstanceRef.current) {
        lossChartInstanceRef.current.destroy();
      }

      const ctx = lossChartRef.current.getContext('2d');
      lossChartInstanceRef.current = new Chart(ctx, {
        type: 'line',
        data: {
          labels: lossData.map(l => l.epoch),
          datasets: [
            {
              label: 'Validation Loss',
              data: lossData.map(l => l.loss),
              borderColor: 'rgb(255, 99, 132)',
              fill: false
            },
            {
              label: 'Validation Accuracy',
              data: lossData.map(l => l.acc * 100),
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
    }

    if (confusionChartRef.current && confData.length > 0) {
      if (confusionChartInstanceRef.current) {
        confusionChartInstanceRef.current.destroy();
      }

      const ctx = confusionChartRef.current.getContext('2d');
      confusionChartInstanceRef.current = new Chart(ctx, {
        type: 'matrix',
        data: {
          datasets: [{
            label: 'Confusion Matrix',
            data: confData.flatMap((row, i) =>
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
  };

  useEffect(() => {
    return () => {
      if (lossChartInstanceRef.current) lossChartInstanceRef.current.destroy();
      if (confusionChartInstanceRef.current) confusionChartInstanceRef.current.destroy();
      if (modelRef.current) {
        modelRef.current.dispose();
        modelRef.current = null;
      }
      if (dataRef.current) {
        if (dataRef.current.trainXs) dataRef.current.trainXs.dispose();
        if (dataRef.current.trainYs) dataRef.current.trainYs.dispose();
        if (dataRef.current.testXs) dataRef.current.testXs.dispose();
        if (dataRef.current.testYs) dataRef.current.testYs.dispose();
        if (dataRef.current.xsMin) dataRef.current.xsMin.dispose();
        if (dataRef.current.xsMax) dataRef.current.xsMax.dispose();
      }
    };
  }, []);

  const cardStyle = {
    border: '1px solid #ddd',
    padding: '15px',
    borderRadius: '8px',
    flex: 1,
    minWidth: '300px'
  };

  return (
    <div style={{ fontFamily: 'Arial', padding: '20px', maxWidth: '900px', margin: 'auto' }}>
      <h1>🌸 Iris Flower Classifier</h1>
      <p><strong>Predict species</strong> from 4 measurements.</p>

      <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
        <div style={cardStyle}>
          <h3>Train Model</h3>
          <button 
            onClick={train}
            style={{ padding: '10px', fontSize: '16px', margin: '5px' }}
          >
            Start Training (100 epochs)
          </button>
          <div style={{ whiteSpace: 'pre-line' }}>{status}</div>
          <canvas ref={lossChartRef} height={150} style={{ marginTop: '10px' }} />
        </div>

        <div style={cardStyle}>
          <h3>Predict New Flower</h3>
          <input
            type="number"
            value={inputs.sl}
            onChange={(e) => setInputs({...inputs, sl: e.target.value})}
            placeholder="Sepal Length (e.g. 5.1)"
            style={{ padding: '10px', fontSize: '16px', margin: '5px', width: 'calc(100% - 20px)' }}
          />
          <input
            type="number"
            value={inputs.sw}
            onChange={(e) => setInputs({...inputs, sw: e.target.value})}
            placeholder="Sepal Width (e.g. 3.5)"
            style={{ padding: '10px', fontSize: '16px', margin: '5px', width: 'calc(100% - 20px)' }}
          />
          <input
            type="number"
            value={inputs.pl}
            onChange={(e) => setInputs({...inputs, pl: e.target.value})}
            placeholder="Petal Length (e.g. 1.4)"
            style={{ padding: '10px', fontSize: '16px', margin: '5px', width: 'calc(100% - 20px)' }}
          />
          <input
            type="number"
            value={inputs.pw}
            onChange={(e) => setInputs({...inputs, pw: e.target.value})}
            placeholder="Petal Width (e.g. 0.2)"
            style={{ padding: '10px', fontSize: '16px', margin: '5px', width: 'calc(100% - 20px)' }}
          />
          <button 
            onClick={handlePredict}
            style={{ padding: '10px', fontSize: '16px', margin: '5px' }}
          >
            Predict Species
          </button>
          <div style={{ whiteSpace: 'pre-line', marginTop: '10px' }}>{result}</div>
        </div>
      </div>

      <div style={{ ...cardStyle, marginTop: '20px' }}>
        <h3>Confusion Matrix</h3>
        <canvas ref={confusionChartRef} />
      </div>
    </div>
  );
}
```

### Save as `IrisClassifier.jsx` in your React project

**Install dependencies:**
```bash
npm install @tensorflow/tfjs chart.js
```

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
