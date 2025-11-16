# Module 9: Deployment, Optimization & Node.js  
## From Browser to Production (Ship Your ML App!)

> **Target Audience**: You completed **Modules 1–8** — you built models, used CNNs/RNNs, transfer learning, and real data.  
> **No theory. No fluff.** Just **save, optimize, deploy, serve** — like a real ML engineer.

---

## Module Overview

| Section | Time | Goal |
|-------|------|------|
| 9.1 Save & Load Models | 15 min | `model.save()` in 3 ways |
| 9.2 Quantization: Shrink 10x | 20 min | 90% smaller, same accuracy |
| 9.3 Convert to TensorFlow.js | 20 min | Python → JS |
| 9.4 Optimize for Web (Bundle) | 20 min | Webpack + CDN |
| 9.5 Node.js API with Express | 25 min | `/predict` endpoint |
| 9.6 Deploy to Vercel/Netlify | 20 min | Live in 2 clicks |
| 9.7 Web Workers & Offloading | 20 min | No UI freeze |
| 9.8 Monitoring & Logging | 15 min | Track predictions |
| 9.9 Mini Project: ML Chatbot API | 30 min | Full stack |
| 9.10 Quiz & Checklist | 15 min | Go live! |
| **Total** | **~3.5 hours** | You’ll **ship production ML**! |

---

## 9.1 Save & Load Models (3 Ways)

| Method | Code | Use Case |
|-------|------|--------|
| **LocalStorage** | `await model.save('localstorage://my-model');` | Browser only |
| **IndexedDB** | `await model.save('indexeddb://my-model');` | Large models |
| **Downloads** | `await model.save('downloads://my-model');` | Export file |

### Load Later

```js
const model = await tf.loadLayersModel('localstorage://my-model');
```

---

## 9.2 Quantization: Make Models Tiny

```bash
npm install @tensorflow/tfjs-automl
```

```js
import { quantizeModel } from '@tensorflow/tfjs-automl';

const quantized = await quantizeModel(originalModel);
await quantized.save('downloads://tiny-model');
```

> **Before**: 4.2 MB  
> **After**: **400 KB** → 10x smaller, 2x faster!

---

## 9.3 Convert Python Model → TensorFlow.js

```bash
pip install tensorflowjs
tensorflowjs_converter --input_format=tf_saved_model ./saved_model ./tfjs_model
```

> Now load in JS:
```js
const model = await tf.loadLayersModel('./tfjs_model/model.json');
```

---

## 9.4 Full Working: **Deployable Web App**

```jsx
// PetClassifierProduction.jsx
import React, { useState, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

export default function PetClassifierProduction() {
  const [model, setModel] = useState(null);
  const [status, setStatus] = useState('Loading model...');
  const [result, setResult] = useState('');
  const [preview, setPreview] = useState('');
  const fileInputRef = useRef(null);
  const workerRef = useRef(null);
  const modelRef = useRef(null); // Use ref for immediate access

  // Load model (cached)
  const loadModel = async () => {
    if (modelRef.current) return;
    try {
      const loadedModel = await tf.loadLayersModel('https://your-domain.com/models/pet-model/model.json');
      modelRef.current = loadedModel; // Store in ref immediately
      setModel(loadedModel);
      setStatus('Ready!');
    } catch (err) {
      setStatus('Error loading model: ' + err.message);
    }
  };

  // Predict with Web Worker
  const handlePredict = () => {
    const file = fileInputRef.current?.files[0];
    const currentModel = modelRef.current;
    if (!file || !currentModel) {
      if (!currentModel) {
        alert("Model not loaded!");
      }
      return;
    }

    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = () => {
      setPreview(img.src);

      // Offload to worker
      if (!workerRef.current) {
        workerRef.current = new Worker('/worker.js');
        workerRef.current.onmessage = (e) => {
          const { label, confidence } = e.data;
          setResult(`${label} (${(confidence*100).toFixed(1)}%)`);
        };
        workerRef.current.onerror = (error) => {
          setResult(`Worker error: ${error.message}`);
        };
      }

      tf.tidy(() => {
        const tensor = tf.browser.fromPixels(img)
          .resizeNearestNeighbor([64, 64])
          .toFloat()
          .div(255)
          .expandDims();
        workerRef.current.postMessage({ 
          tensor: tensor.dataSync(), 
          shape: tensor.shape 
        });
      });
    };
    img.onerror = () => {
      setResult('Error loading image');
    };
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setPreview(URL.createObjectURL(file));
    }
  };

  useEffect(() => {
    loadModel();
    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
      if (modelRef.current) {
        modelRef.current.dispose();
        modelRef.current = null;
      }
    };
  }, []);

  return (
    <div style={{ fontFamily: 'Arial', padding: '20px', textAlign: 'center' }}>
      <h1>Pet Classifier (Production)</h1>
      <input 
        ref={fileInputRef}
        type="file" 
        accept="image/*"
        onChange={handleFileChange}
        style={{ margin: '10px' }}
      />
      {preview && (
        <img 
          src={preview} 
          alt="Preview"
          style={{ maxWidth: '300px', margin: '20px auto', borderRadius: '12px', display: 'block' }}
        />
      )}
      <button 
        onClick={handlePredict}
        style={{ padding: '12px 24px', fontSize: '18px', margin: '10px' }}
      >
        Predict
      </button>
      <div style={{ fontSize: '24px', margin: '20px' }}>{result}</div>
      <div>{status}</div>
    </div>
  );
}
```

### `worker.js` (separate file in public folder)

```js
let model;
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest');

async function loadModel() {
  model = await tf.loadLayersModel('https://your-domain.com/models/pet-model/model.json');
}

loadModel();

self.onmessage = async (e) => {
  const { tensor, shape } = e.data;
  const input = tf.tensor(tensor, shape);
  const pred = model.predict(input).dataSync();
  const label = pred[0] > pred[1] ? 'Cat' : 'Dog';
  const confidence = Math.max(...pred);
  self.postMessage({ label, confidence });
  input.dispose();
};
```

### Save as `PetClassifierProduction.jsx` in your React project

**Note**: Place `worker.js` in your `public` folder for Create React App, or configure it appropriately for your build system.

---

## 9.5 Node.js + Express API (Production)

```bash
mkdir ml-api && cd ml-api
npm init -y
npm install express @tensorflow/tfjs-node cors
```

```js
// server.js
import express from 'express';
import cors from 'cors';
import * as tf from '@tensorflow/tfjs-node';

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

let model;
(async () => {
  model = await tf.loadLayersModel('file://./models/pet-model/model.json');
  console.log('Model loaded');
  app.listen(3000, () => console.log('API running on :3000'));
})();

app.post('/predict', async (req, res) => {
  try {
    const { image } = req.body; // base64
    const buffer = Buffer.from(image, 'base64');
    const img = tf.node.decodeImage(buffer);
    const tensor = img.resizeNearestNeighbor([64, 64]).toFloat().div(255).expandDims();
    const pred = model.predict(tensor).dataSync();
    const label = pred[0] > pred[1] ? 'cat' : 'dog';
    res.json({ label, confidence: Math.max(...pred) });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});
```

### Test with `curl`

```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."}'
```

---

## 9.6 Deploy to Vercel (2 Clicks)

1. Push code to GitHub
2. Go to [vercel.com](https://vercel.com) → Import
3. Done! → `https://your-app.vercel.app`

> Works with **Node.js API + static HTML**

---

## 9.7 Web Workers: No UI Freeze

```js
// main.js
const worker = new Worker('predict-worker.js');
worker.postMessage({ imageData });
worker.onmessage = (e) => {
  document.getElementById('result').textContent = e.data.label;
};
```

---

## 9.8 Monitoring & Logging

```js
// Log every prediction
app.post('/predict', async (req, res) => {
  const start = Date.now();
  // ... predict
  console.log(`Prediction: ${label}, Time: ${Date.now() - start}ms`);
  res.json({ label });
});
```

> Add **Sentry**, **LogRocket**, or **Google Analytics**

---

## 9.9 Mini Project: **ML Chatbot API**

**Build**:
- Frontend: HTML + input
- Backend: Node.js + toxicity model
- Deploy: Vercel
- Features: Real-time filtering

```js
// Use @tensorflow-models/toxicity
const model = await toxicity.load();
const result = await model.classify([userInput]);
```

---

## 9.10 Quiz

1. How to save model to file?  
   → `model.save('downloads://name')`

2. What does quantization do?  
   → Reduces model size 5–10x

3. How to deploy to web?  
   → Vercel/Netlify

4. Why use Web Workers?  
   → Prevent UI freeze

---

## Your Module 9 Checklist

- [ ] Save model → reload in new tab
- [ ] Quantize a model → <1MB
- [ ] Build **Node.js API**
- [ ] Deploy to **Vercel**
- [ ] Use **Web Worker**
- [ ] Build **ML Chatbot API**
- [ ] Explain: “From code to live app in 5 min”

---

## Resources

| Type | Link |
|------|------|
| Save/Load Guide | [tensorflow.org/js/guide/save_load](https://www.tensorflow.org/js/guide/save_load) |
| Quantization | [tfjs-automl](https://github.com/tensorflow/tfjs/tree/master/tfjs-automl) |
| Vercel Deploy | [vercel.com/docs](https://vercel.com/docs) |
| Video | [YouTube: Deploy TF.js in 10 min](https://www.youtube.com/watch?v=0oW4JLP7l8c) |

---
