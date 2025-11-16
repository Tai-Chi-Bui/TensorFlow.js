# Module 8: Pre-trained Models & Transfer Learning  
## From Scratch to Pro in One Click (Real-World AI, Zero Training from Zero)

> **Target Audience**: You completed **Modules 1–7** — you built models, trained on MNIST, handled real data, and used CNNs/RNNs.  
> **No math. No retraining.** Just **download a brain**, **swap the head**, and **use it**.

---

## Module Overview

| Section | Time | Goal |
|-------|------|------|
| 8.1 Why Pre-trained Models? | 10 min | 1000x faster |
| 8.2 Meet TensorFlow Hub Models | 15 min | `tf.loadLayersModel` |
| 8.3 MobileNet: Image Classifier | 25 min | 1000 classes |
| 8.4 Transfer Learning: Fine-tune | 30 min | Cats vs Dogs in 5 min |
| 8.5 Pose Detection (BlazePose) | 25 min | Live webcam |
| 8.6 Text Toxicity & Sentiment | 20 min | NLP in browser |
| 8.7 Save & Deploy Custom Head | 20 min | Export your model |
| 8.8 Node.js + Express API | 20 min | Serve predictions |
| 8.9 Mini Project: Pet Breed Classifier | 30 min | Your own transfer |
| 8.10 Quiz & Debug | 15 min | Master transfer |
| **Total** | **~4 hours** | You’ll **use AI models** like `npm install`! |

---

## 8.1 Why Pre-trained Models? (The Cheat Code)

| From Scratch | Transfer Learning |
|------------|-------------------|
| 10,000 images | 100 images |
| 10 hours training | 5 minutes |
| 70% accuracy | **95%+ accuracy** |

> **Think**:  
> - Pre-trained = A PhD in ImageNet (1.4M images)  
> - You = Add a **custom head** for your task

---

## 8.2 Load from TensorFlow Hub

```jsx
import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
```

```js
const mobilenet = await tf.loadGraphModel(
  'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/classification/5/default/1',
  { fromTFHub: true }
);
```

---

## 8.3 Full Working: **MobileNet Image Classifier**

```jsx
// MobileNetClassifier.jsx
import React, { useState, useRef, useEffect } from 'react';
import * as mobilenet from '@tensorflow-models/mobilenet';

export default function MobileNetClassifier() {
  const [model, setModel] = useState(null);
  const [result, setResult] = useState('');
  const [liveResult, setLiveResult] = useState('');
  const [preview, setPreview] = useState('');
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const fileInputRef = useRef(null);
  const webcamRef = useRef(null);
  const webcamIntervalRef = useRef(null);
  const modelRef = useRef(null); // Use ref for immediate access

  // Load MobileNet
  const loadModel = async () => {
    if (modelRef.current) return;
    setResult('Loading MobileNet...');
    try {
      const loadedModel = await mobilenet.load();
      modelRef.current = loadedModel; // Store in ref immediately
      setModel(loadedModel);
      setResult('Model loaded!');
    } catch (error) {
      setResult(`Error loading model: ${error.message}`);
    }
  };

  // Classify uploaded image
  const handleClassify = async () => {
    await loadModel();
    const currentModel = modelRef.current;
    if (!currentModel) {
      alert("Model not loaded!");
      return;
    }

    const file = fileInputRef.current?.files[0];
    if (!file) {
      alert("Upload an image!");
      return;
    }

    const img = new Image();
    img.src = URL.createObjectURL(file);
    await img.decode();
    setPreview(img.src);

    try {
      const predictions = await currentModel.classify(img);
      const top3 = predictions.slice(0, 3);

      setResult(
        `Top 3 Predictions:\n` +
        top3.map(p => `• ${p.className} (${(p.probability*100).toFixed(1)}%)`).join('\n')
      );
    } catch (error) {
      setResult(`Error: ${error.message}`);
    }
  };

  // Webcam
  const startWebcam = async () => {
    await loadModel();
    const currentModel = modelRef.current;
    if (!currentModel) {
      alert("Model not loaded!");
      return;
    }

    const video = webcamRef.current;
    if (!video) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
      setIsWebcamActive(true);

      webcamIntervalRef.current = setInterval(async () => {
        if (video.readyState === 4 && modelRef.current) {
          try {
            const preds = await modelRef.current.classify(video);
            const top = preds[0];
            setLiveResult(`${top.className} (${(top.probability*100).toFixed(1)}%)`);
          } catch (error) {
            console.error('Classification error:', error);
          }
        }
      }, 1000);
    } catch (err) {
      alert('Error accessing webcam: ' + err.message);
    }
  };

  const stopWebcam = () => {
    if (webcamIntervalRef.current) {
      clearInterval(webcamIntervalRef.current);
      webcamIntervalRef.current = null;
    }
    const video = webcamRef.current;
    if (video?.srcObject) {
      video.srcObject.getTracks().forEach(t => t.stop());
    }
    setIsWebcamActive(false);
    setLiveResult('');
  };

  // Preview
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setPreview(URL.createObjectURL(file));
    }
  };

  useEffect(() => {
    return () => {
      stopWebcam();
      // Note: MobileNet models don't need dispose() - they're pre-trained and managed by the library
    };
  }, []);

  const cardStyle = {
    border: '1px solid #ddd',
    padding: '20px',
    borderRadius: '12px',
    margin: '20px 0'
  };

  return (
    <div style={{ fontFamily: 'Arial', padding: '20px', maxWidth: '900px', margin: 'auto' }}>
      <h1>MobileNet: 1000-Class Image Classifier</h1>
      <p>Pre-trained on <strong>ImageNet</strong>. No training needed!</p>

      <div style={cardStyle}>
        <h3>Upload Image</h3>
        <input 
          ref={fileInputRef}
          type="file" 
          accept="image/*"
          onChange={handleFileChange}
          style={{ margin: '10px 0' }}
        />
        {preview && (
          <img 
            src={preview} 
            alt="Preview"
            style={{ maxWidth: '100%', borderRadius: '8px', margin: '10px 0' }}
          />
        )}
        <button 
          onClick={handleClassify}
          style={{ padding: '12px 20px', fontSize: '16px', margin: '5px' }}
        >
          Classify with MobileNet
        </button>
        <div style={{ marginTop: '15px', fontSize: '18px', whiteSpace: 'pre-line', background: '#f0f8ff', padding: '10px', borderRadius: '8px' }}>
          {result}
        </div>
      </div>

      <div style={cardStyle}>
        <h3>Live Webcam</h3>
        <video 
          ref={webcamRef}
          autoPlay
          playsInline
          width={640}
          height={480}
          style={{ display: isWebcamActive ? 'block' : 'none' }}
        />
        <button 
          onClick={startWebcam}
          style={{ padding: '12px 20px', fontSize: '16px', margin: '5px' }}
        >
          Start Webcam
        </button>
        <button 
          onClick={stopWebcam}
          style={{ padding: '12px 20px', fontSize: '16px', margin: '5px' }}
        >
          Stop
        </button>
        <div style={{ marginTop: '15px', fontSize: '18px' }}>
          {liveResult && <strong>Live: {liveResult}</strong>}
        </div>
      </div>
    </div>
  );
}
```

### Save as `MobileNetClassifier.jsx` in your React project

**Install dependencies:**
```bash
npm install @tensorflow-models/mobilenet
```

---

## 8.4 Transfer Learning: **Cats vs Dogs in 5 Minutes**

```js
// Step 1: Load MobileNet (freeze base)
const base = await mobilenet.load();
const x = base.model.getLayer('global_average_pooling2d_1').output;

// Step 2: Add custom head
const prediction = tf.layers.dense({
  units: 2,
  activation: 'softmax',
  name: 'custom_head'
}).apply(x);

const transferModel = tf.model({ inputs: base.model.inputs, outputs: prediction });

// Step 3: Freeze base layers
for (const layer of transferModel.layers) {
  if (layer.name !== 'custom_head') {
    layer.trainable = false;
  }
}

// Step 4: Compile & train on your data
transferModel.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy' });
await transferModel.fitDataset(yourDataset, { epochs: 5 });
```

> **Result**: 95%+ accuracy with **100 images**!

---

## 8.5 Live Pose Detection with **BlazePose**

```jsx
import * as poseDetection from '@tensorflow-models/pose-detection';

// In your React component
const [detector, setDetector] = useState(null);
const webcamRef = useRef(null);

const startPose = async () => {
  const newDetector = await poseDetection.createDetector(
    poseDetection.SupportedModels.BlazePose
  );
  setDetector(newDetector);
  
  const interval = setInterval(async () => {
    if (webcamRef.current && newDetector) {
      const poses = await newDetector.estimatePoses(webcamRef.current);
      drawKeypoints(poses[0]?.keypoints);
    }
  }, 100);
  
  return () => clearInterval(interval);
};
```

> **Try it**: Real-time skeleton tracking!

---

## 8.6 Text Models: Toxicity & Sentiment

```jsx
import * as toxicity from '@tensorflow-models/toxicity';

// In your React component
const [toxicityModel, setToxicityModel] = useState(null);

useEffect(() => {
  const loadModel = async () => {
    const model = await toxicity.load();
    setToxicityModel(model);
  };
  loadModel();
}, []);

// Use it
const sentences = ['I hate you'];
const predictions = await toxicityModel?.classify(sentences);
console.log(predictions);
```

---

## 8.7 Save Custom Transfer Model

```js
await transferModel.save('localstorage://cats-dogs-model');
```

---

## 8.8 Node.js + Express API

```js
// server.js
import express from 'express';
import * as tf from '@tensorflow/tfjs-node';
import * as mobilenet from '@tensorflow-models/mobilenet';

const app = express();
let model;

(async () => {
  model = await mobilenet.load();
  app.listen(3000, () => console.log('API ready'));
})();

app.post('/classify', async (req, res) => {
  const buffer = req.body.image; // from multer
  const img = tf.node.decodeImage(buffer);
  const preds = await model.classify(img);
  res.json(preds);
});
```

---

## 8.9 Mini Project: **Pet Breed Classifier**

**Task**:
1. Use MobileNet base
2. Add head for **Pug vs Beagle**
3. Train on 50 images
4. Build UI

---

## 8.10 Quiz

1. What is **transfer learning**?  
   → Reuse pre-trained model, train only last layer.

2. How to freeze layers?  
   → `layer.trainable = false`

3. Name 3 pre-trained models.  
   → MobileNet, BlazePose, Toxicity

4. Can you run in Node.js?  
   → Yes! With `tfjs-node`

---

## Your Module 8 Checklist

- [ ] Classify 5 images with MobileNet
- [ ] Live webcam classification
- [ ] Build **Cats vs Dogs transfer model**
- [ ] Run **BlazePose** on webcam
- [ ] Build **Pet Breed Classifier**
- [ ] Explain: “Pre-trained = 1000x faster ML”

---

## Resources

| Type | Link |
|------|------|
| MobileNet | [tfjs-models/mobilenet](https://github.com/tensorflow/tfjs-models/tree/master/mobilenet) |
| BlazePose | [tfjs-models/pose-detection](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection) |
| Transfer Guide | [tensorflow.org/js/tutorials/transfer](https://www.tensorflow.org/js/tutorials/transfer) |
| Video | [YouTube: Transfer Learning in 10 min](https://www.youtube.com/watch?v=QfNvhPx5Px8) |

---

