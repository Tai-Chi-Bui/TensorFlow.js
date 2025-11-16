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

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet"></script>
```

```js
const mobilenet = await tf.loadGraphModel(
  'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/classification/5/default/1',
  { fromTFHub: true }
);
```

---

## 8.3 Full Working: **MobileNet Image Classifier**

```html
<!DOCTYPE html>
<html>
<head>
  <title>Module 8: MobileNet Image Classifier</title>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet"></script>
  <style>
    body { font-family: Arial; padding: 20px; max-width: 900px; margin: auto; }
    .card { border: 1px solid #ddd; padding: 20px; border-radius: 12px; margin: 20px 0; }
    img { max-width: 100%; border-radius: 8px; }
    button { padding: 12px 20px; font-size: 16px; margin: 5px; }
    .result { margin-top: 15px; font-size: 18px; }
    .top3 { background: #f0f8ff; padding: 10px; border-radius: 8px; }
  </style>
</head>
<body>

  <h1>MobileNet: 1000-Class Image Classifier</h1>
  <p>Pre-trained on **ImageNet**. No training needed!</p>

  <div class="card">
    <h3>Upload Image</h3>
    <input type="file" id="imageUpload" accept="image/*" />
    <img id="preview" />
    <button onclick="classify()">Classify with MobileNet</button>
    <div id="result" class="result"></div>
  </div>

  <div class="card">
    <h3>Live Webcam</h3>
    <video id="webcam" autoplay playsinline width="640" height="480"></video>
    <button onclick="startWebcam()">Start Webcam</button>
    <button onclick="stopWebcam()">Stop</button>
    <div id="liveResult"></div>
  </div>

  <script>
    let model;
    let webcamInterval;

    // Load MobileNet
    async function loadModel() {
      if (model) return;
      document.getElementById('result').innerHTML = 'Loading MobileNet...';
      model = await mobilenet.load();
      document.getElementById('result').innerHTML = '<span style="color:green">Model loaded!</span>';
    }

    // Classify uploaded image
    async function classify() {
      await loadModel();
      const file = document.getElementById('imageUpload').files[0];
      if (!file) return alert("Upload an image!");

      const img = new Image();
      img.src = URL.createObjectURL(file);
      await img.decode();
      document.getElementById('preview').src = img.src;

      const predictions = await model.classify(img);
      const top3 = predictions.slice(0, 3);

      document.getElementById('result').innerHTML = `
        <div class="top3">
          <strong>Top 3 Predictions:</strong><br>
          ${top3.map(p => `• <strong>${p.className}</strong> (${(p.probability*100).toFixed(1)}%)`).join('<br>')}
        </div>
      `;
    }

    // Webcam
    async function startWebcam() {
      await loadModel();
      const video = document.getElementById('webcam');
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;

      webcamInterval = setInterval(async () => {
        if (video.readyState === 4) {
          const preds = await model.classify(video);
          const top = preds[0];
          document.getElementById('liveResult').innerHTML = `
            <strong>Live:</strong> ${top.className} (${(top.probability*100).toFixed(1)}%)
          `;
        }
      }, 1000);
    }

    function stopWebcam() {
      clearInterval(webcamInterval);
      const video = document.getElementById('webcam');
      if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
      }
      document.getElementById('liveResult').innerHTML = '';
    }

    // Preview
    document.getElementById('imageUpload').addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) {
        document.getElementById('preview').src = URL.createObjectURL(file);
      }
    });
  </script>

</body>
</html>
```

### Save as `module8-mobilenet.html` → Open in Chrome

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

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/blazepose"></script>

<script>
  let detector;
  async function startPose() {
    detector = await poseDetection.createDetector(poseDetection.SupportedModels.BlazePose);
    const video = document.getElementById('webcam');
    setInterval(async () => {
      const poses = await detector.estimatePoses(video);
      drawKeypoints(poses[0]?.keypoints);
    }, 100);
  }
</script>
```

> **Try it**: Real-time skeleton tracking!

---

## 8.6 Text Models: Toxicity & Sentiment

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/toxicity"></script>

<script>
  const toxicity = await toxicity.load();
  const sentences = ['I hate you'];
  const predictions = await toxicity.classify(sentences);
  console.log(predictions);
</script>
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

**You’re an AI engineer now!**  
You **download**, **adapt**, and **deploy** world-class models.

> **Save as `MODULE-8.md`**  
> Next: **Module 9** — Deployment, optimization, and shipping!

---
