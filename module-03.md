# Module 3: Tensors – The Building Blocks of Machine Learning  
## From JavaScript Arrays to Supercharged Data (Zero Math, All Code)

> **Target Audience**: You completed **Module 1 & 2** — you know what ML is, trained a `y=2x-1` model, and used `tf.tensor2d()`.  
> **No math. No confusion.** Just **tensors**, **operations**, and **real JS power**.

---

## Module Overview

| Section | Time | Goal |
|-------|------|------|
| 3.1 What Are Tensors? | 10 min | Understand data in TF.js |
| 3.2 Create Tensors (All Ways) | 15 min | From arrays to images |
| 3.3 Tensor Properties & Debugging | 15 min | `.shape`, `.print()`, `.dataSync()` |
| 3.4 Basic Operations (No Loops!) | 20 min | `add`, `mul`, `matMul` |
| 3.5 Reshape, Slice, Expand | 20 min | Transform data like a pro |
| 3.6 Memory Management | 10 min | Avoid leaks with `.dispose()` |
| 3.7 Real-World: Image as Tensor | 20 min | Load & display a 28x28 digit |
| 3.8 Interactive Playground | 20 min | Live tensor editor |
| 3.9 Node.js Tensor Ops | 15 min | Run in terminal |
| 3.10 Mini Project: Build a Tensor Calculator | 20 min | Your own ops app |
| 3.11 Quiz & Debug | 15 min | Master tensors |
| **Total** | **~3 hours** | You’ll **wield tensors** like a JS ninja! |

---

## 3.1 What Are Tensors? (Think: JS Arrays on Steroids)

| JS Array | Tensor |
|--------|--------|
| `let arr = [1, 2, 3];` | `let t = tf.tensor([1, 2, 3]);` |
| Manual `for` loops | **Vectorized** — 100x faster |
| No shape info | Knows `.shape`, `.rank`, `.dtype` |
| Lives in CPU | Can live in **GPU (WebGL)** |

> **Analogy**:  
> - JS Array = Bicycle  
> - Tensor = Rocket Ship

---

## 3.2 Create Tensors – Every Way

### 1. From Arrays

```js
const scalar = tf.tensor(42);
const vector = tf.tensor([1, 2, 3]);
const matrix = tf.tensor([[1, 2], [3, 4]]);
const tensor3d = tf.tensor([[[1], [2]], [[3], [4]]]);
```

### 2. With `tf.tensor1d`, `2d`, `3d`, `4d`

```js
const t1 = tf.tensor1d([1, 2, 3]);
const t2 = tf.tensor2d([[1, 2], [3, 4]], [2, 2]);
```

### 3. Zeros, Ones, Random

```js
tf.zeros([2, 3]);        // [[0,0,0], [0,0,0]]
tf.ones([3]);            // [1,1,1]
tf.randomNormal([2, 2]); // Random values ~ N(0,1)
tf.randomUniform([3]);   // Random [0,1]
```

---

## 3.3 Tensor Properties & Debugging

```js
const t = tf.tensor2d([[1, 2], [3, 4]]);

console.log(t.shape);     // [2, 2]
console.log(t.rank);      // 2
console.log(t.dtype);     // 'float32'
console.log(t.size);      // 4

t.print();                // Pretty print in console
console.log(t.dataSync()); // [1, 2, 3, 4] → JS array (blocks!)
```

> **Warning**: `dataSync()` **blocks** the GPU → use only for debugging!

---

## 3.4 Basic Operations – No Loops!

### Element-wise

```js
const a = tf.tensor2d([[1, 2], [3, 4]]);
const b = tf.tensor2d([[5, 6], [7, 8]]);

tf.add(a, b).print();      // [[6, 8], [10, 12]]
tf.sub(a, b).print();      // [[-4, -4], [-4, -4]]
tf.mul(a, b).print();      // [[5, 12], [21, 32]]
tf.div(a, b).print();      // [[0.2, 0.33], [0.43, 0.5]]
```

### Math Functions

```js
tf.sqrt(a).print();        // [[1, 1.41], [1.73, 2]]
tf.square(a).print();      // [[1, 4], [9, 16]]
tf.abs(tf.tensor([-1, -2])).print(); // [1, 2]
```

### Matrix Multiplication

```js
const m1 = tf.tensor2d([[1, 2], [3, 4]]); // [2,2]
const m2 = tf.tensor2d([[5], [6]]);      // [2,1]
tf.matMul(m1, m2).print();               // [[17], [39]] → [2,1]
```

---

## 3.5 Reshape, Slice, Expand

```js
const t = tf.tensor([1, 2, 3, 4, 5, 6]);

t.reshape([2, 3]).print();     // [[1,2,3], [4,5,6]]
t.reshape([3, 2]).print();     // [[1,2], [3,4], [5,6]]

t.slice([1], [3]).print();     // [2, 3, 4]
t.expandDims(0).shape;         // [1, 6] → adds batch dim
```

---

## 3.6 Memory Management (Critical!)

```js
const t = tf.tensor([1, 2, 3]);
t.dispose(); // Free GPU memory

// Or use tidy (auto-cleanup)
const result = tf.tidy(() => {
  const a = tf.tensor([1, 2]);
  const b = tf.tensor([3, 4]);
  return tf.add(a, b); // a, b auto-disposed
});
```

> **Rule**: Always `dispose()` or use `tf.tidy()` in production.

---

## 3.7 Real-World: Image as Tensor (MNIST Digit)

```jsx
// ImageTensor.jsx
import React, { useRef, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

export default function ImageTensor() {
  const canvasRef = useRef(null);
  const [tensorOutput, setTensorOutput] = useState('');

  useEffect(() => {
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.scale(10, 10); // 28px → 280px
    }
  }, []);

  const loadDigit = async () => {
    // Load MNIST test data (first 1000)
    const data = await fetch('https://storage.googleapis.com/learnjs-data/mnist-demo/mnist_test.csv')
      .then(r => r.text());
    const lines = data.split('\n').slice(1, 1001);
    const randomLine = lines[Math.floor(Math.random() * lines.length)];
    const values = randomLine.split(',').map(Number);

    const label = values[0];
    const pixels = values.slice(1); // 784 values (28x28)

    // Create 28x28 tensor
    const imageTensor = tf.tensor(pixels, [28, 28]);

    // Draw on canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, 28, 28);
      const imgData = ctx.createImageData(28, 28);
      for (let i = 0; i < pixels.length; i++) {
        const intensity = pixels[i];
        const idx = i * 4;
        imgData.data[idx] = intensity;
        imgData.data[idx+1] = intensity;
        imgData.data[idx+2] = intensity;
        imgData.data[idx+3] = 255;
      }
      ctx.putImageData(imgData, 0, 0);
    }

    // Show tensor
    const slice = imageTensor.slice([0,0], [5,5]);
    const output = 
      `Label: ${label}\nShape: [${imageTensor.shape.join(', ')}]\nFirst 5x5:\n` +
      slice.arraySync()
        .map(row => row.map(v => v.toFixed(0).padStart(3)).join(' '))
        .join('\n');
    
    setTensorOutput(output);
    slice.dispose();
    imageTensor.dispose();
  };

  return (
    <div style={{ textAlign: 'center', fontFamily: 'Arial' }}>
      <h2>28×28 MNIST Digit → Tensor</h2>
      <canvas 
        ref={canvasRef}
        width={280}
        height={280}
        style={{ border: '2px solid #333', imageRendering: 'pixelated' }}
      />
      <br />
      <button 
        onClick={loadDigit}
        style={{ padding: '10px', margin: '10px' }}
      >
        Load Random Digit
      </button>
      <pre style={{ textAlign: 'left', margin: '20px auto', maxWidth: '400px' }}>
        {tensorOutput}
      </pre>
    </div>
  );
}
```

### Save as `ImageTensor.jsx` in your React project

---

## 3.8 Interactive Tensor Playground

```jsx
// TensorPlayground.jsx
import React, { useState } from 'react';
import * as tf from '@tensorflow/tfjs';

export default function TensorPlayground() {
  const [code, setCode] = useState(
    `const a = tf.tensor2d([[1, 2], [3, 4]]);
const b = tf.tensor2d([[5, 6], [7, 8]]);
tf.add(a, b).print();`
  );
  const [output, setOutput] = useState('');

  const run = () => {
    setOutput('Running...\n');
    let outputText = '';

    tf.tidy(() => {
      try {
        // Capture console.log for tensor.print()
        const originalLog = console.log;
        console.log = (msg) => {
          outputText += msg + '\n';
        };

        const result = eval(code);
        
        if (result && typeof result.print === 'function') {
          result.print();
        } else {
          outputText += 'Result: ' + JSON.stringify(result) + '\n';
        }

        console.log = originalLog;
        setOutput(outputText || 'No output');
      } catch (e) {
        setOutput('Error: ' + e.message + '\n');
      }
    });
  };

  return (
    <div style={{ fontFamily: 'monospace', padding: '20px' }}>
      <h2>Live Tensor Editor</h2>
      <textarea
        value={code}
        onChange={(e) => setCode(e.target.value)}
        rows={10}
        placeholder="Enter tensor code..."
        style={{ width: '100%', fontFamily: 'monospace', padding: '10px' }}
      />
      <br />
      <button 
        onClick={run}
        style={{ padding: '10px', margin: '5px' }}
      >
        Run Code
      </button>
      <pre style={{ width: '100%', fontFamily: 'monospace', background: '#f4f4f4', padding: '10px', overflow: 'auto' }}>
        {output}
      </pre>
    </div>
  );
}
```

### Save as `TensorPlayground.jsx` in your React project

---

## 3.9 Node.js Tensor Ops

```js
// node-tensor.js
import * as tf from '@tensorflow/tfjs-node';

const a = tf.tensor2d([[1, 2], [3, 4]]);
const b = tf.tensor2d([[5, 6], [7, 8]]);

tf.add(a, b).print();
tf.matMul(a, b.transpose()).print();

console.log('Shape:', a.shape);
a.dispose();
b.dispose();
```

Run:
```bash
node node-tensor.js
```

---

## 3.10 Mini Project: **Tensor Calculator App**

**Build**: A web app where users:
1. Enter two matrices
2. Choose operation (`+`, `×`, etc.)
3. See result + shape

**Bonus**: Add `reshape` slider.

---

## 3.11 Quiz: Are You a Tensor Master?

1. What does `tf.tensor2d([[1,2],[3,4]], [2,2])` do?  
   → Creates 2x2 tensor from array.

2. How to get JS array from tensor?  
   → `.dataSync()` or `.arraySync()`.

3. Why use `tf.tidy()`?  
   → Auto memory cleanup.

4. What’s `matMul`?  
   → Matrix multiplication.

5. Can you reshape `[1,2,3,4]` → `[2,2]`?  
   → Yes! `tensor.reshape([2,2])`

---

## Your Module 3 Checklist

- [ ] Run image demo → see 28x28 digit
- [ ] Use playground → try 5 ops
- [ ] Run Node.js example
- [ ] Build **Tensor Calculator**
- [ ] Explain: “Tensors are fast, typed, GPU-ready arrays”

---

## Resources

| Type | Link |
|------|------|
| Official Guide | [tensorflow.org/js/guide/tensors](https://www.tensorflow.org/js/guide/tensors) |
| MNIST Demo | [storage.googleapis.com/learnjs-data/mnist-demo](https://storage.googleapis.com/learn...) |
| Video | [YouTube: Tensors in 8 min](https://www.youtube.com/watch?v=5bZ6i3k7f4w) |

---

