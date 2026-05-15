# Pretraining Analysis: Initialization Study (MLP)

## Title

**Pretraining Analysis of a Small-Scale Multi-Layer Perceptron**

---

## 1. Objective

To study how different initialization schemes influence the linear algebraic properties of a neural network before training. The focus is on:

* Kaiming initialization
* Xavier initialization
* Standard Normal initialization

We analyze:

* Singular value spectra
* Weight distributions
* Weight matrices (W)
* Gram matrices (W^\top W)

---

## 2. Experimental Setup

**Model:** 3-layer MLP
**Layer dimensions:** (784 → 100) → (100 → 50) → (50 → 10)
**Activation:** ReLU
**Framework:** PyTorch

**Initializations used:**

* **Kaiming:** fan-in based
* **Xavier:** variance scaled (fan-based)
* **Normal:** ( \mathcal{N}(0, 0.02^2) )

---

## 3. Observations

### 3.1 Weight Distribution

* **Kaiming:** Approximately Gaussian. Layer 1 weights mostly in [-0.2, 0.2], mean ≈ 0. Later layers remain normally shaped with slight outliers.
* **Xavier:** Similar bell shape to Kaiming but slightly denser around zero due to lower variance.
* **Normal:** Narrower spread (≈ [-0.075, 0.075]) and higher concentration near zero because variance is fixed and not scaled by fan-in.

---

### 3.2 Singular Value Analysis

Each layer has 100 singular values in Layer 1 (min(784,100)), then 50 and 10 respectively in later layers.

* **Kaiming:**
  Singular values mostly > 1.8.
  Layer 2 peaks around ~2.25, Layer 3 returns to ~1.8.
  This matches the theoretical estimate:
  ( s_{\max} \approx \sqrt{2/n}(\sqrt{m}+\sqrt{n}) ), which predicts values in the 2–3 range.
  Since analysis is often on (W W^\top), the dominant spectrum shifts slightly upward.

* **Xavier:**
  Similar to Kaiming in Layer 1 but slightly smaller spread.
  Layer 2 peaks near ~2.0, indicating more controlled scaling.

* **Normal:**
  Much smaller singular values: ~0.75 in Layer 1 and ~0.30 in Layer 2.
  This occurs because there is no fan-in scaling, so the matrix contracts space more strongly and reduces spectral magnitude.

---

### 3.3 Matrix Structure

* **Weight matrices (W):**
  Dense, symmetric spread around zero, no visible sparsity.

* **Gram matrices (W^\top W):**
  Symmetric and positive semidefinite.
  Kaiming/Xavier show larger eigenvalue spread; Normal shows compressed spectrum and smaller numerical scale.

---

## 4. Determinant / Log-Det Behaviour

* **Kaiming:** Larger determinants due to larger singular values.
* **Xavier:** Moderate determinant scale.
* **Normal:** Often near zero because small singular values shrink volume significantly.

Determinant size directly reflects how much the layer expands or contracts space.

---

## 5. Comparative Insights

* **Largest spectral norm:** Kaiming

* **Most balanced scaling:** Xavier

* **Smallest spectrum / strongest contraction:** Normal

* **Conditioning:**
  Kaiming and Xavier maintain better spread of singular values.
  Normal tends toward spectral shrinkage.

* **Isotropy:**
  Xavier appears closest to isotropic scaling across directions.

---

## 6. Geometric Interpretation

* Kaiming expands space more strongly, stretching several directions.
* Xavier maintains a more balanced transformation.
* Normal contracts space, reducing signal magnitude across layers.

This directly affects gradient flow and signal preservation.

---

## 7. Key Takeaways

* **Best signal preservation:** Xavier / Kaiming
* **Risk of vanishing:** Normal initialization
* **Risk of explosion:** Mildly present in Kaiming due to larger spectral edge
* **Most numerically stable:** Xavier

---

## 8. Future Extensions

* Track singular values during training
* Compare with orthogonal initialization
* Study Jacobian spectra
* Analyze deeper networks

---

## 9. Appendix

**Layer Shapes:**

* L1: (784, 100)
* L2: (100, 50)
* L3: (50, 10)

**Normal Init Parameters:**

* Mean = 0
* Std = 0.02

**Notes:**
Finite-width effects and randomness explain small variations in spectral values across layers.
