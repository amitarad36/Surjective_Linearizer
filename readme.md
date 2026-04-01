# Surjective Linearizer

> Extending the Linearizer framework with a surjective encoder — replacing the invertible `g` with a surjective classifier that assumes a pseudo-inverse `g†`.

---

## Overview

The [Linearizer](https://arxiv.org/abs/2510.08570) framework expresses a neural network as:

```
f(x) = g⁻¹ᵧ(A · gₓ(x))
```

where `gₓ` and `gᵧ` are **invertible** networks and `A` is a linear operator. This enables classical linear-algebra tools (SVD, pseudoinverse, projections) to operate over nonlinear mappings.

This project replaces the invertible encoder `gₓ` with the **surjective classifier `g`** from [SPNN](https://arxiv.org/abs/2602.06042). Unlike a bijective encoder, `g` is surjective — it maps images to a lower-dimensional attribute space and admits a Moore-Penrose pseudo-inverse `g'` by construction. The modified formulation becomes:

```
f(x) = g'(A · g(x))
```

---

## Key Idea

| Component | Linearizer | Surjective Linearizer |
|---|---|---|
| Encoder `gₓ` | Invertible (bijective) | Surjective (`g` from SPNN) |
| Decoder `gᵧ⁻¹` | Exact inverse | Moore-Penrose pseudo-inverse `g'` |
| Latent space | Full-dimensional | Compressed attribute space |

---

## Related Work

- **Linearizer:** [Who Said Neural Networks Aren't Linear?](https://arxiv.org/abs/2510.08570)
- **SPNN:** [Pseudo-Invertible Neural Networks](https://arxiv.org/abs/2602.06042)
