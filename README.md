# GPT-2 From Scratch

A C++ implementation of GPT-2 components built from scratch, with the goal of understanding how the underlying architecture works without relying on high-level machine learning frameworks.

The tokenizer, tensor library, and the layer pieces needed for a transformer are in place. Attention, transformer blocks, and a full training loop are not implemented yet.

## Current Progress

### Byte-Level BPE Tokenizer

The tokenizer currently includes:

* UTF-8 ↔ `std::wstring` conversion
* UTF-8 byte-level tokenization
* BPE vocabulary training
* Pair-frequency statistics
* BPE merge operations
* Token encoding
* Token decoding
* GPT-4 style pre-tokenization using `std::wregex`
* The same pre-tokenization during training and encoding
* Reading and writing UTF-8 text
* Saving and loading merge rules

The tokenizer learns new token IDs starting from `256`, since the first 256 token IDs represent raw byte values. Training and encoding both split text into the same regex chunks before applying merges, so a merge never crosses a pre-token boundary.

Merge files are a custom integer format, one rule per line: `left right new_id`. This is not the official GPT-2 string merge format.

### File I/O

Basic file utilities are implemented for:

* Reading a UTF-8 text file, including newline characters
* Writing a human-readable dump of learned merges
* Saving and loading the integer merge list

### Codec

The `Codec` component currently handles:

* UTF-8 encoding and decoding
* Conversion from UTF-8 strings to byte tokens
* BPE pair statistics
* Merging token pairs

### Tensors

`Tensor` supports elementwise arithmetic with broadcasting, matrix multiplication, reductions (`sum`, `max`, `mean`, including `keepdims`), and the usual math operations (`exp`, `log`, `sqrt`, `pow`).

### Layers

Each layer implements `forward` and `backward`. Layers with weights also expose `parameters()`, `gradients()`, and `zeroGrad()`.

* `Embedding` maps token ids to vectors and scatters the gradient back into the matching rows
* `Linear` accepts a matrix or a sequence tensor `[batch, length, features]`
* `LayerNorm` normalizes the last dimension
* `Dropout` uses inverted dropout while training and is an identity at evaluation time
* `ReLU` and `GeLU` (tanh approximation)
* `Softmax` along a chosen axis, with `-1` meaning the last axis
* `Sequential` runs layers in order and sends gradients back in reverse order

### Loss and optimizer

* `CrossEntropyLoss` takes probabilities and a same-shaped target. Its backward pass is the derivative with respect to those probabilities, `-target / output`, so it can be chained through `Softmax`
* `AdamW` updates the tensors returned by `parameters()` and `gradients()`

```cpp
optimizer.step(layer.parameters(), layer.gradients());
```

## Current Pipeline

The tokenizer pipeline is:

```text
Text
  ↓
Pre-tokenization
  ↓
UTF-8
  ↓
Byte-level tokens
  ↓
BPE merges
  ↓
Token IDs
```

Decoding reverses this process:

```text
Token IDs
  ↓
Expand BPE tokens
  ↓
UTF-8 bytes
  ↓
UTF-8 → Unicode text
  ↓
Text
```

A forward step through the layers that exist today looks like this:

```text
Token IDs
  ↓
Embedding
  ↓
Dropout / LayerNorm / Linear / GeLU / Softmax
  ↓
CrossEntropyLoss
  ↓
backward, in reverse layer order
  ↓
AdamW
```

## Example

The current implementation can train BPE merge rules on a text corpus and then encode and decode new text:

```cpp
Tokenizer tokenizer;

auto texts = FileIO::read("Training Files/romeo_and_juliet.txt");

tokenizer.train(texts, 100);

auto tokens = tokenizer.encode(
    L"Two households, both alike in dignity, "
    L"In fair Verona, where we lay our scene,"
);

auto decoded = tokenizer.decode(tokens);
```

`main.cpp` loads `Training Files/merges.txt` when that file exists. Otherwise it trains on `Training Files/input.txt` and writes `merges.txt` and `tokens.txt`. Paths are relative to the working directory. An older `merges.txt` produced before pre-tokenization was applied during training will not match the current encoder, so delete it and train again after that change.

## Project Status

This project is **work in progress**.

Currently implemented:

* [x] UTF-8 codec
* [x] Byte-level tokenization
* [x] BPE statistics
* [x] BPE merge operations
* [x] BPE training on the same pre-tokens used by encoding
* [x] Token encoding
* [x] Token decoding
* [x] Pre-tokenization
* [x] Text file input/output
* [x] Tokenizer merge persistence
* [x] Tensor implementation, including broadcasting
* [x] Embedding
* [x] Linear layer, including sequence-shaped inputs
* [x] Layer normalization
* [x] Dropout
* [x] ReLU and GeLU
* [x] Softmax
* [x] Sequential container
* [x] Backward passes for the layers above
* [x] Cross-entropy loss
* [x] AdamW
* [x] Self-attention
* [x] Multi-head attention
* [x] Feed-forward block
* [x] Transformer blocks

Planned:

* [ ] GPT-2 model
* [ ] Training loop that ties the tokenizer, model, loss, and optimizer together

## Goal

The main goal of this project is educational: to implement the core components of GPT-2 from the ground up and understand the mathematics and engineering behind modern transformer-based language models.

The implementation is intentionally built step by step rather than relying on frameworks such as PyTorch or TensorFlow.
