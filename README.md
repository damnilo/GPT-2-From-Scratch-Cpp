# GPT-2 From Scratch

A C++ implementation of GPT-2 components built from scratch, with the goal of understanding how the underlying architecture works without relying on high-level machine learning frameworks.

The project is currently focused on building the tokenizer and the supporting infrastructure before implementing the transformer architecture.

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
* Basic GPT-style text pre-tokenization using `std::wregex`
* Support for English alphabet characters, numbers, whitespace, and punctuation
* Reading and writing files from/to files
* Loading trained BPE vocabulary
* Custom Tensor operations (matmul, dot, relu, softmax, layer normalization...)

The tokenizer learns new token IDs starting from `256`, since the first 256 token IDs represent raw byte values.

### File I/O

Basic file utilities are implemented for:

* Reading UTF-8 text files
* Writing token sequences to files
* Converting file contents into `std::wstring` for tokenizer training
* Loading trained vocabulary

### Codec

The `Codec` component currently handles:

* UTF-8 encoding and decoding
* Conversion from UTF-8 strings to byte tokens
* BPE pair statistics
* Merging token pairs

## Current Pipeline

The current tokenizer pipeline is:

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

## Example

The current implementation can train BPE merge rules on a text corpus and then encode and decode new text:

```cpp
Tokenizer tokenizer;

auto texts = FileIO::read(
    R"(Training Files/romeo_and_juliet.txt)"
);

tokenizer.train(texts, 100);

auto tokens = tokenizer.encode(
    L"Two households, both alike in dignity, "
    L"In fair Verona, where we lay our scene,"
);

auto decoded = tokenizer.decode(tokens);
```

## Project Status

This project is **work in progress**.

Currently implemented:

* [x] UTF-8 codec
* [x] Byte-level tokenization
* [x] BPE statistics
* [x] BPE merge operations
* [x] BPE training
* [x] Token encoding
* [x] Token decoding
* [x] Basic pre-tokenization
* [x] Text file input/output
* [x] Tokenizer vocabulary persistence
* [x] Tokenizer loading/saving
* [x] GPT-2 vocabulary and merge format
* [x] Tensor implementation
* [x] Layer normalization
* [x] Linear layers

Planned:

* [ ] Self-attention
* [ ] Multi-head attention
* [ ] Feed-forward network
* [ ] Transformer blocks
* [ ] GPT-2 model
* [ ] Training/inference pipeline

## Goal

The main goal of this project is educational: to implement the core components of GPT-2 from the ground up and understand the mathematics and engineering behind modern transformer-based language models.

The implementation is intentionally built step by step rather than relying on frameworks such as PyTorch or TensorFlow.
