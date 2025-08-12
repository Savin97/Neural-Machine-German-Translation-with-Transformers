# Transformer-based Neural Machine Translation (English ↔ German)

This project implements a **Transformer-based sequence-to-sequence neural machine translation** model trained on the **Multi30k English–German dataset**. It follows the *"Attention Is All You Need"* architecture, leveraging **multi-head self-attention** for efficient and accurate translation.

## 📌 Overview

* **Frameworks:** PyTorch, TorchText, SpaCy
* **Architecture:** Encoder–Decoder with multi-head self-attention (`nn.Transformer`)
* **Task:** English ↔ German translation
* **Evaluation:** BLEU score & qualitative translation analysis

## 🚀 Features

* Tokenization and preprocessing of the Multi30k bilingual dataset using SpaCy
* Subword vocabulary creation for robust handling of rare words and morphology
* Implementation of Transformer encoder–decoder in PyTorch with positional encoding
* Hyperparameter tuning: batch size, embedding dimension, learning rate, layers, attention heads
* Model evaluation with BLEU and human-readable translation samples

## 📂 Project Structure

```
├── data/                  # Dataset loading & preprocessing scripts
├── model/                 # Transformer model definition
├── train.py               # Training loop
├── evaluate.py            # Evaluation scripts
├── utils.py               # Helper functions (masks, tokenization, vocab)
├── outputs/               # Sample translations & BLEU scores
└── README.md              # Project documentation
```

## 📊 Example Outputs

| Input (DE)                                                         | Output (EN)                                                           |
| ------------------------------------------------------------------ | --------------------------------------------------------------------- |
| ein mann in einem blauen hemd steht auf der seite eines gebäudes . | Skateboarder in a blue bathing suit standing on the edge of a house . |
| zwei männer in einem restaurant unterhalten sich .                 | 2 friends are talking to each other in a restaurant .                 |
| ein kind spielt mit einem ball auf einem großen feld .             | Surfer is playing with a large board on a large surface .             |

*(Note: Translation quality limited by small dataset size! Improvements discussed below.)*

## 🔧 Future Improvements

* Add **padding masks** for source, target, and memory to improve alignment
* Use **subword tokenization** (SentencePiece/BPE) to reduce rare-word errors
* Apply **label smoothing**, **Noam learning rate warm-up**, and **gradient clipping** for more stable training
* Implement **length-penalized beam search** to reduce repetitive or irrelevant phrases

## 📜 References

* Vaswani et al., *Attention Is All You Need*, 2017
* [PyTorch `nn.Transformer` Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
* [TorchText Multi30k Dataset](https://pytorch.org/text/stable/datasets.html#multi30k)
