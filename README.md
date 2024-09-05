# Machine Translation: English to French using Seq2Seq Model

This repository contains code for a machine translation task from English to French using a sequence-to-sequence (Seq2Seq) model with attention mechanism. The implementation leverages PyTorch and the `d2l` library.

## Table of Contents

1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Prediction](#prediction)
7. [Exercises](#exercises)
8. [References](#references)

## Installation

Ensure you have Python installed. Then, install the required libraries:

```bash
!pip install d2l
```

## Dataset

The dataset used is an English-French translation dataset. The code downloads and extracts the dataset automatically.

## Preprocessing

1. **Reading the Dataset**: The dataset is downloaded and read into raw text format.
2. **Preprocessing the Data**: The raw text is preprocessed by converting to lowercase and inserting spaces around punctuation.
3. **Tokenization**: The text is tokenized into source (English) and target (French) sequences.

## Model Architecture

1. **Encoder**: A GRU-based encoder that processes the input sequence.
2. **Decoder**: A GRU-based decoder that generates the output sequence.
3. **Loss Function**: Custom masked softmax cross-entropy loss to handle padding.
4. **Training**: The model is trained using teacher forcing and gradient clipping.

## Training

The training function initializes the model parameters, sets up the optimizer and loss function, and trains the model over a specified number of epochs.

## Prediction

The prediction function takes an English sentence, processes it through the trained model, and outputs the translated French sentence.

## Exercises

### Exercise 9.5.7.1: Impact of `num_examples` on Vocabulary Sizes

In this exercise, we will explore how varying the `num_examples` parameter in the `load_data_nmt` function affects the sizes of the source and target vocabularies. The `num_examples` parameter determines the number of translation examples used to build the vocabularies. By experimenting with different values, we can observe how the vocabulary sizes change.

#### Experiment with Different `num_examples`

We will test the `load_data_nmt` function with different values of `num_examples` and observe the resulting vocabulary sizes.

```python
def test_vocab_sizes(num_examples_list):
    for num_examples in num_examples_list:
        print(f"num_examples = {num_examples}")
        train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=64, num_steps=10, num_examples=num_examples)
        print(f"Source vocabulary size: {len(src_vocab)}")
        print(f"Target vocabulary size: {len(tgt_vocab)}\n")

num_examples_list = [100, 500, 1000, 5000, 10000]
test_vocab_sizes(num_examples_list)
```

#### Results

After running the above code, we should see the sizes of the source and target vocabularies for each `num_examples` value. Here is an example of what the output might look like:

```
num_examples = 100
Source vocabulary size: 40
Target vocabulary size: 40

num_examples = 500
Source vocabulary size: 159
Target vocabulary size: 163

num_examples = 1000
Source vocabulary size: 266
Target vocabulary size: 321

num_examples = 5000
Source vocabulary size: 875
Target vocabulary size: 1231

num_examples = 10000
Source vocabulary size: 1505
Target vocabulary size: 2252
```

#### Conclusion

As we can see from the results, increasing the `num_examples` value leads to larger vocabulary sizes for both the source and target languages. This is because more examples provide a greater variety of words, resulting in more comprehensive vocabularies. However, it's important to balance the `num_examples` value to ensure that the vocabulary is rich enough without excessively increasing the computational resources required.

By understanding and controlling the `num_examples` parameter, we can better manage the trade-off between vocabulary richness and efficiency in machine translation tasks.

### Exercise 9.5.7.2: Appropriateness of Word-Level Tokenization for Chinese and Japanese

Word-level tokenization is generally not suitable for languages like Chinese and Japanese due to their unique morphological characteristics and the absence of explicit word boundaries. Instead, character-level tokenization or subword techniques are more effective.

Chinese and Japanese are morphologically distinct from Western languages, characterized by logographic (Chinese) and syllabic (Japanese kana) writing systems. In these languages, words are not separated by spaces, and a single character can represent an entire word or syllable. Tokenizing Chinese or Japanese text at the word level requires advanced text segmentation techniques. Tools like Jieba for Chinese (https://github.com/fxsjy/jieba) and MeCab for Japanese (https://taku910.github.io/mecab/) are commonly used to identify word boundaries. These tools use complex algorithms that consider context and grammatical rules, which simple space-based tokenization cannot achieve (Dive into Deep Learning, Section 9.5.2).

Character-level tokenization is a more straightforward and often more effective approach for these languages. This method avoids the complexity of word segmentation and allows machine learning models to process each character individually, which can be more efficient in terms of processing and memory. Modern models, such as BERT and Transformer, often use subword or character-level tokenization to handle multiple languages. For instance, BERT uses WordPiece tokenization, which splits words into subwords or characters, enabling a more granular and effective representation for languages like Chinese and Japanese.

#### Conclusion

For languages like Chinese and Japanese, word-level tokenization is not ideal due to the absence of explicit word boundaries and the complexity involved in segmentation. Instead, character-level tokenization or the use of subword techniques are more effective approaches, allowing for a more accurate and efficient representation of text. These approaches are supported by specialized segmentation tools and state-of-the-art models that better handle the unique morphology of these languages.

## References

- [Dive into Deep Learning](https://d2l.ai/)