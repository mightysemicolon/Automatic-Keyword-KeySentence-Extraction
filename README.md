# Automatic Keyword & Key-Sentence Extraction with SVD and NMF

An unsupervised pipeline that finds the most important **keywords** and **key sentences** in a text using only linear algebra: no labels and no pretrained models. The example text is the fairy tale *Beauty and the Beast*.

The project compares two approaches on the same TF-IDF term–sentence matrix:

1. **SVD baseline:** rank terms and sentences by the leading singular vectors.
2. **NMF + QR with column pivoting:** factorize the matrix into non-negative topics, then pick one representative sentence per topic with a hand-written Householder QR step.

## Pipeline

**1. Preprocessing**
- Split the text into sentences on periods (173 sentences).
- Remove newline characters.

**2. Term–sentence matrix**
- Count terms with `CountVectorizer`, removing English stop words.
- Weight them with `TfidfTransformer`, giving a **964 × 173** TF-IDF matrix (terms × sentences).

**3. SVD baseline**
- Compute the SVD of the TF-IDF matrix.
- **Keywords:** terms ranked by the first left singular vector **u₁**.
- **Key sentences:** sentences ranked by the first right singular vector **v₁**.

**4. NMF + QR with column pivoting** (rank k = 5)
- Factorize the matrix as **A ≈ C·D** with scikit-learn's `NMF`, where *C* holds term weights per topic and *D* holds topic weights per sentence.
- For each step *i = 0 … k−1*:
  1. **Pivot:** choose the sentence (column of *D*) with the largest remaining norm in rows *i…k−1*, and swap it into position *i*.
  2. **Householder reflection:** build *Q* to zero out the entries below the diagonal in that column, then update *C ← C·Q·T* and *D ← T⁻¹·Qᵀ·D*.
  3. **Read off the result:** the pivoted sentence is the key sentence for topic *i*, and the largest entries in column *i* of *C* are its keywords.

The pivoting, Householder reflection and factor updates are written from scratch in NumPy.

## Results

**SVD:** the top keywords are *beauty, said, beast, father, monster, die*. The highest-ranked sentences are mostly dialogue between Beauty, her father and the Beast.

**NMF + QR pivoting:** five key sentences, each with its top two keywords.

| Step | Key sentence | Keywords |
|:---:|---|---|
| 1 | "No, dear Beast," said Beauty, "you must not die | said, beast |
| 2 | The horse went of himself into the stable, and the good man and his daughter came into the great hall… | good, went |
| 3 | The two eldest had a great deal of pride, because they were rich | great, deal |
| 4 | "What will you have, Beauty?" said her father | father, beauty |
| 5 | Farewell Beauty | beauty, farewell |

Compared with the SVD baseline, the NMF approach spreads its picks across different parts of the story: the Beast, the palace, the sisters, the father and the farewell. The SVD picks cluster around the main dialogue.

## Repository contents

| File | Description |
|---|---|
| `Automatic_KeyWord_and_KeySentence_Extraction.ipynb` | Full pipeline with outputs |
| `Beauty and the beast.txt` | Input text |

## Running it

```bash
pip install numpy pandas scikit-learn jupyter
jupyter notebook Automatic_KeyWord_and_KeySentence_Extraction.ipynb
```

The notebook reads `Beauty and the beast.txt` from the same folder. To analyze another text, replace that file. The NMF rank is the second argument of `get_key_word_sentence_each_step(Tfidf_text, 5)`.

## Background

The method follows the "Automatic Key Word and Key Sentence Extraction" chapter of Lars Eldén, *Matrix Methods in Data Mining and Pattern Recognition* (SIAM). Low-rank factorizations of the term–document matrix expose its main topics, and QR with column pivoting chooses the most representative sentences.
