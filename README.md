The notebook processes a sample text file (Beauty and the Beast) and applies a structured pipeline to uncover the most significant linguistic patterns. It leverages TF-IDF (Term Frequency–Inverse Document Frequency) to quantify word importance, followed by Non-negative Matrix Factorization (NMF) to uncover latent topics and extract the most relevant words for each.
By combining linguistic preprocessing with matrix factorization, the project automatically identifies both salient words and representative sentences, enabling concise and meaningful summaries of large documents.

🔍 Key Features

Text preprocessing: sentence tokenization, newline and punctuation removal

Construction of a term-document matrix using CountVectorizer and TfidfTransformer

Extraction of keywords via NMF topic modeling

Identification of key sentences that represent dominant topics

Fully interpretable, step-by-step implementation suitable for learning or extension

🧠 Techniques

TF-IDF weighting for importance scoring

Dimensionality reduction via NMF

Topic-word and document-topic matrix interpretation
