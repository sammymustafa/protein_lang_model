from setup import *

# Create a dictionary mapping an amino acid to its vector
def get_aa_embeddings(model, alphabet):
  embedding_matrix = model.embed_tokens.weight.data.numpy()
  aa_tokens = alphabet.standard_toks[:-2]
  aa_to_index = alphabet.to_dict()
  aa_to_embeddings = {aa: embedding_matrix[aa_to_index[aa]] for aa in aa_tokens}
  return aa_to_embeddings

# Reduce to 2D using t-SNE or other technique and inspect visually
esm6_aa_embeddings = get_aa_embeddings(esm6_model, esm6_alphabet)
esm150_aa_embeddings = get_aa_embeddings(esm150_model, esm150_alphabet)

aa_names = list(esm6_aa_embeddings.keys())

# Convert embeddings to numpy arrays for t-SNE
embedding_vectors_esm6 = list(esm6_aa_embeddings.values())
embedding_vectors_esm150 = list(esm150_aa_embeddings.values())
# Convert the list of vectors to a NumPy array
embedding_matrix_esm6 = np.stack(embedding_vectors_esm6, axis=0)
embedding_matrix_esm150 = np.stack(embedding_vectors_esm150, axis=0)

# Run t-SNE
from sklearn.manifold import TSNE
tsne_esm6 = TSNE(n_components=2, perplexity=3, random_state=42).fit_transform(embedding_matrix_esm6)
tsne_esm150 = TSNE(n_components=2, perplexity=3, random_state=42).fit_transform(embedding_matrix_esm150)

# Function to plot the embeddings
def plot_embeddings(tsne_results, aa_names, title):
    plt.figure(figsize=(12, 8))
    for i, label in enumerate(aa_names):
        x, y = tsne_results[i, :]
        plt.scatter(x, y)
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5,2), ha='center')
    plt.title(title)
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.show()

# Plot for ESM-6
plot_embeddings(tsne_esm6, aa_names, 't-SNE of Amino Acid Embeddings for ESM-6 Model')

# Plot for ESM-150
plot_embeddings(tsne_esm150, aa_names, 't-SNE of Amino Acid Embeddings for ESM-150 Model')