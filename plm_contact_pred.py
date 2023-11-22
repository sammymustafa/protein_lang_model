from setup import *
from sklearn.metrics import average_precision_score


def predict_contacts(model, alphabet, sequence):
  """Predict contacts with ESM."""
  batch_converter = alphabet.get_batch_converter()
  data = [("prot", sequence)]
  batch_labels, batch_strs, batch_tokens = batch_converter(data)
  batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

  with torch.no_grad():
      num_layers = model.num_layers
      results = model(batch_tokens, repr_layers=[num_layers], return_contacts=True)
  token_representations = results["representations"][num_layers]

  tokens_len = batch_lens[0]
  attention_contacts = results["contacts"][0]
  return attention_contacts[: tokens_len, : tokens_len]


def compare_contacts(pred_contacts, true_contacts):
  """Compares two contact maps.

  Prints the average precision and plots both maps.

  Parameters
  ----------
  pred_contacts:
    array of shape (num_residues, num_residues)
  true_contacts:
    array of shape (num_residues, num_residues)

  """
  # COMPLETE HERE
  # Make sure to flatten before computing the average precision
  # Flatten the contact maps to turn them into 1D arrays
  pred_contacts_flat = pred_contacts.flatten()
  true_contacts_flat = true_contacts.flatten()

  # Compute the average precision score
  avg_precision = average_precision_score(true_contacts_flat, pred_contacts_flat)

  # Print the average precision score
  print(f'Average Precision Score: {avg_precision}')

  # Plot the predicted and true contact maps for visual comparison
  fig, axes = plt.subplots(1, 2, figsize=(10, 5))

  # Plot predicted contacts
  axes[0].imshow(pred_contacts, cmap='viridis', origin='lower')
  axes[0].set_title('Predicted Contacts')
  axes[0].set_xlabel('Residue Index')
  axes[0].set_ylabel('Residue Index')

  # Plot true contacts
  axes[1].imshow(true_contacts, cmap='viridis', origin='lower')
  axes[1].set_title('True Contacts')
  axes[1].set_xlabel('Residue Index')
  axes[1].set_ylabel('Residue Index')

  plt.tight_layout()
  plt.show()

  return avg_precision

# we use the pdb_seq and contacts variables for esm6
pred_contacts = predict_contacts(esm6_model, esm6_alphabet, pdb_seq)
compare_contacts(pred_contacts, contacts)

# Try the esm150_model and the esm150_alphabet
pred_contacts = predict_contacts(esm150_model, esm150_alphabet, pdb_seq)
compare_contacts(pred_contacts, contacts)

# The ESM-150 model performs over 3x better than the ESM-6 model but is still very low with an average precision metric value of 0.27 precision.