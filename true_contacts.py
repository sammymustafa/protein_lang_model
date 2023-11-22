from setup import *

PDB_ID = "7JTX" # The PDB ID of the protein you chose
PDB_CHAIN = "A" # The chain to load (you can view chains in the 3D view on rcsb.org)


def compute_contacts(coords, threshold=9.0):
  """Compute the pairwise contacts.

  Here we define a contact as the C-alpha atom
  of 2 amino acids being within 9 Angstrom of each other.

  Parameters
  ----------
  coords:
    array of shape (num_residues, 3)
  threshold:
    distance threshold to consider a contact

  Returns
  -------
  contacts:
    binary array of shape (num_residues, num_residues)

  """
  num_residues = coords.shape[0]
  contacts = np.zeros((num_residues, num_residues), dtype=int)

  # Compute pairwise distance matrix
  for i in range(num_residues):
      for j in range(i+1, num_residues):
          distance = np.linalg.norm(coords[i] - coords[j])
          if distance < threshold:
              contacts[i, j] = 1
              contacts[j, i] = 1  # The contact matrix is symmetric

  return contacts

def load_pdb(pdb_id, chain_id):
  pdbl = PDBList()
  pdbp = PDBParser()
  pdbl.retrieve_pdb_file(pdb_id.upper(), file_format="pdb", pdir="./")
  struct = pdbp.get_structure("struct", f"pdb{pdb_id.lower()}.ent")
  model = struct[0]

  sequence = []
  coords = []
  for chain in model:
    if chain.id != chain_id:
      continue
    for residue in chain:
      if residue.resname == "HOH":
        continue
      try:
        coords.append(residue['CA'].get_vector().get_array())
      except:
        raise ValueError("There are missing atoms in this structure, try another!")
      resname = AA_NAME_MAP[residue.resname]
      sequence.append(resname)

  return "".join(sequence), np.array(coords)


pdb_seq, coords = load_pdb(PDB_ID, PDB_CHAIN)
contacts = compute_contacts(coords)
plt.imshow(contacts)