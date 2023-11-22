from setup import *
from Bio.SVDSuperimposer import SVDSuperimposer

PDB_ID = "7JTX"
PDB_CHAIN = "A"
pdb_seq, true_coords = load_pdb(PDB_ID, PDB_CHAIN)
print(pdb_seq)

def load_pdb_file(file_name):
  pdbp = PDBParser()
  struct = pdbp.get_structure("struct", file_name)
  model = struct[0]

  sequence = []
  coords = []
  for chain in model:
    if chain.id != "A":
      continue
    for residue in chain:
      coords.append(residue['CA'].get_vector().get_array())
      resname = AA_NAME_MAP[residue.resname]
      sequence.append(residue.resname)

  return "".join(sequence), np.array(coords)

af2_coords = load_pdb_file("/test_52b46_unrelaxed_rank_005_alphafold2_ptm_model_1_seed_000.pdb")
esm2_coords = load_pdb_file("/esm.pdb")

def compute_rmsd(coords1, coords2):
    """This will align coords2 onto coords1."""
    coords1 = np.asarray(coords1)
    coords2 = np.asarray(coords2)
    sup = SVDSuperimposer()
    sup.set(coords1, coords2)
    sup.run()
    rms = sup.get_rms()
    return rms

print("RMSD between Predicted AF2 & ESM2 Structures:", compute_rmsd(af2_coords[1], esm2_coords[1]))
print("RMSD between AF2 & True Structures:", compute_rmsd(af2_coords[1], true_coords))
print("RMSD between ESM2 & True Structures:", compute_rmsd(esm2_coords[1], true_coords))