from setup import *

new_sequence = "PGLVEVKPNLVATRLXXXXXXXXXXXXREELLAWLRERSPGNTVVVDLRDAXXXXXXXXXLEVKDFRWPRGRPPPLESYAPLCEYFESVLSDGKGNVVALHCATGRGRTAAAIACYLLHSGEFKEAEEAIRWVGETIGGRGEVLTNPSLRAAVREYERLIREGLTYEPVTLLLHKLSFSKIPNVKGGTCNIQLEITQGDTVIYQSDVGPTKREKDKLVFELPEPLEVSGDVRVTFYHVDPETNTRTLLTSFWFNTFFIPGXXXXXXXXXXXXXXXXXXXXXXXXXXXXXSNKPLVKTLSKDDLDNAAYDTDEKTFPKGFTLELYFTLKN"
from Bio import Align
aligner = Align.PairwiseAligner()
alignments = aligner.align(pdb_seq, new_sequence)
print(alignments[0])

# Sequence_similarity
print("Alignment Score =", alignments[0].score)

# RMSD
mpnn_coords = load_pdb_file("/content/test_638d6_unrelaxed_rank_005_alphafold2_ptm_model_4_seed_000.pdb")
rmsd = compute_rmsd(true_coords, mpnn_coords[1])
print("RMSD:", rmsd)