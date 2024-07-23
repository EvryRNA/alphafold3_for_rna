import os
import subprocess
from typing import List

import Bio
from Bio import PDB
from Bio.PDB import PDBIO

COMMAND = "lib/usalign/USalign -mol RNA $REFERENCE $PRED -o tmp/struct1"


def align_structure(reference_struct: str, prediction_struct: str, out_path: str):
    """
    Align the structures and create a new PDB with the reference with chain A and
    prediction with chain B.
    Args:
        :param reference_struct: The reference structure to superimpose
        :param prediction_struct: The predicted model to superimpose to the reference
        :param out_path: Where to save the superimposed structures
    """
    if not reference_struct.endswith(".pdb") or not prediction_struct.endswith(".pdb"):
        return None
    command = COMMAND.replace("$REFERENCE", reference_struct).replace(
        "$PRED", prediction_struct
    )
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error in superimposing {reference_struct} and {prediction_struct}")
        return None
    command = "rm tmp/*.pml"
    os.system(command)
    try:
        merge_structures(prediction_struct, "tmp/struct1.pdb", out_path)
    except (
        Bio.PDB.PDBExceptions.PDBIOException,
        ValueError,
        Bio.PDB.PDBExceptions.PDBConstructionException,
        FileNotFoundError,
    ):
        print(f"Error in merging {reference_struct} and {prediction_struct}")
    os.system("rm tmp/struct1.pdb")


def get_last_chain(chains: List):
    """
    Return the new chain id.
    """
    ids = [chain.id for chain in chains]
    orders = [ord(id) for id in ids]
    max_order = max(orders)
    return chr(max_order + 1).upper()


def merge_structures(struct1: str, struct2: str, out_path: str):
    """Merge struct1 with chain A and struct2 with chain B"""
    pdb1 = PDB.PDBParser().get_structure("pdb1", struct1)
    pdb2 = PDB.PDBParser().get_structure("pdb2", struct2)
    chains = list(pdb2.get_chains())
    new_id = get_last_chain(chains)
    chains[0].id = new_id
    chains[0].detach_parent()
    pdb1[0].add(chains[0])
    io = PDBIO()
    io.set_structure(pdb1)
    io.save(out_path)
