COLOR_DATASET = {
    "RNA-Puzzles": "#EF8927",
    "CASP-RNA": "#A0E9FF",
    "RNASolo": "#E72929",
    "RNA3DB_0 (Context)": "#fde724",
    "RNA3DB_0": "#00569e",
    "RNA3DB_Long": "#81A263",
}
COLOR_INTERACTIONS = {"STACK": "#ef8927", "WC": "#83b8d6", "nWC": "#621038"}
METRICS_ALL = [
    "RMSD",
    "P-VALUE",
    "INF-ALL",
    "INF-WC",
    "INF-NWC",
    "INF-STACK",
    "DI",
    "MCQ",
    "TM-score",
    "GDT-TS",
    "εRMSD",
    "lDDT",
]
COLORS_MAPPING = {
    "RMSD": "#e10000",
    "INF-ALL": "#656567",
    "CAD": "#ee7f00",
    "TM-score": "#8b1b58",
    "GDT-TS": "#76885B",
    "lDDT": "#31b2cb",
    "P-VALUE": "#B67352",
    "εRMSD": "#FFD23F",
    "MCQ": "#005793",
    "INF-STACK": "#ef8927",
    "INF-WC": "#83b8d6",
    "INF-NWC": "#621038",
    "LCS-10": "#3dbc75",
}
OLD_TO_NEW = {
    "BARNABA-eRMSD": "εRMSD",
    "LCS-TA-COVERAGE-10": "LCS-10",
    "rnacomposer": "RNAComposer (TP)",
    "isrna": "IsRNA1 (Ab)",
    "3drna": "3dRNA (TP)",
    "rhofold": "RhoFold (DL)",
    "simrna": "SimRNA (Ab)",
    "vfold": "Vfold3D (TP)",
    "eprna": "epRNA",
    "rp14_free": "rp14f",
    "rp14_bound": "rp14b",
    "lddt": "lDDT",
    "trrosettarna": "trRosettaRNA (DL)",
    "mcsym": "MC-Sym (TP)",
    "vfoldpipeline": "Vfold-Pipeline (TP)",
    "rnajp": "RNAJP (Ab)",
    "alphafold": "AlphaFold 3 (DL)",
    "alphafold3": "AlphaFold 3 (DL)",
    "alphafold3c": "AlphaFold 3 (Context)",
    "best": "Challenge-best",
    "RNA_PUZZLES": "RNA-Puzzles",
    "RNASOLO": "RNASolo",
    "RNA3DB_LONG": "RNA3DB_Long",
    "RNA3DB": "RNA3DB_0",
    "CASP_RNA": "CASP-RNA",
    "CASP": "CASP-RNA",
    # "INF-ALL": r"$INF_{all}$"
}

SUB_METRICS = [
    "RMSD",
    "P-VALUE",
    "εRMSD",
    "TM-score",
    "GDT-TS",
    "INF-ALL",
    "CAD",
    "lDDT",
    "MCQ",
    "LCS-10",
]

ALL_MODELS = [
    "mcsym",
    # "ifoldrna",
    "vfold",
    "rnacomposer",
    "simrna",
    "3drna",
    "isrna",
    "rhofold",
    "trrosettarna",
    "vfoldpipeline",
    "rnajp",
    "alphafold3",
    "alphafold3c",
    "best",
]
ALL_MODELS_CLEAN = [OLD_TO_NEW.get(name) for name in ALL_MODELS]

# Higher is better
ASC_METRICS = [
    "INF-ALL",
    "TM-score",
    "GDT-TS",
    "lDDT",
    "INF-WC",
    "INF-NWC",
    "INF-STACK",
    "CAD",
    "GDT-TS",
]
# Lower is better
DESC_METRICS = ["RMSD", "P-VALUE", "DI", "εRMSD", "MCQ"]
MODELS_TO_GROUP = {
    "Challenge-best": "",
    "AlphaFold3": "Deep learning",
    "RNAJP": "Ab initio",
    "Vfold-Pipeline": "Template-based",
    "trRosettaRNA": "Deep learning",
    "RhoFold": "Deep learning",
    "IsRNA1": "Ab initio",
    "3dRNA": "Template-based",
    "SimRNA": "Ab initio",
    "RNAComposer": "Template-based",
    "Vfold3D": "Template-based",
    "MC-Sym": "Template-based",
}
ORDER_MODELS = [
    "Challenge-best",
    "AlphaFold 3 (DL)",
    "AlphaFold 3 (Context)",
    "RNAJP (Ab)",
    "Vfold-Pipeline (TP)",
    "trRosettaRNA (DL)",
    "RhoFold (DL)",
    "IsRNA1 (Ab)",
    "3dRNA (TP)",
    "SimRNA (Ab)",
    "RNAComposer (TP)",
    "Vfold3D (TP)",
    "MC-Sym (TP)",
]
