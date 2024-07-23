from typing import Any
import json
from Bio.PDB import (
    PDBParser,
    MMCIFIO,
    MMCIFParser,
    PDBIO,
    FastMMCIFParser,
    Atom,
    Model,
    Chain,
    Residue,
    Structure,
    PDBParser,
)


def update_fig_box_plot(
    fig: Any,
    legend_coordinates=(0, -0.10),
    marker_size=6,
) -> Any:
    params_axes = dict(
        showgrid=True,
        gridcolor="#d6d6d6",
        linecolor="black",
        zeroline=False,
        linewidth=1,
        showline=True,
        mirror=True,
        gridwidth=1,
        griddash="dot",
    )
    fig.update_xaxes(**params_axes)
    fig.update_yaxes(**params_axes)
    fig.update_layout(dict(plot_bgcolor="white"), margin=dict(l=0, r=5, b=0, t=20))
    if marker_size is not None:
        param_marker = dict(
            opacity=1, line=dict(width=0.5, color="DarkSlateGrey"), size=marker_size
        )
        fig.update_traces(marker=param_marker, selector=dict(mode="markers"))
        fig.update_traces(marker=dict(size=10))
    fig.update_layout(
        font=dict(
            family="Computer Modern",
            size=18,  # Set the font size here
        )
    )
    fig.update_layout(
        legend=dict(
            orientation="v",
            bgcolor="#f3f3f3",
            bordercolor="black",
            borderwidth=1,
        ),
    )
    fig.update_layout(
        legend=dict(
            x=legend_coordinates[0],
            y=legend_coordinates[1],
        ),
    )
    return fig


def update_bar_plot(fig: Any, legend_coordinates=(0.8, -0.15)) -> Any:
    params_axes = dict(
        showgrid=True,
        gridcolor="#d6d6d6",
        linecolor="black",
        zeroline=False,
        linewidth=1,
        showline=True,
        mirror=True,
        gridwidth=1,
        griddash="dot",
    )
    fig.update_xaxes(**params_axes)
    fig.update_yaxes(**params_axes)
    fig.update_layout(dict(plot_bgcolor="white"), margin=dict(l=0, r=5, b=0, t=20))
    fig.update_layout(
        font=dict(
            family="Computer Modern",
            size=26,
        )
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            bgcolor="#f3f3f3",
            bordercolor="black",
            borderwidth=1,
            x=-0.12,
            y=-0.25,
        ),
    )
    return fig


def read_json(in_path: str):
    with open(in_path, "r") as f:
        data = json.load(f)
    return data


def convert_cif_to_pdb(in_cif: str, out_pdb: str):
    """
    Convert a .cif file to a .pdb file
    :param in_cif: path to a .cif file
    :param out_pdb: path to save the .pdb file
    :return:
    """
    p = MMCIFParser()
    struc = p.get_structure("my_structure", in_cif)
    io = PDBIO()
    io.set_structure(struc)
    io.save(out_pdb)


def get_sequence_from_pdb(pdb_path: str):
    parser = PDBParser()
    structure = parser.get_structure("X", pdb_path)
    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == " ":
                    sequence += residue.get_resname()
    return sequence
