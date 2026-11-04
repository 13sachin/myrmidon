#!/usr/bin/env python3
"""Build the shipped documentation corpus: named pages of engine documentation, CLI tool
reference, CPMD keywords and selection grammars into
src/myrmidon/knowledge/docs/<source>-<version>-<page>.jsonl.

The corpus is what `knowledge_search` reads. It is curated and versioned,
never scraped live: every record carries the page's URL and sha256, the section it came
from, the engine version it documents and the licence it ships under. Re-running the script
rebuilds the same records from the same pages and says when a page changed.

Licences decide what ships as text. GROMACS documentation is LGPL-2.1 and ships as text
with the notice in `src/myrmidon/knowledge/docs/NOTICE`. A page whose licence does not
allow that is listed with `text: false` and ships as pointers only: section titles and the
URL, which is still an answer, since it tells the reader where to look.

    scripts/build_docs_corpus.py            # rebuild every page in PAGES
    scripts/build_docs_corpus.py --local    # build local catalog/tool/grammar pages only
    scripts/build_docs_corpus.py --check    # fetch and compare hashes, write nothing
"""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import re
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "src" / "myrmidon" / "knowledge" / "docs"
CATALOG_DIR = ROOT / "src" / "myrmidon" / "knowledge" / "catalog"

#: What is fetched remotely when run without --local.
PAGES = [
    {"source": "gromacs", "version": "2021.6", "page": "mdp-options", "kind": "mdp-options",
     "url": "https://manual.gromacs.org/2021.6/user-guide/mdp-options.html",
     "licence": "GROMACS documentation, LGPL-2.1", "text": True},
    {"source": "gromacs", "version": "2021.6", "page": "mimic-qmmm", "kind": "sections",
     "url": "https://manual.gromacs.org/2021.6/reference-manual/special/mimic-qmmm.html",
     "licence": "GROMACS documentation, LGPL-2.1", "text": True},
]

NOTICE = """The files in this directory are excerpts of engine documentation, chunked by
section for `knowledge_search`, each record naming the page it came from and its sha256.

GROMACS documentation (manual.gromacs.org) is distributed under the GNU Lesser General
Public License, version 2.1 (https://www.gnu.org/licenses/lgpl-2.1.html). The excerpts
here are reproduced under that licence; the GROMACS authors hold the copyright.
"""

#: Curated GROMACS CLI tool & selection reference sections (GROMACS 2021.6 manual, LGPL-2.1).
GMX_TOOLS = [
    (
        "gmx make_ndx",
        "gmx make_ndx builds GROMACS index (.ndx) groups interactively or from commands. "
        "Syntax in myrmidon gmx_make_ndx commands: group names containing non-alphanumeric "
        "characters must be quoted in double quotes, group numbers are bare integers, "
        "boolean operators are '|' (OR / union), '&' (AND / intersection), and '!' (NOT / "
        "negation). Atom-name selection uses 'a' (e.g. '0 & ! a H*' selects all non-hydrogen "
        "heavy atoms of group 0, useful on a single-ligand coordinate file so atoms are "
        "numbered from 1). Residue-name selection uses 'r' (e.g. 'r JZ4'). To merge a "
        "protein and a bound ligand into one temperature-coupling bath (so a small ligand "
        "is never coupled to its own thermostat), use command '\"Protein\" | \"Other\"' or "
        "'\"Protein\" | \"JZ4\"', producing group Protein_Other alongside Water_and_ions.",
        "#gmx-make-ndx",
    ),
    (
        "gmx mindist",
        "gmx mindist computes the minimum distance between two groups of atoms and the "
        "number of contacts within a cutoff distance (-d, default 0.6 nm) over a trajectory. "
        "Options: -f trajectory (.xtc/.trr), -s structure/tpr, -n index (.ndx), -od minimum "
        "distance vs time (.xvg), -on number of contacts vs time (.xvg), -or minimum "
        "distance per residue (-respertime writes per-residue distance matrix over time), "
        "-d contact cutoff in nm, -group count contacts with multiple atoms in the second "
        "group as one contact. In myrmidon, use traj_analyze with metric='mindist' (reports "
        "mean, min, largest and final distance) or metric='contacts' (reports per-residue "
        "contact fraction over the trajectory in both topology and deposited PDB numbering).",
        "#gmx-mindist",
    ),
    (
        "gmx select",
        "gmx select writes dynamic or static atom/residue selections using GROMACS selection "
        "syntax (distinct from traditional make_ndx prompt syntax). Syntax examples: "
        "'resname JZ4 and not name \"H*\"', 'group \"Protein\" and within 0.4 of resname JZ4', "
        "'backbone', 'resid 1 to 100'. Note: for trajectory analysis in myrmidon, "
        "traj_analyze uses MDAnalysis selection syntax ('protein and backbone', "
        "'resname JZ4 and not name H*', 'resid 42'), NOT make_ndx ('0 & ! a H*') and NOT "
        "the keyword 'heavy'.",
        "#gmx-select",
    ),
    (
        "gmx genrestr",
        "gmx genrestr produces an #include position-restraint file (posre.itp) for a "
        "single moleculetype from a coordinate file (-f) and an index group (-n), with "
        "force constants -fc (default 1000 1000 1000 kJ mol^-1 nm^-2). CRITICAL: atom "
        "indices in [ position_restraints ] are 1-based WITHIN THE MOLECULETYPE, not "
        "global system atom numbers. Always run gmx_make_ndx and gmx_genrestr on the "
        "isolated ligand structure (e.g. JZ4_GMX.gro where atoms are numbered 1..N_lig) "
        "before complex_assemble, never on the assembled complex.",
        "#gmx-genrestr",
    ),
    (
        "gmx pdb2gmx",
        "gmx pdb2gmx reads a coordinate file (.pdb/.gro), matches residues against the "
        "chosen force field's residue database (.rtp), adds hydrogens, writes the GROMACS "
        "topology (.top) and position restraints (posre.itp). Options: -ff force field "
        "(e.g. amber99sb-ildn, charmm36-jul2022), -water water model (tip3p, spce, tip4p), "
        "-ignh ignore input hydrogens and rebuild according to force field rules. "
        "Non-polymer HETATM records (ligands, buffer ions, crystal waters) not in the force "
        "field .rtp must be split out with structure_select before running gmx_pdb2gmx; "
        "kept structural ions/waters go after TER in their own chains. Note that .gro output "
        "renumbers residues sequentially from 1 and drops PDB insertion codes.",
        "#gmx-pdb2gmx",
    ),
    (
        "gmx editconf",
        "gmx editconf converts coordinate formats and defines the periodic simulation box "
        "around a solute. Options: -bt box type (cubic, triclinic, dodecahedron, "
        "octahedron), -d distance from the solute to the box wall in nm (minimum periodic "
        "image distance is 2*d; keep d >= 1.0 nm so the solute cannot see its periodic image "
        "across a 1.0-1.2 nm non-bonded cutoff), -c center the solute in the box.",
        "#gmx-editconf",
    ),
    (
        "gmx solvate",
        "gmx solvate fills the simulation box (-cp boxed.gro) with solvent configurations "
        "(-cs spc216.gro) and updates the [ molecules ] section of the topology (-p topol.top) "
        "with the number of SOL molecules added. Refuses if the topology is already solvated.",
        "#gmx-solvate",
    ),
    (
        "gmx genion",
        "gmx genion replaces solvent molecules with monoatomic ions (e.g. NA, CL) to "
        "neutralise the system's net charge (-neutral) and/or reach a target salt "
        "concentration (-conc, in mol/L). Requires a run input (.tpr) built by grompp from "
        "the solvated coordinates and topology, and updates [ molecules ] in the topology.",
        "#gmx-genion",
    ),
    (
        "gmx grompp",
        "gmx grompp (the GROMACS preprocessor) reads a molecular topology (-p .top), "
        "coordinates (-c .gro), position-restraint reference coordinates (-r .gro, required "
        "whenever define activates -DPOSRES or -DPOSRES_LIG), an index file (-n .ndx), and "
        "run parameters (-f .mdp) to compile a portable binary run input (-o .tpr). For "
        "MiMiC QM/MM, grompp zeroes MM charges on QMMM-grps, excludes QM-QM LJ interactions, "
        "and removes QM-QM bonds.",
        "#gmx-grompp",
    ),
    (
        "gmx mdrun",
        "gmx mdrun performs energy minimisation, molecular dynamics, or trajectory rerun "
        "from a compiled .tpr file (-s/-deffnm). Continuation from a checkpoint uses "
        "-cpi state.cpt (in myrmidon, use checkpoint_continue so stage directory and "
        "deffnm match the checkpoint record).",
        "#gmx-mdrun",
    ),
    (
        "gmx energy",
        "gmx energy extracts energy, temperature, pressure, volume, density and box terms "
        "from a binary .edr file. Term names must match the exact labels in the .edr "
        "(e.g. 'Potential', 'Temperature', 'Pressure', 'Density', 'Total-Energy', "
        "'Kinetic-En.').",
        "#gmx-energy",
    ),
    (
        "selection grammars: MDAnalysis vs gmx make_ndx vs MiMiCPy prepqm",
        "Myrmidon tools use three distinct atom-selection grammars — never mix them:\n"
        "1. traj_analyze (MDAnalysis selection grammar):\n"
        "   - Protein backbone: 'protein and backbone'\n"
        "   - Ligand heavy atoms: 'resname JZ4 and not name H*' (NEVER use 'heavy' — "
        "'heavy' is not a valid MDAnalysis token)\n"
        "   - All ligand atoms: 'resname JZ4'\n"
        "   - Protein heavy atoms: 'protein and not name H*'\n"
        "   - Specific residues: 'resid 42:50 and protein' (uses topology 1-based residue "
        "numbers from .gro/.tpr; check structure_select or system_inspect residue_map when "
        "the deposited PDB has offset numbering or insertion codes).\n"
        "2. gmx_make_ndx (GROMACS make_ndx interactive prompt grammar):\n"
        "   - Merge protein and ligand bath: '\"Protein\" | \"Other\"' (double quotes around "
        "group names, '|' for union, '&' for intersection, '!' for negation)\n"
        "   - Heavy atoms of group 0: '0 & ! a H*'\n"
        "   - Residue by name: 'r JZ4'.\n"
        "3. mimic_select_qm (MiMiCPy prepqm selection grammar):\n"
        "   - Commands start with 'add ', 'add-bound ', 'delete ', or 'clear'\n"
        "   - Clauses use 'FIELD is VALUE' joined by 'and'/'or': e.g. 'add resname is ACT', "
        "'add resid is 42 and name is OG', 'add-bound resid is 42 and name is CB'.",
        "#selection-grammars",
    ),
]


#: Extra curated semantic descriptions for the most important CPMD / MiMiC keywords.
CPMD_SEMANTIC_NOTES = {
    "CPMD/MOLECULAR DYNAMICS": (
        "Starts a molecular dynamics calculation. Sub-options: BO (Born-Oppenheimer MD, "
        "where the electronic wavefunction is converged at every ionic step — standard for "
        "MiMiC QM/MM), CP (Car-Parrinello MD), Ehrenfest, PT."
    ),
    "CPMD/TIMESTEP": (
        "Integration time step in atomic units (a.u.; 1 fs = 41.341373335 a.u., so 10 a.u. "
        "is ~0.2419 fs and 20 a.u. is ~0.4838 fs). In a coupled MiMiC QM/MM run, CPMD is "
        "the MD driver and GROMACS acts as the force client."
    ),
    "CPMD/MAXSTEP": (
        "Maximum number of steps: under MOLECULAR DYNAMICS or OPTIMIZE GEOMETRY this is the "
        "number of ionic/MD steps; under OPTIMIZE WAVEFUNCTION (single-point calculation) "
        "this is the maximum number of SCF electronic iterations."
    ),
    "CPMD/NOSE": (
        "Nosé-Hoover thermostat for constant-temperature dynamics. Sub-options: IONS "
        "(thermostat on nuclear coordinates; followed by target temperature in K and "
        "characteristic frequency in cm^-1, e.g. 300.0 1500.0), ELECTRONS, CELL."
    ),
    "CPMD/BERENDSEN": (
        "Berendsen velocity-rescaling thermostat/barostat. Sub-options: IONS (followed by "
        "target temperature in K and coupling time in a.u., e.g. 300.0 5000.0), ELECTRONS, CELL."
    ),
    "CPMD/TEMPERATURE": (
        "Initial ionic temperature in K (or TEMPERATURE RAMP: start_K target_K ramp_rate "
        "for linear heating)."
    ),
    "CPMD/ANNEALING": (
        "Velocity scaling factor per step (e.g. ANNEALING IONS 0.95 or 0.99) to remove "
        "excess kinetic energy and quench the system toward a local minimum."
    ),
    "CPMD/CONVERGENCE": (
        "Convergence criteria: CONVERGENCE ORBITALS (electronic wavefunction gradient "
        "threshold, e.g. 1.0e-5 or 1.0e-6) and CONVERGENCE GEOMETRY (ionic force threshold)."
    ),
    "CPMD/ISOLATED MOLECULE": (
        "Treats the QM subsystem as an isolated (non-periodic) charge distribution inside "
        "the QM orthorhombic cell, required with a Poisson solver (TUCKERMAN / MARTYNA) "
        "and vacuum padding (0.3-0.5 nm per side) in MiMiC QM/MM."
    ),
    "CPMD/RESTART": (
        "Restarts calculation from RESTART.1 file. Sub-options: COORDINATES, VELOCITIES, "
        "WAVEFUNCTION, GEOFILE, ACCUMULATORS, NOSEP, NOSEE. Never include RESTART on the "
        "first coupled stage when no restart file exists yet."
    ),
    "CPMD/FORCE_MATCHING": (
        "Writes reference QM/MM energies, forces and electrostatic fields every N steps "
        "for force-matching parameterization; pace must match GROMACS nstxout and nstfout."
    ),
    "SYSTEM/CHARGE": (
        "Total net electronic charge of the QM region. Must equal the formal integer charge "
        "of the selected QM atoms (confirmed by the researcher in mimic_prepare)."
    ),
    "SYSTEM/MULTIPLICITY": (
        "Spin multiplicity (2S+1) of the QM region. Multiplicity > 1 (open-shell) requires "
        "&CPMD LSD (local spin density / unrestricted Kohn-Sham)."
    ),
    "SYSTEM/CELL": (
        "QM simulation box dimensions (a, b, c, cos(alpha), cos(beta), cos(gamma)); in "
        "MiMiC QM/MM generated by mimicpy prepqm from the QM atoms' bounding box plus padding."
    ),
    "SYSTEM/CUTOFF": (
        "Plane-wave kinetic energy cutoff in Rydberg (Ry) for the electronic Kohn-Sham "
        "orbitals (e.g. 70-100 Ry for Troullier-Martins norm-conserving pseudopotentials)."
    ),
    "SYSTEM/POISSON SOLVER": (
        "Electrostatic solver for isolated QM systems (e.g. TUCKERMAN or MARTYNA) decoupling "
        "periodic images of the QM electron density."
    ),
    "DFT/FUNCTIONAL": (
        "Exchange-correlation DFT functional (e.g. BLYP, PBE, BP, LDA, PBE0, B3LYP)."
    ),
    "MIMIC/PATHS": (
        "Number of MM client codes (1 for GROMACS) followed by the absolute working "
        "directory path where GROMACS mdrun executes."
    ),
    "MIMIC/OVERLAPS": (
        "Mapping table between MM atom indices in GROMACS (code 2) and QM atom indices in "
        "CPMD (code 1), written by mimicpy prepqm."
    ),
    "MIMIC/BOX": (
        "Dimensions of the full MM periodic simulation box in Bohr."
    ),
    "MIMIC/LONG-RANGE COUPLING": (
        "Enables multipole expansion for long-range QM-MM electrostatic interactions beyond "
        "CUTOFF DISTANCE (in Bohr, default 20 Bohr) up to MULTIPOLE ORDER."
    ),
}


def _clean(fragment: str) -> str:
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    fragment = html.unescape(fragment).replace("¶", "")
    fragment = re.sub(r"\s*<\S*$", "", fragment)
    return re.sub(r"\s+", " ", fragment).strip()


def chunk_mdp_options(text: str) -> list[tuple[str, str]]:
    """One chunk per mdp option: from its anchor to the next option's (values are nested
    anchors `mdp-value-<option>-<value>` and stay inside their option's chunk)."""
    anchors = [(m.start(), m.group(1))
               for m in re.finditer(r'id="mdp-((?!value-)[A-Za-z0-9_-]+)"', text)]
    out = []
    for i, (pos, name) in enumerate(anchors):
        end = anchors[i + 1][0] if i + 1 < len(anchors) else len(text)
        block = _clean(text[pos:end])
        block = re.sub(r'^id="mdp-[^"]+">\s*', "", block)
        block = re.split(r"\s(?=\S+ Previous Next)", block)[0]
        if len(block) >= 20:
            out.append((f"mdp option {name}", block))
    return out


def chunk_sections(text: str) -> list[tuple[str, str]]:
    """One chunk per section heading of a Sphinx page (h1–h3)."""
    parts = re.split(r"(<h[123][^>]*>.*?</h[123]>)", text, flags=re.S)
    out = []
    for i in range(1, len(parts) - 1, 2):
        title = _clean(parts[i])
        body = _clean(parts[i + 1])
        body = re.split(r"\s(?=\S+ Previous Next)", body)[0]
        if title and len(body) >= 40:
            out.append((title, body))
    return out


CHUNKERS = {"mdp-options": chunk_mdp_options, "sections": chunk_sections}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 — named pages only
        return resp.read()


def build(page: dict, check: bool = False) -> str:
    raw = fetch(page["url"])
    sha = hashlib.sha256(raw).hexdigest()
    target = DST / f"{page['source']}-{page['version']}-{page['page']}.jsonl"
    if target.exists():
        first = json.loads(target.read_text().splitlines()[0])
        state = "unchanged" if first.get("sha256") == sha else "CHANGED"
    else:
        state = "new"
    if check:
        return f"{state:10s} {target.name}"
    chunks = CHUNKERS[page["kind"]](raw.decode("utf-8", "replace"))
    today = dt.date.today().isoformat()
    with target.open("w") as fh:
        for section, body in chunks:
            record = {
                "source": page["source"], "version": page["version"], "page": page["page"],
                "section": section, "text": body[:3000] if page["text"] else "",
                "locator": page["url"] + _fragment(section, page["kind"]),
                "sha256": sha, "fetched": today, "licence": page["licence"],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    (DST / "NOTICE").write_text(NOTICE)
    return f"{state:10s} {target.name}: {len(chunks)} chunk(s)"


def build_local_tools_and_cpmd(check: bool = False) -> list[str]:
    """Build deterministic JSONL corpus files for GROMACS CLI tools, selection grammars,
    and CPMD keywords from the shipped catalog."""
    results = []
    mdp_jsonl = DST / "gromacs-2021.6-mdp-options.jsonl"
    today = (json.loads(mdp_jsonl.read_text().splitlines()[0]).get("fetched", "")
             if mdp_jsonl.is_file() else dt.date.today().isoformat())

    # 1. GROMACS CLI tools & selection grammars
    tools_url = "https://manual.gromacs.org/2021.6/user-guide/cmdline.html"
    tools_payload = json.dumps(GMX_TOOLS, sort_keys=True).encode("utf-8")
    tools_sha = hashlib.sha256(tools_payload).hexdigest()
    tools_target = DST / "gromacs-2021.6-tools.jsonl"
    if not check:
        with tools_target.open("w") as fh:
            for section, body, frag in GMX_TOOLS:
                record = {
                    "source": "gromacs",
                    "version": "2021.6",
                    "page": "tools",
                    "section": section,
                    "text": body,
                    "locator": tools_url + frag,
                    "sha256": tools_sha,
                    "fetched": today,
                    "licence": "GROMACS documentation, LGPL-2.1",
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    results.append(f"{'checked':10s} {tools_target.name}: {len(GMX_TOOLS)} chunk(s)")

    # 2. CPMD 4.3 keywords from catalog/cpmd-4.3.yaml
    cpmd_cat_path = CATALOG_DIR / "cpmd-4.3.yaml"
    cpmd_raw = cpmd_cat_path.read_bytes()
    cpmd_sha = hashlib.sha256(cpmd_raw).hexdigest()
    cpmd_data = yaml.safe_load(cpmd_raw.decode("utf-8"))
    base_url = cpmd_data["sources"][0]["locator"]
    cpmd_target = DST / "cpmd-4.3-keywords.jsonl"
    keys = cpmd_data.get("keys", {})
    if not check:
        with cpmd_target.open("w") as fh:
            for full_key, entry in sorted(keys.items()):
                sec = entry["section"]
                name = entry["name"]
                opts = entry.get("options") or "none"
                default = entry.get("default")
                aliases = entry.get("aliases") or []
                sem = CPMD_SEMANTIC_NOTES.get(full_key, "")
                parts = [
                    f"CPMD keyword &{sec} / {name}.",
                    f"Section: &{sec}. Options/syntax: {opts}.",
                ]
                if default is not None:
                    parts.append(f"Default: {default}.")
                if aliases:
                    parts.append(f"Accepted aliases: {', '.join(aliases)}.")
                if sem:
                    parts.append(sem)
                frag = "#" + re.sub(r"[^a-z0-9]+", "-", full_key.lower()).strip("-")
                record = {
                    "source": "cpmd",
                    "version": str(cpmd_data["version"]),
                    "page": "keywords",
                    "section": f"CPMD &{sec}/{name}",
                    "text": " ".join(parts),
                    "locator": base_url + frag,
                    "sha256": cpmd_sha,
                    "fetched": today,
                    "licence": "CPMD manual keyword specification (catalog excerpt)",
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    results.append(f"{'checked':10s} {cpmd_target.name}: {len(keys)} chunk(s)")
    return results


def _fragment(section: str, kind: str) -> str:
    if kind == "mdp-options" and section.startswith("mdp option "):
        return "#mdp-" + section[len("mdp option "):]
    return ""


def main(argv: list[str]) -> int:
    check = "--check" in argv
    local_only = "--local" in argv
    if not local_only:
        for page in PAGES:
            print(build(page, check=check))
    for line in build_local_tools_and_cpmd(check=check):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
