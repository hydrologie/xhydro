import pytest
import tempfile
import shutil
from pathlib import Path
from xhydro.modelling._hydrotel import Hydrotel
import pandas as pd

def make_dummy_project(tmp_path):
    # Crée le dossier simulation/simulation attendu par Hydrotel
    sim_dir = tmp_path / "simulation" / "simulation"
    sim_dir.mkdir(parents=True, exist_ok=True)
    # Crée un fichier simulation.csv minimal dans le dossier simulation/simulation
    sim_csv = sim_dir / "simulation.csv"
    sim_csv.write_text(
        "SIMULATION_HYDROTEL_VERSION;4.1.5.0014\n"
        "DATE_DEBUT;2023-08-01 00:00\n"
        "DATE_FIN;2023-08-10 00:00\n"
        "FICHIER_STATIONS_METEO;dummy.nc\n"
        "PAS_DE_TEMPS;24\n"
        "LECTURE_ETAT_FONTE_NEIGE;\n"
    )
    # Crée un fichier output.csv minimal dans le dossier simulation/simulation
    output_csv = sim_dir / "output.csv"
    output_csv.write_text("TRONCONS;DEBITS_AVAL\n1;1\n")
    # Crée un dossier de projet minimal avec un dossier etat/ et un fichier d'état
    etat_dir = tmp_path / "etat"
    etat_dir.mkdir()
    # Crée un fichier d'état de neige pour 2023-08-01
    state_file = etat_dir / "fonte_neige_2023080100.csv"
    state_file.write_text("dummy state file")
    # Crée un fichier projet minimal
    project_file = tmp_path / "dummy.csv"
    project_file.write_text("SIMULATION COURANTE;simulation\n")
    # Crée une config minimale
    config = {
        "model_name": "Hydrotel",
        "project_dir": str(tmp_path),
        "project_file": "dummy.csv",
        "simulation_config": {
            "SIMULATION HYDROTEL VERSION": "4.1.5.0014",
            "DATE DEBUT": "2023-08-01",
            "DATE FIN": "2023-08-10",
            "FICHIER STATIONS METEO": "dummy.nc",
            "PAS DE TEMPS": 24
        },
        "output_config": {},
        "use_defaults": False,
        "executable": "hydrotel.exe"
    }
    return config, etat_dir, state_file

def test_set_initial_states(tmp_path):
    config, etat_dir, state_file = make_dummy_project(tmp_path)
    ht = Hydrotel(
        config["project_dir"],
        config["project_file"],
        config["executable"],
        simulation_config=config.get("simulation_config"),
        output_config=config.get("output_config"),
        use_defaults=config.get("use_defaults", False)
    )
    # Test: auto-detect l'état le plus proche avant DATE DEBUT
    ht.set_initial_states(auto_detect=True)
    # Vérifie que la config a bien été modifiée
    sim_cfg = ht.simulation_config
    assert "LECTURE ETAT FONTE NEIGE" in sim_cfg
    assert "fonte_neige_2023080100.csv" in sim_cfg["LECTURE ETAT FONTE NEIGE"]
    # Test: date explicite
    ht.set_initial_states(date="2023-08-01")
    sim_cfg2 = ht.simulation_config
    assert "LECTURE ETAT FONTE NEIGE" in sim_cfg2
    assert "fonte_neige_2023080100.csv" in sim_cfg2["LECTURE ETAT FONTE NEIGE"]
    # Test: désactivation manuelle
    ht.set_initial_states(states_dict={"fonte_neige": ""})
    sim_cfg3 = ht.simulation_config
    if sim_cfg3["LECTURE ETAT FONTE NEIGE"] != "":
        print(f"Valeur réelle de LECTURE ETAT FONTE NEIGE : {sim_cfg3['LECTURE ETAT FONTE NEIGE']!r}")
    assert sim_cfg3["LECTURE ETAT FONTE NEIGE"] in ("", ".")


if __name__ == "__main__":
    import tempfile
    import traceback
    from pathlib import Path
    tmp = tempfile.TemporaryDirectory()
    try:
        test_set_initial_states(Path(tmp.name))
        print("Test passed (exécuté sans pytest)")
    except Exception as e:
        print("Test failed:")
        traceback.print_exc()
