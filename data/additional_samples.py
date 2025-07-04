# GPT data
import pandas as pd
import numpy as np


# Data fetched using GPT and validated/adjusted afterwards.
gpt_data = [

    # Could not validate. Which porosity to use? Which is HAP and which is B-TCP values.
    {
        "title": "Bone-like structure by modified freeze casting",
        "journal": "Scientific Reports",
        "year": 2020,
        "doi": "10.1038/s41598-020-64757-z",
        "name_part1": "HAP",
        "name_fluid1": "camphene",
        "material_group": "Ceramic",
        "dia_part_1": 0.02,
        "wf_disp_1": np.nan,
        "wf_bind_1": np.nan,
        "temp_cold": -15+273,
        "time_sinter_1": 3,
        "temp_sinter_1": 1250,
        "vf_total": 0.35,
        "porosity": 60.58
    },
    {
        "title": "Bone-like structure by modified freeze casting",
        "journal": "Scientific Reports",
        "year": 2020,
        "doi": "10.1038/s41598-020-64757-z",
        "name_part1": "HAP",
        "name_fluid1": "camphene",
        "material_group": "Ceramic",
        "dia_part_1": 0.02,
        "wf_disp_1": np.nan,
        "wf_bind_1": np.nan,
        "temp_cold": -15+273,
        "time_sinter_1": 3,
        "temp_sinter_1": 1250,
        "vf_total": 0.40,
        "porosity": 54.11
    },
    # {
    #     "title": "Assessing porosity limit in freeze-cast sintered Li₄Ti₅O₁2 (LTO)",
    #     "journal": "Int. J. Appl. Ceram. Tech.",
    #     "year": 2025,
    #     "doi": "10.1111/ijac.14883",
    #     "name_part1": "LTO",
    #     "name_fluid1": "water",
    #     "material_group": "Ceramic",
    #     "dia_part_1": np.nan,
    #     "wf_disp_1": np.nan,
    #     "wf_bind_1": np.nan,
    #     "temp_cold": np.nan,
    #     "time_sinter_1": np.nan,
    #     "temp_sinter_1": np.nan,
    #     "vf_total": 0.30,
    #     "porosity": 50.00
    # },
    # {
    #     "title": "Assessing porosity limit in freeze-cast sintered Li₄Ti₅O₁2 (LTO)",
    #     "journal": "Int. J. Appl. Ceram. Tech.",
    #     "year": 2025,
    #     "doi": "10.1111/ijac.14883",
    #     "name_part1": "LTO",
    #     "name_fluid1": "water",
    #     "material_group": "Ceramic",
    #     "dia_part_1": np.nan,
    #     "wf_disp_1": np.nan,
    #     "wf_bind_1": np.nan,
    #     "temp_cold": np.nan,
    #     "time_sinter_1": np.nan,
    #     "temp_sinter_1": np.nan,
    #     "vf_total": 0.37,
    #     "porosity": 36.00
    # },


    # # couldnt validate
    # {
    #     "name_part1": "HAP",
    #     "name_fluid1": "water",
    #     "material_group": "Ceramic",
    #     "dia_part_1": np.nan,
    #     "wf_disp_1": np.nan,
    #     "wf_bind_1": np.nan,
    #     "temp_cold": -20,
    #     "time_sinter_1": 3,
    #     "temp_sinter_1": 1350,
    #     "vf_total": 0.50,
    #     "porosity": 50.0,
    #     "title": "Freeze casting of porous hydroxyapatite scaffolds. I. Processing and general microstructure",
    #     "journal": "Journal of Biomedical Materials Research Part B: Applied Biomaterials",
    #     "doi": "10.1002/jbm.b.30957"
    # },


]


# ####################################################################################################
# ####################################################################################################
dseek_data = [


    # Effect of particle size and freezing conditions on freeze-casted scaffold [not-validated could not find]
    # {
    #     "title": "Effect of particle size and freezing conditions on freeze-casted scaffold",
    #     "journal": "Ceramics International",
    #     "year": 2019,
    #     "doi": "10.1016/j.ceramint.2019.03.004",
    #     "name_part1": "HAP",
    #     "name_fluid1": "camphene",
    #     "material_group": "Ceramic",
    #     "dia_part_1": 0.02,
    #     "wf_disp_1": np.nan,
    #     "name_disp_1": np.nan,
    #     "wf_bind_1": np.nan,
    #     "name_binder1": np.nan,
    #     "temp_cold": 5 + 273.15,  # 5°C to K
    #     "time_sinter_1": 3.0,
    #     "temp_sinter_1": 1250.0,
    #     "vf_total": 0.30,
    #     "porosity": 71.80
    # },
    # {
    #     "title": "Effect of particle size and freezing conditions on freeze-casted scaffold",
    #     "journal": "Ceramics International",
    #     "year": 2019,
    #     "doi": "10.1016/j.ceramint.2019.03.004",
    #     "name_part1": "TCP", # beta tcp
    #     "name_fluid1": "camphene",
    #     "material_group": "Ceramic",
    #     "dia_part_1": 10.00,
    #     "wf_disp_1": np.nan,
    #     "name_disp_1": np.nan,
    #     "wf_bind_1": np.nan,
    #     "name_binder1": np.nan,
    #     "temp_cold": 5 + 273.15,
    #     "time_sinter_1": 3.0,
    #     "temp_sinter_1": 1250.0,
    #     "vf_total": 0.30,
    #     "porosity": 43.00
    # },

    # Freeze casting of hydroxyapatite-titania composites for bone substitutes [validated]
    {
        "title": "Freeze casting of hydroxyapatite-titania composites for bone substitutes",
        "journal": "Journal of Biomedical Materials Research Part A",
        "year": 2023,
        "doi": "10.1002/jbm.a.37645",
        "name_part1": "HAP",
        "name_fluid1": "water",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": 1.0,
        "name_disp_1": "Dynol 604",
        "wf_bind_1": 2.0,
        "name_binder1": "PVA+PEG",
        "temp_cold": 0 + 273.15,  # 0°C to K
        "time_sinter_1": 3.0,
        "temp_sinter_1": 1250.0+ 273,
        "vf_total": 0.10,
        "porosity": 55.2
    },
    {
        "title": "Freeze casting of hydroxyapatite-titania composites for bone substitutes",
        "journal": "Journal of Biomedical Materials Research Part A",
        "year": 2023,
        "doi": "10.1002/jbm.a.37645",
        "name_part1": "HAP",
        "name_fluid1": "water",
        "name_part2": "TiO2",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": 1.0,
        "name_disp_1": "Dynol 604",
        "wf_bind_1": 2.0,
        "name_binder1": "PVA+PEG",
        "temp_cold": 0 + 273.15,
        "time_sinter_1": 3.0,
        "temp_sinter_1": 1250.0+ 273,
        "vf_total": 0.10,
        "porosity": 53.0
    },
    {
        "title": "Freeze casting of hydroxyapatite-titania composites for bone substitutes",
        "journal": "Journal of Biomedical Materials Research Part A",
        "year": 2023,
        "doi": "10.1002/jbm.a.37645",
        "name_part1": "HAP",
        "name_part2": "TiO2",
        "name_fluid1": "water",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": 1.0,
        "name_disp_1": "Dynol 604",
        "wf_bind_1": 2.0,
        "name_binder1": "PVA+PEG",
        "temp_cold": 0 + 273.15,
        "time_sinter_1": 3.0,
        "temp_sinter_1": 1250.0+ 273,
        "vf_total": 0.10,
        "porosity": 54.0
    },
    {
        "title": "Freeze casting of hydroxyapatite-titania composites for bone substitutes",
        "journal": "Journal of Biomedical Materials Research Part A",
        "year": 2023,
        "doi": "10.1002/jbm.a.37645",
        "name_part1": "HAP",
        "name_part2": "TiO2",
        "name_fluid1": "water",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": 1.0,
        "name_disp_1": "Dynol 604",
        "wf_bind_1": 2.0,
        "name_binder1": "PVA+PEG",
        "temp_cold": 0 + 273.15,
        "time_sinter_1": 3.0,
        "temp_sinter_1": 1250.0 + 273,
        "vf_total": 0.10,
        "porosity": 52.7
    },


    # Freeze-casting uniformity and domains
    # {
    #     "title": "Freeze-casting uniformity and domains",
    #     "journal": "Materials Science & Engineering C",
    #     "year": 2024,
    #     "doi": "10.1016/j.msec.2024.114567",
    #     "name_part1": "LCSM",
    #     "name_fluid1": "water",
    #     "material_group": "Ceramic",
    #     "dia_part_1": np.nan,
    #     "wf_disp_1": np.nan,
    #     "name_disp_1": np.nan,
    #     "wf_bind_1": np.nan,
    #     "name_binder1": np.nan,
    #     "temp_cold": -10 + 273.15,  # -10°C to K
    #     "time_sinter_1": np.nan,
    #     "temp_sinter_1": np.nan,
    #     "vf_total": np.nan,
    #     "porosity": 0.416
    # },

    # Biomimetic Materials by Freeze Casting (Update)  [could nto be found]
    # {
    #     "title": "Biomimetic Materials by Freeze Casting (Update)",
    #     "journal": "Journal of Materials Research",
    #     "year": 2021,
    #     "doi": "10.1557/jmr.2021.82",
    #     "name_part1": "ZrO2",
    #     "name_fluid1": "water",
    #     "material_group": "Ceramic",
    #     "dia_part_1": 0.50,
    #     "wf_disp_1": 1.0,
    #     "name_disp_1": np.nan,
    #     "wf_bind_1": 1.5,
    #     "name_binder1": np.nan,
    #     "temp_cold": 0 + 273.15,
    #     "time_sinter_1": 2.0,
    #     "temp_sinter_1": 1400.0,
    #     "vf_total": 0.20,
    #     "porosity": 75.00
    # },


    # New HAP-BNNT composite (Sci. Direct 2023) [could nto be validated not open sourrce]
    # {
    #     "title": "Freeze casting to engineer gradient porosity in hydroxyapatite-boron nitride nanotube composite scaffold",
    #     "journal": "Ceramics International",
    #     "year": 2023,
    #     "doi": "10.1016/j.ceramint.2023.05.123",
    #     "name_part1": "HAP",
    #     "name_part2": "BNNT",
    #     "name_fluid1": "camphene",
    #     "material_group": "Ceramic Composite",
    #     "dia_part_1": 0.01,  # BNNT diameter
    #     "wf_disp_1": 0.5,
    #     "name_disp_1": "NaDDBS",
    #     "wf_bind_1": np.nan,
    #     "name_binder1": np.nan,
    #     "temp_cold": -20 + 273.15,  # Estimated from process
    #     "time_sinter_1": 3.0,
    #     "temp_sinter_1": 1250.0,
    #     "vf_total": 0.30,  # 30 vol% solids
    #     "porosity": 72.0  # Avg. from structural analysis :cite[1]
    # },

    # # Al2O3 scaffolds with lamellar pores (Sci. Direct 2025) [notvalidated* porosity]
    # {
    #     "title": "Effect of pore architecture on quasistatic compressive deformation of freeze-cast porous alumina scaffolds",
    #     "journal": "Journal of the European Ceramic Society",
    #     "year": 2025,
    #     "doi": "10.1016/j.jeurceramsoc.2025.01.045",
    #     "name_part1": "Al2O3",  # BP
    #     "name_fluid1": "water",
    #     "material_group": "Ceramic",
    #     "dia_part_1": 8.1,  # Bigger platelets
    #     "wf_disp_1": 1.0,
    #     "name_disp_1": "PAA-NH₄",
    #     "wf_bind_1": np.nan,
    #     "name_binder1": np.nan,
    #     "temp_cold": -196 + 273.15,  # Liquid N2 cooling
    #     "time_sinter_1": 2.0,
    #     "temp_sinter_1": 1500.0,
    #     "vf_total": 0.15,  # 15 vol%
    #     "porosity": 86.0  # >85% porosity :cite[2]
    # },

    # The Production of Porous Hydroxyapatite Scaffolds with Graded Porosity by Sequential Freeze-Casting [validated]
    {
        "title": "The Production of Porous Hydroxyapatite Scaffolds with Graded Porosity by Sequential Freeze-Casting",
        "journal": "Materials",
        "year": 2017,
        "doi": "10.3390/ma10040367",
        "name_part1": "HAP",
        "name_fluid1": "camphene",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": 0.5,
        "name_disp_1": "Hypermer KD-4",
        "wf_bind_1": np.nan,
        "name_binder1": np.nan,
        "temp_cold": 42 + 273.15,  # Camphene solidification temp
        "time_sinter_1": 2.0,
        "temp_sinter_1": 1250.0 + 273,
        "vf_total": 0.10,
        "porosity": 76.0,
    },
    {
        "title": "The Production of Porous Hydroxyapatite Scaffolds with Graded Porosity by Sequential Freeze-Casting",
        "journal": "Materials",
        "year": 2017,
        "doi": "10.3390/ma10040367",
        "name_part1": "HAP",
        "name_fluid1": "camphene",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": 0.5,
        "name_disp_1": "Hypermer KD-4",
        "wf_bind_1": np.nan,
        "name_binder1": np.nan,
        "temp_cold": 42 + 273.15,  # Camphene solidification temp
        "time_sinter_1": 2.0,
        "temp_sinter_1": 1250.0 + 273,
        "vf_total": 0.15,
        "porosity": 68.0,
    },
    {
        "title": "The Production of Porous Hydroxyapatite Scaffolds with Graded Porosity by Sequential Freeze-Casting",
        "journal": "Materials",
        "year": 2017,
        "doi": "10.3390/ma10040367",
        "name_part1": "HAP",
        "name_fluid1": "camphene",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": 0.5,
        "name_disp_1": "Hypermer KD-4",
        "wf_bind_1": np.nan,
        "name_binder1": np.nan,
        "temp_cold": 42 + 273.15,  # Camphene solidification temp
        "time_sinter_1": 2.0,
        "temp_sinter_1": 1250.0 + 273,
        "vf_total": 0.20,
        "porosity": 60.0,
    },    {
        "title": "The Production of Porous Hydroxyapatite Scaffolds with Graded Porosity by Sequential Freeze-Casting",
        "journal": "Materials",
        "year": 2017,
        "doi": "10.3390/ma10040367",
        "name_part1": "HAP",
        "name_fluid1": "camphene",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": 0.5,
        "name_disp_1": "Hypermer KD-4",
        "wf_bind_1": np.nan,
        "name_binder1": np.nan,
        "temp_cold": 42 + 273.15,  # Camphene solidification temp
        "time_sinter_1": 2.0,
        "temp_sinter_1": 1250.0 + 273,
        "vf_total": 0.25,
        "porosity": 52.0,
    },
    {
        "title": "The Production of Porous Hydroxyapatite Scaffolds with Graded Porosity by Sequential Freeze-Casting",
        "journal": "Materials",
        "year": 2017,
        "doi": "10.3390/ma10040367",
        "name_part1": "HAP",
        "name_fluid1": "camphene",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": 0.5,
        "name_disp_1": "Hypermer KD-4",
        "wf_bind_1": np.nan,
        "name_binder1": np.nan,
        "temp_cold": 42 + 273.15,  # Camphene solidification temp
        "time_sinter_1": 2.0,
        "temp_sinter_1": 1250.0 + 273,
        "vf_total": 0.40,
        "porosity": 27.0,
    },
    {
        "title": "The Production of Porous Hydroxyapatite Scaffolds with Graded Porosity by Sequential Freeze-Casting",
        "journal": "Materials",
        "year": 2017,
        "doi": "10.3390/ma10040367",
        "name_part1": "HAP",
        "name_fluid1": "camphene",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": 0.5,
        "name_disp_1": "Hypermer KD-4",
        "wf_bind_1": np.nan,
        "name_binder1": np.nan,
        "temp_cold": 42 + 273.15,  # Camphene solidification temp
        "time_sinter_1": 2.0,
        "temp_sinter_1": 1250.0 + 273,
        "vf_total": 0.50,
        "porosity": 10.0,
    },
    # #########################################
    # Magnetic field-assisted Al2O3 (J. Mater. Res. 2020) [validated]
    {
        "title": "Design of porous aluminum oxide ceramics using magnetic field-assisted freeze-casting",
        "journal": "Journal of Materials Research",
        "year": 2020,
        "doi": "10.1557/jmr.2020.150",
        "name_part1": "Al2O3",
        "name_part2": "Fe3O4",
        "name_fluid1": "water",
        "material_group": "Ceramic Composite",
        "dia_part_1": 0.4,  # Typical alumina particle size
        "wf_disp_1": 1.5,
        "name_disp_1": "PVA",
        "wf_bind_1": 3.0,
        "name_binder1": "PVA",
        "temp_cold": -20 + 273.15,
        "time_sinter_1": 3.0,
        "temp_sinter_1": 1500.0,
        "vf_total": 0.10 + 0.06,
        "porosity": 76  # Controlled unidirectional pores :cite[3]
    },
    {
        "title": "Design of porous aluminum oxide ceramics using magnetic field-assisted freeze-casting",
        "journal": "Journal of Materials Research",
        "year": 2020,
        "doi": "10.1557/jmr.2020.150",
        "name_part1": "Al2O3",
        "name_part2": "Fe3O4",
        "name_fluid1": "water",
        "material_group": "Ceramic Composite",
        "dia_part_1": 0.4,  # Typical alumina particle size
        "wf_disp_1": 1.5,
        "name_disp_1": "PVA",
        "wf_bind_1": 3.0,
        "name_binder1": "PVA",
        "temp_cold": -20 + 273.15,
        "time_sinter_1": 6.0,
        "temp_sinter_1": 1500.0,
        "vf_total": 0.10  + 0.06,
        "porosity": 72  # Controlled unidirectional pores :cite[3]
    },    {
        "title": "Design of porous aluminum oxide ceramics using magnetic field-assisted freeze-casting",
        "journal": "Journal of Materials Research",
        "year": 2020,
        "doi": "10.1557/jmr.2020.150",
        "name_part1": "Al2O3",
        "name_part2": "Fe3O4",
        "name_fluid1": "water",
        "material_group": "Ceramic Composite",
        "dia_part_1": 0.4,  # Typical alumina particle size
        "wf_disp_1": 1.5,
        "name_disp_1": "PVA",
        "wf_bind_1": 3.0,
        "name_binder1": "PVA",
        "temp_cold": -20 + 273.15,
        "time_sinter_1": 9.0,
        "temp_sinter_1": 1500.0,
        "vf_total":  0.10 + 0.06,
        "porosity": 68  # Controlled unidirectional pores :cite[3]
    },



]


# ####################################################################################################
# ####################################################################################################
manu_data =  [
    # Rare materials, no trained on those subsets
    # {
    #     "title": "Anisotropic Freeze-Cast Collagen Scaffolds for Tissue Engineering Applications",
    #     "journal": "J Mech Behav Biomed Mater",
    #     "year": 2018,
    #     "doi": "10.1016/j.jmbbm.2018.09.012",
    #     "name_part1": "collagen",
    #     "name_fluid1": "acetic acid",
    #     "material_group": "Biomaterial",
    #     "dia_part_1": np.nan,
    #     "wf_disp_1": np.nan,
    #     "name_disp_1": np.nan,
    #     "wf_bind_1": np.nan,
    #     "name_binder1": np.nan,
    #     "temp_cold": -20 + 273.15,
    #     "time_sinter_1": np.nan,
    #     "temp_sinter_1": np.nan,
    #     "vf_total": np.nan,
    #     "porosity": 97.95,
    #     "scaffold_type": "Longitudinal",
    #     "applied_cooling_rate_C_min": 10
    # },
    # {
    #     "title": "Anisotropic Freeze-Cast Collagen Scaffolds for Tissue Engineering Applications",
    #     "journal": "J Mech Behav Biomed Mater",
    #     "year": 2018,
    #     "doi": "10.1016/j.jmbbm.2018.09.012",
    #     "name_part1": "collagen",
    #     "name_fluid1": "acetic acid",
    #     "material_group": "Biomaterial",
    #     "dia_part_1": np.nan,
    #     "wf_disp_1": np.nan,
    #     "name_disp_1": np.nan,
    #     "wf_bind_1": np.nan,
    #     "name_binder1": np.nan,
    #     "temp_cold": -20 + 273.15,
    #     "time_sinter_1": np.nan,
    #     "temp_sinter_1": np.nan,
    #     "vf_total": np.nan,
    #     "porosity": 97.9,
    #     "scaffold_type": "Longitudinal",
    #     "applied_cooling_rate_C_min": 1
    # },
    # {
    #     "title": "Anisotropic Freeze-Cast Collagen Scaffolds for Tissue Engineering Applications",
    #     "journal": "J Mech Behav Biomed Mater",
    #     "year": 2018,
    #     "doi": "10.1016/j.jmbbm.2018.09.012",
    #     "name_part1": "collagen",
    #     "name_fluid1": "acetic acid",
    #     "material_group": "Biomaterial",
    #     "dia_part_1": np.nan,
    #     "wf_disp_1": np.nan,
    #     "name_disp_1": np.nan,
    #     "wf_bind_1": np.nan,
    #     "name_binder1": np.nan,
    #     "temp_cold": -20 + 273.15,
    #     "time_sinter_1": np.nan,
    #     "temp_sinter_1": np.nan,
    #     "vf_total": np.nan,
    #     "porosity": 97.85,
    #     "scaffold_type": "Longitudinal",
    #     "applied_cooling_rate_C_min": 0.1
    # },
    # {
    #     "title": "Anisotropic Freeze-Cast Collagen Scaffolds for Tissue Engineering Applications",
    #     "journal": "J Mech Behav Biomed Mater",
    #     "year": 2018,
    #     "doi": "10.1016/j.jmbbm.2018.09.012",
    #     "name_part1": "collagen",
    #     "name_fluid1": "acetic acid",
    #     "material_group": "Biomaterial",
    #     "dia_part_1": np.nan,
    #     "wf_disp_1": np.nan,
    #     "name_disp_1": np.nan,
    #     "wf_bind_1": np.nan,
    #     "name_binder1": np.nan,
    #     "temp_cold": -20 + 273.15,
    #     "time_sinter_1": np.nan,
    #     "temp_sinter_1": np.nan,
    #     "vf_total": np.nan,
    #     "porosity": 97.85,
    #     "scaffold_type": "Radial",
    #     "applied_cooling_rate_C_min": 10
    # },
    # {
    #     "title": "Anisotropic Freeze-Cast Collagen Scaffolds for Tissue Engineering Applications",
    #     "journal": "J Mech Behav Biomed Mater",
    #     "year": 2018,
    #     "doi": "10.1016/j.jmbbm.2018.09.012",
    #     "name_part1": "collagen",
    #     "name_fluid1": "acetic acid",
    #     "material_group": "Biomaterial",
    #     "dia_part_1": np.nan,
    #     "wf_disp_1": np.nan,
    #     "name_disp_1": np.nan,
    #     "wf_bind_1": np.nan,
    #     "name_binder1": np.nan,
    #     "temp_cold": -20 + 273.15,
    #     "time_sinter_1": np.nan,
    #     "temp_sinter_1": np.nan,
    #     "vf_total": np.nan,
    #     "porosity": 97.85,
    #     "scaffold_type": "Radial",
    #     "applied_cooling_rate_C_min": 1
    # },
    # {
    #     "title": "Anisotropic Freeze-Cast Collagen Scaffolds for Tissue Engineering Applications",
    #     "journal": "J Mech Behav Biomed Mater",
    #     "year": 2018,
    #     "doi": "10.1016/j.jmbbm.2018.09.012",
    #     "name_part1": "collagen",
    #     "name_fluid1": "acetic acid",
    #     "material_group": "Biomaterial",
    #     "dia_part_1": np.nan,
    #     "wf_disp_1": np.nan,
    #     "name_disp_1": np.nan,
    #     "wf_bind_1": np.nan,
    #     "name_binder1": np.nan,
    #     "temp_cold": -20 + 273.15,
    #     "time_sinter_1": np.nan,
    #     "temp_sinter_1": np.nan,
    #     "vf_total": np.nan,
    #     "porosity": 97.85,
    #     "scaffold_type": "Radial",
    #     "applied_cooling_rate_C_min": 0.1
    # },
    # {
    #     "title": "Tailoring the 3D porous structure of conducting PEDOT:PSS gels via ice-templating",
    #     "journal": "J. Mater. Chem. C",
    #     "year": 2023,
    #     "doi": "10.1039/d3tc01110k",
    #     "name_part1": "PEDOT",
    #     "name_part2": "PSS",
    #     "name_fluid1": "water",
    #     "material_group": "Polymer",
    #     "dia_part_1": np.nan,
    #     "wf_disp_1": np.nan,
    #     "name_disp_1": np.nan,
    #     "wf_bind_1": np.nan,
    #     "name_binder1": np.nan,
    #     "temp_cold": np.nan,
    #     "time_sinter_1": np.nan,
    #     "temp_sinter_1": np.nan,
    #     "vf_total": np.nan,
    #     "porosity": 98,
    #     "method": "Quenched Freezing (Lyophilized Dispersion)"
    # },
    # {
    #     "title": "Comparative Properties of Porous Phyllosilicate-Based Ceramics Shaped by Freeze-Tape Casting",
    #     "journal": "Ceramics",
    #     "year": 2022,
    #     "doi": "10.3390/ceramics5010007",
    #     "name_part1": "KORS",
    #     "name_fluid1": "water",
    #     "material_group": "Ceramic",
    #     "dia_part_1": np.nan,
    #     "wf_disp_1": np.nan,
    #     "name_disp_1": np.nan,
    #     "wf_bind_1": np.nan,
    #     "name_binder1": np.nan,
    #     "temp_cold": -5 + 273.15,
    #     "time_sinter_1": 1,
    #     "temp_sinter_1": 1200,
    #     "vf_total": np.nan,
    #     "porosity": 80
    # },

    # huge change due to second material [validated]
    # Dual-Scale Porosity Alumina Structures Using Ceramic/Camphene Suspensions Containing Polymer Microspheres [validated]
    {
        "title": "Dual-Scale Porosity Alumina Structures Using Ceramic/Camphene Suspensions Containing Polymer Microspheres",
        "journal": "Materials",
        "year": 2022,
        "doi": "10.3390/ma15113875",
        "name_part1": "Al2O3",
        "name_fluid1": "Camphene",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": np.nan,
        "name_disp_1": "Hypermer KD-4",
        "wf_bind_1": np.nan,
        "name_binder1": np.nan,
        "temp_cold": -20 + 273.15,
        "time_sinter_1": 3.0,
        "temp_sinter_1": 1550.0,
        "vf_total": 0.3,
        "porosity": 45.7,
        "pmma_content_vol_percent": 0
    },
    {
        "title": "Dual-Scale Porosity Alumina Structures Using Ceramic/Camphene Suspensions Containing Polymer Microspheres",
        "journal": "Materials",
        "year": 2022,
        "doi": "10.3390/ma15113875",
        "name_part1": "Al2O3",
        "name_fluid1": "Camphene",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": np.nan,
        "name_disp_1": "Hypermer KD-4",
        "wf_bind_1": np.nan,
        "name_binder1": np.nan,
        "temp_cold": -20 + 273.15,
        "time_sinter_1": 3.0,
        "temp_sinter_1": 1550.0,
        "vf_total": 0.3,
        "porosity": 71.4,
        "pmma_content_vol_percent": 40
    }
]

# Create DataFrame
df_dseek = pd.DataFrame(dseek_data)
df_gpt = pd.DataFrame(gpt_data)
df_manu = pd.DataFrame(manu_data)
# Concatenate DataFrames
df_additional_data = pd.concat([df_dseek, df_gpt, df_manu], ignore_index=True)

# Remove duplicates based on 'doi' and 'porosity'
df_additional_data =  df_additional_data.drop_duplicates(subset=["doi", "porosity"], keep="last")
df_additional_data[['name_disp_1', 'name_part2']] = ""
df_additional_data['porosity'] = df_additional_data['porosity'] / 100
