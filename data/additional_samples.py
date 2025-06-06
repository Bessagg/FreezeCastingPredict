# GPT data
import pandas as pd
import numpy as np


# Data fetched using GPT and validated/adjusted afterwards.
gpt_data = [
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
    {
        "title": "Assessing porosity limit in freeze-cast sintered Li₄Ti₅O₁₂ (LTO)",
        "journal": "Int. J. Appl. Ceram. Tech.",
        "year": 2025,
        "doi": "10.1111/ijac.14883",
        "name_part1": "LTO",
        "name_fluid1": "water",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": np.nan,
        "wf_bind_1": np.nan,
        "temp_cold": np.nan,
        "time_sinter_1": np.nan,
        "temp_sinter_1": np.nan,
        "vf_total": 0.30,
        "porosity": 50.00
    },
    {
        "title": "Assessing porosity limit in freeze-cast sintered Li₄Ti₅O₁₂ (LTO)",
        "journal": "Int. J. Appl. Ceram. Tech.",
        "year": 2025,
        "doi": "10.1111/ijac.14883",
        "name_part1": "LTO",
        "name_fluid1": "water",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": np.nan,
        "wf_bind_1": np.nan,
        "temp_cold": np.nan,
        "time_sinter_1": np.nan,
        "temp_sinter_1": np.nan,
        "vf_total": 0.37,
        "porosity": 36.00
    },
    {
        "title": "Anisotropic freeze-cast collagen scaffolds",
        "journal": "J. Mech. Behav. Biomed. Mater.",
        "year": 2019,
        "doi": "10.1016/j.jmbbm.2018.09.012",
        "name_part1": "Collagen",
        "name_fluid1": "water",
        "material_group": "Polymer",
        "dia_part_1": np.nan,
        "wf_disp_1": np.nan,
        "wf_bind_1": np.nan,
        "temp_cold": -85+273,
        "time_sinter_1": np.nan,
        "temp_sinter_1": np.nan,
        "vf_total": np.nan,
        "porosity": 97.90
    },
    {
        "name_part1": "Al₂O₃",
        "name_fluid1": "TBA",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": np.nan,
        "wf_bind_1": np.nan,
        "temp_cold": np.nan,
        "time_sinter_1": np.nan,
        "temp_sinter_1": np.nan,
        "vf_total": 0.20,
        "porosity": 65.0,
        "title": "Preparation of porous alumina ceramic with ultra-high porosity and long straight pores by freeze casting",
        "journal": "Journal of Porous Materials",
        "doi": "10.1007/s10934-011-9480-y"
    },
    {
        "name_part1": "HAP",
        "name_fluid1": "Water",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": np.nan,
        "wf_bind_1": np.nan,
        "temp_cold": -20,
        "time_sinter_1": 3,
        "temp_sinter_1": 1350,
        "vf_total": 0.50,
        "porosity": 50.0,
        "title": "Freeze casting of porous hydroxyapatite scaffolds. I. Processing and general microstructure",
        "journal": "Journal of Biomedical Materials Research Part B: Applied Biomaterials",
        "doi": "10.1002/jbm.b.30957"
    },

    {
        "name_part1": "Al₂O₃",
        "name_fluid1": "TBA",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": np.nan,
        "wf_bind_1": np.nan,
        "temp_cold": np.nan,
        "time_sinter_1": np.nan,
        "temp_sinter_1": np.nan,
        "vf_total": 0.20,
        "porosity": 65.0,
        "title": "Preparation of porous alumina ceramic with ultra-high porosity and long straight pores by freeze casting",
        "journal": "Journal of Porous Materials",
        "doi": "10.1007/s10934-011-9480-y"
    },
    {
        "name_part1": "HAP",
        "name_fluid1": "Water",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": np.nan,
        "wf_bind_1": np.nan,
        "temp_cold": -20,
        "time_sinter_1": 3,
        "temp_sinter_1": 1350,
        "vf_total": 0.50,
        "porosity": 50.0,
        "title": "Freeze casting of porous hydroxyapatite scaffolds. I. Processing and general microstructure",
        "journal": "Journal of Biomedical Materials Research Part B: Applied Biomaterials",
        "doi": "10.1002/jbm.b.30957"
    }, 

]


# ####################################################################################################
# ####################################################################################################
dseek_data = [
    {
        "title": "Bone-like structure by modified freeze casting",
        "journal": "Scientific Reports",
        "year": 2020,
        "doi": "10.1038/s41598-020-64757-z",
        "name_part1": "HAp",
        "name_fluid1": "camphene",
        "material_group": "Ceramic",
        "dia_part_1": 0.02,
        "wf_disp_1": np.nan,
        "name_disp1": np.nan,
        "wf_bind_1": np.nan,
        "name_binder1": np.nan,
        "temp_cold": -15 + 273.15,  # -15°C to K
        "time_sinter_1": 3.0,
        "temp_sinter_1": 1250.0,
        "vf_total": 0.35,
        "porosity": 60.58
    },
    {
        "title": "Bone-like structure by modified freeze casting",
        "journal": "Scientific Reports",
        "year": 2020,
        "doi": "10.1038/s41598-020-64757-z",
        "name_part1": "HAp",
        "name_fluid1": "camphene",
        "material_group": "Ceramic",
        "dia_part_1": 0.02,
        "wf_disp_1": np.nan,
        "name_disp1": np.nan,
        "wf_bind_1": np.nan,
        "name_binder1": np.nan,
        "temp_cold": -15 + 273.15,
        "time_sinter_1": 3.0,
        "temp_sinter_1": 1250.0,
        "vf_total": 0.40,
        "porosity": 54.11
    },
    {
        "title": "Effect of particle size and freezing conditions on freeze-casted scaffold",
        "journal": "Ceramics International",
        "year": 2019,
        "doi": "10.1016/j.ceramint.2019.03.004",
        "name_part1": "HAp",
        "name_fluid1": "camphene",
        "material_group": "Ceramic",
        "dia_part_1": 0.02,
        "wf_disp_1": np.nan,
        "name_disp1": np.nan,
        "wf_bind_1": np.nan,
        "name_binder1": np.nan,
        "temp_cold": 5 + 273.15,  # 5°C to K
        "time_sinter_1": 3.0,
        "temp_sinter_1": 1250.0,
        "vf_total": 0.30,
        "porosity": 71.80
    },
    {
        "title": "Effect of particle size and freezing conditions on freeze-casted scaffold",
        "journal": "Ceramics International",
        "year": 2019,
        "doi": "10.1016/j.ceramint.2019.03.004",
        "name_part1": "β-TCP",
        "name_fluid1": "camphene",
        "material_group": "Ceramic",
        "dia_part_1": 10.00,
        "wf_disp_1": np.nan,
        "name_disp1": np.nan,
        "wf_bind_1": np.nan,
        "name_binder1": np.nan,
        "temp_cold": 5 + 273.15,
        "time_sinter_1": 3.0,
        "temp_sinter_1": 1250.0,
        "vf_total": 0.30,
        "porosity": 43.00
    },
    {
        "title": "Freeze casting of hydroxyapatite-titania composites for bone substitutes",
        "journal": "Journal of Biomedical Materials Research Part A",
        "year": 2023,
        "doi": "10.1002/jbm.a.37645",
        "name_part1": "HAp",
        "name_fluid1": "water",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": 1.0,
        "name_disp1": "Dynol 604",
        "wf_bind_1": 2.0,
        "name_binder1": "PVA+PEG",
        "temp_cold": 0 + 273.15,  # 0°C to K
        "time_sinter_1": 3.0,
        "temp_sinter_1": 1250.0,
        "vf_total": 0.10,
        "porosity": 81.20
    },
    {
        "title": "Freeze casting of hydroxyapatite-titania composites for bone substitutes",
        "journal": "Journal of Biomedical Materials Research Part A",
        "year": 2023,
        "doi": "10.1002/jbm.a.37645",
        "name_part1": "HAp/TiO₂",
        "name_fluid1": "water",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": 1.0,
        "name_disp1": "Dynol 604",
        "wf_bind_1": 2.0,
        "name_binder1": "PVA+PEG",
        "temp_cold": 0 + 273.15,
        "time_sinter_1": 3.0,
        "temp_sinter_1": 1250.0,
        "vf_total": 0.10,
        "porosity": 76.50
    },
    {
        "title": "Freeze casting of hydroxyapatite-titania composites for bone substitutes",
        "journal": "Journal of Biomedical Materials Research Part A",
        "year": 2023,
        "doi": "10.1002/jbm.a.37645",
        "name_part1": "HAp/TiO₂",
        "name_fluid1": "water",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": 1.0,
        "name_disp1": "Dynol 604",
        "wf_bind_1": 2.0,
        "name_binder1": "PVA+PEG",
        "temp_cold": 0 + 273.15,
        "time_sinter_1": 3.0,
        "temp_sinter_1": 1250.0,
        "vf_total": 0.10,
        "porosity": 73.80
    },
    {
        "title": "Freeze casting of hydroxyapatite-titania composites for bone substitutes",
        "journal": "Journal of Biomedical Materials Research Part A",
        "year": 2023,
        "doi": "10.1002/jbm.a.37645",
        "name_part1": "HAp/TiO₂",
        "name_fluid1": "water",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": 1.0,
        "name_disp1": "Dynol 604",
        "wf_bind_1": 2.0,
        "name_binder1": "PVA+PEG",
        "temp_cold": 0 + 273.15,
        "time_sinter_1": 3.0,
        "temp_sinter_1": 1250.0,
        "vf_total": 0.10,
        "porosity": 68.40
    },
    {
        "title": "Freeze-casting uniformity and domains",
        "journal": "Materials Science & Engineering C",
        "year": 2024,
        "doi": "10.1016/j.msec.2024.114567",
        "name_part1": "LCSM",
        "name_fluid1": "water",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": np.nan,
        "name_disp1": np.nan,
        "wf_bind_1": np.nan,
        "name_binder1": np.nan,
        "temp_cold": -10 + 273.15,  # -10°C to K
        "time_sinter_1": np.nan,
        "temp_sinter_1": np.nan,
        "vf_total": np.nan,
        "porosity": 55.00
    },
    {
        "title": "Biomimetic Materials by Freeze Casting (Update)",
        "journal": "Journal of Materials Research",
        "year": 2021,
        "doi": "10.1557/jmr.2021.82",
        "name_part1": "ZrO₂",
        "name_fluid1": "water",
        "material_group": "Ceramic",
        "dia_part_1": 0.50,
        "wf_disp_1": 1.0,
        "name_disp1": np.nan,
        "wf_bind_1": 1.5,
        "name_binder1": np.nan,
        "temp_cold": 0 + 273.15,
        "time_sinter_1": 2.0,
        "temp_sinter_1": 1400.0,
        "vf_total": 0.20,
        "porosity": 75.00
    },
    # New HAP-BNNT composite (Sci. Direct 2023)
    {
        "title": "Freeze casting to engineer gradient porosity in hydroxyapatite-boron nitride nanotube composite scaffold",
        "journal": "Ceramics International",
        "year": 2023,
        "doi": "10.1016/j.ceramint.2023.05.123",
        "name_part1": "HAp/BNNT",
        "name_fluid1": "camphene",
        "material_group": "Ceramic Composite",
        "dia_part_1": 0.01,  # BNNT diameter
        "wf_disp_1": 0.5,
        "name_disp1": "NaDDBS",
        "wf_bind_1": np.nan,
        "name_binder1": np.nan,
        "temp_cold": -20 + 273.15,  # Estimated from process
        "time_sinter_1": 3.0,
        "temp_sinter_1": 1250.0,
        "vf_total": 0.30,  # 30 vol% solids
        "porosity": 72.0  # Avg. from structural analysis :cite[1]
    },

    # Al₂O₃ scaffolds with lamellar pores (Sci. Direct 2025)
    {
        "title": "Effect of pore architecture on quasistatic compressive deformation of freeze-cast porous alumina scaffolds",
        "journal": "Journal of the European Ceramic Society",
        "year": 2025,
        "doi": "10.1016/j.jeurceramsoc.2025.01.045",
        "name_part1": "Al₂O₃ (BP)",
        "name_fluid1": "water",
        "material_group": "Ceramic",
        "dia_part_1": 8.1,  # Bigger platelets
        "wf_disp_1": 1.0,
        "name_disp1": "PAA-NH₄",
        "wf_bind_1": np.nan,
        "name_binder1": np.nan,
        "temp_cold": -196 + 273.15,  # Liquid N₂ cooling
        "time_sinter_1": 2.0,
        "temp_sinter_1": 1500.0,
        "vf_total": 0.15,  # 15 vol%
        "porosity": 86.0  # >85% porosity :cite[2]
    },

    # Gradient HAP scaffold (Materials 2017)
    {
        "title": "The Production of Porous Hydroxyapatite Scaffolds with Graded Porosity by Sequential Freeze-Casting",
        "journal": "Materials",
        "year": 2017,
        "doi": "10.3390/ma10040367",
        "name_part1": "HAp",
        "name_fluid1": "camphene",
        "material_group": "Ceramic",
        "dia_part_1": np.nan,
        "wf_disp_1": 0.5,
        "name_disp1": "Hypermer KD-4",
        "wf_bind_1": np.nan,
        "name_binder1": np.nan,
        "temp_cold": 42 + 273.15,  # Camphene solidification temp
        "time_sinter_1": 2.0,
        "temp_sinter_1": 1250.0,
        "vf_total": 0.25,  # 25 vol%
        "porosity": 65.0  # Mid-range value :cite[8]
    },

    # Magnetic field-assisted Al₂O₃ (J. Mater. Res. 2020)
    {
        "title": "Design of porous aluminum oxide ceramics using magnetic field-assisted freeze-casting",
        "journal": "Journal of Materials Research",
        "year": 2020,
        "doi": "10.1557/jmr.2020.150",
        "name_part1": "Al₂O₃/Fe₃O₄",
        "name_fluid1": "water",
        "material_group": "Ceramic Composite",
        "dia_part_1": 0.4,  # Typical alumina particle size
        "wf_disp_1": 1.5,
        "name_disp1": "PVA",
        "wf_bind_1": 3.0,
        "name_binder1": "PVA",
        "temp_cold": -20 + 273.15,
        "time_sinter_1": 3.0,
        "temp_sinter_1": 1500.0,
        "vf_total": 0.20,
        "porosity": 68.0  # Controlled unidirectional pores :cite[3]
    },

    # HAP-TiO₂ composites (J Biomed Mater Res A 2023)
    {
        "title": "Freeze casting of hydroxyapatite-titania composites for bone substitutes",
        "journal": "Journal of Biomedical Materials Research Part A",
        "year": 2023,
        "doi": "10.1002/jbm.a.37645",
        "name_part1": "HAp/TiO₂",
        "name_fluid1": "water",
        "material_group": "Ceramic Composite",
        "dia_part_1": np.nan,
        "wf_disp_1": 1.0,
        "name_disp1": "Dynol 604",
        "wf_bind_1": 2.0,
        "name_binder1": "PVA+PEG",
        "temp_cold": 0 + 273.15,
        "time_sinter_1": 3.0,
        "temp_sinter_1": 1250.0,
        "vf_total": 0.10,
        "porosity": 76.50  # HT-50 composition :cite[10]
    }

]

# Create DataFrame
df_dseek = pd.DataFrame(dseek_data)
df_gpt = pd.DataFrame(gpt_data)
# Concatenate DataFrames
df_additional_data = pd.concat([df_dseek, df_gpt], ignore_index=True)

# Remove duplicates based on 'doi' and 'porosity'
df_combined = df_additional_data.drop_duplicates(subset=["doi", "porosity"], keep="last")