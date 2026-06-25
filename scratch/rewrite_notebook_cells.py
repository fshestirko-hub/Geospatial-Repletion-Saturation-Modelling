import json
from pathlib import Path

notebook_path = Path(r"c:\Users\fedka\Desktop\Geospatial Repletion & Saturation Modelling - Copy\notebooks\1_part_master_notebook.ipynb")

if not notebook_path.exists():
    print(f"Error: Notebook not found at {notebook_path}")
    exit(1)

with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Define the exact mapping of cells we want to rewrite based on unique keywords
replacements = {
    "plot_linear_series": [
        "import os\n",
        "from IPython.display import display, HTML\n",
        "\n",
        "# Render linear series if missing from environment\n",
        "if not (PLOTS_DIR / \"linear_series.png\").exists():\n",
        "    plot_linear_series(spark, phone_df, PLOTS_DIR)\n",
        "\n",
        "# Render gravity magnitude trace if missing from environment\n",
        "if not (PLOTS_DIR / \"gravity_magnitude.png\").exists():\n",
        "    plot_gravity_magnitude(spark, phone_df, PLOTS_DIR)\n",
        "\n",
        "# Display inline\n",
        "if (PLOTS_DIR / \"linear_series.png\").exists():\n",
        "    display(HTML('<img src=\"../data/plots/linear_series.png\" width=\"800\" style=\"margin-bottom: 20px;\" />'))\n",
        "if (PLOTS_DIR / \"gravity_magnitude.png\").exists():\n",
        "    display(HTML('<img src=\"../data/plots/gravity_magnitude.png\" width=\"800\" />'))\n"
    ],
    "plot_attractor_geometries": [
        "import os\n",
        "from IPython.display import display, HTML\n",
        "\n",
        "# Render attractor geometries if missing from environment\n",
        "if not (PLOTS_DIR / \"topographic_map.png\").exists():\n",
        "    plot_attractor_geometries(spark, phone_df, PLOTS_DIR)\n",
        "\n",
        "# Render variance comparison if missing from environment\n",
        "if not (PLOTS_DIR / \"phone_vs_watch_comparison.png\").exists():\n",
        "    plot_variance_comparison(spark, phone_df, watch_df, PLOTS_DIR)\n",
        "\n",
        "# Display inline\n",
        "if (PLOTS_DIR / \"topographic_map.png\").exists():\n",
        "    display(HTML('<img src=\"../data/plots/topographic_map.png\" width=\"800\" style=\"margin-bottom: 20px;\" />'))\n",
        "if (PLOTS_DIR / \"phone_vs_watch_comparison.png\").exists():\n",
        "    display(HTML('<img src=\"../data/plots/phone_vs_watch_comparison.png\" width=\"800\" />'))\n"
    ],
    "plot_population_heterogeneity": [
        "import os\n",
        "from IPython.display import display, HTML\n",
        "\n",
        "# Render population heterogeneity if missing from environment\n",
        "if not (PLOTS_DIR / \"population_heterogeneity_violins.png\").exists():\n",
        "    plot_population_heterogeneity(spark, phone_df, PLOTS_DIR)\n",
        "\n",
        "# Render 3D attractor maps if missing from environment\n",
        "if not (PLOTS_DIR / \"multivariate_user_attractors_3d.png\").exists():\n",
        "    plot_multivariate_user_attractors_3d(spark, phone_df, PLOTS_DIR)\n",
        "\n",
        "# Display inline\n",
        "if (PLOTS_DIR / \"population_heterogeneity_violins.png\").exists():\n",
        "    display(HTML('<img src=\"../data/plots/population_heterogeneity_violins.png\" width=\"800\" style=\"margin-bottom: 20px;\" />'))\n",
        "if (PLOTS_DIR / \"multivariate_user_attractors_3d.png\").exists():\n",
        "    display(HTML('<img src=\"../data/plots/multivariate_user_attractors_3d.png\" width=\"800\" />'))\n"
    ],
    "plot_agent_kinematics": [
        "from IPython.display import display, HTML\n",
        "from src.verify_bootstrap import plot_agent_kinematics\n",
        "\n",
        "# Verify kinematics profiles of simulated agent if missing from environment\n",
        "if not (PLOTS_DIR / \"diagnostic_agent_kinematics.png\").exists():\n",
        "    plot_agent_kinematics(spark, synth_df, PLOTS_DIR)\n",
        "\n",
        "# Display inline\n",
        "if (PLOTS_DIR / \"diagnostic_agent_kinematics.png\").exists():\n",
        "    display(HTML('<img src=\"../data/plots/diagnostic_agent_kinematics.png\" width=\"800\" />'))\n"
    ],
    "plot_population_density_comparison": [
        "from IPython.display import display, HTML\n",
        "from src.verify_bootstrap import plot_population_density_comparison\n",
        "\n",
        "# Compare empirical user vs synthetic agent distributions if missing from environment\n",
        "if not (PLOTS_DIR / \"diagnostic_population_violins.png\").exists():\n",
        "    plot_population_density_comparison(spark, phone_df, synth_df, PLOTS_DIR)\n",
        "\n",
        "# Display inline\n",
        "if (PLOTS_DIR / \"diagnostic_population_violins.png\").exists():\n",
        "    display(HTML('<img src=\"../data/plots/diagnostic_population_violins.png\" width=\"800\" />'))\n"
    ],
    "plot_temporal_autocorrelation": [
        "from IPython.display import display, HTML\n",
        "from src.verify_bootstrap import plot_temporal_autocorrelation\n",
        "\n",
        "# Compare stride signatures using temporal autocorrelation (ACF) if missing from environment\n",
        "if not (PLOTS_DIR / \"diagnostic_temporal_acf.png\").exists():\n",
        "    plot_temporal_autocorrelation(spark, phone_df, synth_df, PLOTS_DIR)\n",
        "\n",
        "# Display inline\n",
        "if (PLOTS_DIR / \"diagnostic_temporal_acf.png\").exists():\n",
        "    display(HTML('<img src=\"../data/plots/diagnostic_temporal_acf.png\" width=\"800\" />'))\n"
    ]
}

updated = False
for cell in notebook.get("cells", []):
    if cell.get("cell_type") == "code":
        source_text = "".join(cell.get("source", []))
        for key, new_source in replacements.items():
            if key in source_text:
                print(f"Rewriting cell for {key}")
                cell["source"] = new_source
                updated = True
                break

if updated:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print("Notebook cell contents rewritten successfully.")
else:
    print("No cells found matching the keywords.")
