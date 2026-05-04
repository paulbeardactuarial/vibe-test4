import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shiny import App, ui, render

from mortality import pv_annuity

DISCOUNT_RATE = 0.05
IMPROVEMENT_RATE = 0.01
AGES = np.arange(50, 81)
GENDERS = ["male", "female"]

app_ui = ui.page_fluid(
    ui.h3("Joint Life Annuity Present Values — PMA92C20 / PFA92C20"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_slider(
                "age_diff",
                "Age Difference (Life 2 − Life 1)",
                min=-20, max=20, value=0, step=1,
            ),
        ),
        ui.output_plot("pv_plot", height="620px"),
    ),
)


def server(input, output, session):
    @output
    @render.plot
    def pv_plot():
        age_diff = input.age_diff()

        records = []
        for g1 in GENDERS:
            for g2 in GENDERS:
                pv_fd = pv_annuity(
                    g1, AGES, DISCOUNT_RATE, IMPROVEMENT_RATE,
                    "joint", g2, age_diff, "first_death",
                )
                pv_sd = pv_annuity(
                    g1, AGES, DISCOUNT_RATE, IMPROVEMENT_RATE,
                    "joint", g2, age_diff, "second_death",
                )
                for i, a in enumerate(AGES):
                    records.append({
                        "age": a,
                        "Life 1": g1.capitalize(),
                        "Life 2": g2.capitalize(),
                        "Annuity Type": "First Death",
                        "pv": pv_fd[i],
                    })
                    records.append({
                        "age": a,
                        "Life 1": g1.capitalize(),
                        "Life 2": g2.capitalize(),
                        "Annuity Type": "Second Death",
                        "pv": pv_sd[i],
                    })

        df = pd.DataFrame(records)

        palette = {"First Death": "steelblue", "Second Death": "coral"}

        g = sns.FacetGrid(
            df,
            col="Life 1",
            row="Life 2",
            height=3.2,
            aspect=1.3,
            margin_titles=True,
        )
        g.map_dataframe(
            sns.lineplot,
            x="age", y="pv",
            hue="Annuity Type",
            palette=palette,
        )
        g.set(xlim=(50, 80), ylim=(0, 30))
        g.set_axis_labels("Age (Life 1)", "Present Value")
        g.set_titles(col_template="Life 1: {col_name}", row_template="Life 2: {row_name}")
        g.add_legend(title="Annuity Type")
        g.figure.suptitle(
            f"Joint Life Annuity PV  |  Age Diff = {age_diff:+d}  |  "
            f"Discount Rate = {DISCOUNT_RATE:.0%}  |  Improvement = {IMPROVEMENT_RATE:.0%}",
            y=1.02, fontsize=11,
        )
        g.figure.tight_layout()
        return g.figure


app = App(app_ui, server)
