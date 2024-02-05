# Python
import numpy as np
import pandas as pd
import pydeck as pdk
import seaborn as sns
import streamlit as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from scipy import stats
from statsmodels.formula.api import ols

# Set page configuration
st.set_page_config(page_title="EuroPet Sales", page_icon="🛒", layout="wide")

# Assuming the Excel file is named 'data.xlsx' and is in the same directory as this script.
df: pd.DataFrame = pd.read_excel("EuroPet.xls")

# Title for the app
st.title("Fueling Sales at EuroPet ⛽️ 🛒")

# Sidebar controls
option: str = st.sidebar.radio(
    "Flow of the App 🔄",
    (
        "1️⃣ | Explore Marseille 🗺️",
        "2️⃣ | Peek into the Data 👀",
        "3️⃣ | Dive into the Numbers 📊",
        "4️⃣ | Predict the Future Pt. 1 🎢",
        "5️⃣ | Predict the Future Pt. 2 🎢",
        "6️⃣ | Predict the Future Pt. 3 🎢",
    ),
)

# CONSTANTS
# Coordinates for Marseille
MARSEILLE_COORDS: tuple = (43.2965, 5.3698)

# Set the style for matplotlib plots
plt.style.use("rose-pine-dawn")

# Set the font for matplotlib plots
plt.rcParams["font.family"] = "Gill Sans"

# Set the resolution for matplotlib plots
plt.rcParams["figure.dpi"] = 300

# Define the colors for the plots
PLOT_COLORS: list = ["#c4a7e7", "#9ccfd8", "#ea9a97", "#f6c177"]
PLOT_COLORS2: list = ["#b4637a", "#ea9d34", "#286983", "#907aa9"]


# FUNCTIONS
def create_and_display_plot(column_name, plot_type):
    # cycle through the colors from plot_colors
    color = PLOT_COLORS[df.columns.get_loc(column_name) % len(PLOT_COLORS)]

    if column_name not in df:
        st.error(
            "Oops! It seems like the selected column is missing. 🙈 Please double-check your DataFrame and try again."
        )
        return

    # Create figure
    plt.figure(figsize=(14, 6))
    ax = plt.gca()

    # Plot histogram or boxplot based on user's selection
    data = df[column_name]
    # Calculate statistics
    min_value = data.min()
    max_value = data.max()
    mean_value = data.mean()
    std_dev = data.std()
    if plot_type == "Thrilling Histogram":
        n, bins, patches = plt.hist(data, bins=10, color=color, alpha=0.7)

        # Overlay line chart for normal distribution
        num_std_dev = 3  # Number of standard deviations to include on either side of the mean
        x_min_gauss = mean_value - num_std_dev * std_dev
        x_max_gauss = mean_value + num_std_dev * std_dev
        x_gauss = np.linspace(x_min_gauss, x_max_gauss, 300)
        y_gauss = stats.norm.pdf(x_gauss, mean_value, std_dev)

        # Adjust the scaling of the Gaussian curve to fit the histogram
        scaling_factor = max(n) / max(y_gauss)
        ax.plot(x_gauss, y_gauss * scaling_factor, color=color)

        # Set limits and title
        x_pad = (max_value - min_value) * 0.2  # 20% padding
        plt.xlim(min_value - x_pad, max_value + x_pad)

        # Draw lines and annotations for min, max, and mean
        plt.axvline(min_value, color="#b4637a", linestyle="-", lw=2)
        plt.axvline(max_value, color="#b4637a", linestyle="-", lw=2)
        plt.axvline(mean_value, color="#286983", linestyle="-", lw=2)

        # Annotations
        ax.annotate(
            f"Min: {min_value:.2f}",
            (min_value, 5),
            textcoords="offset points",
            xytext=(-15, 60),
            ha="right",
            size=11,
        )
        ax.annotate(
            f"Max: {max_value:.2f}",
            (max_value, 5),
            textcoords="offset points",
            xytext=(60, 60),
            ha="right",
            size=11,
        )
        ax.annotate(
            f"Mean: {mean_value:.2f}",
            (mean_value, 5),
            textcoords="offset points",
            xytext=(-8, -10),
            ha="right",
            size=11,
        )

        # Set labels and grid
        ax.set_xlabel(column_name, fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)
        ax.grid(True)
        plt.title(f"{column_name} Histogram", fontsize=16, color=color, pad=20)
    elif plot_type == "Adventurous Boxplot":
        plt.boxplot(
            data,
            vert=False,  # Set boxplot to horizontal
            patch_artist=True,
            boxprops=dict(facecolor=color, color="#56526e"),
            capprops=dict(color="#56526e"),
            whiskerprops=dict(color="#56526e"),
            flierprops=dict(color="#56526e", markeredgecolor=color),
            medianprops=dict(color="#908caa"),
        )

        # Set labels and grid
        ax.set_xlabel("Value", fontsize=14)  # Swap x and y labels
        ax.set_ylabel(column_name, fontsize=14)
        ax.grid(True)
        plt.title(f"{column_name} Boxplot", fontsize=21, color=color, pad=20)

        # Draw cross markers for min, max, and mean
        plt.scatter(
            min_value, 1, marker="x", color="#b4637a", s=100, zorder=10
        )
        plt.scatter(
            max_value, 1, marker="x", color="#b4637a", s=100, zorder=10
        )
        plt.scatter(
            mean_value, 1, marker="x", color="#286983", s=100, zorder=10
        )

        # Annotations
        ax.annotate(
            f"Min: {min_value:.2f}",
            (min_value, 1),
            textcoords="offset points",
            xytext=(18, -22),
            ha="right",
            size=11,
        )
        ax.annotate(
            f"Max: {max_value:.2f}",
            (max_value, 1),
            textcoords="offset points",
            xytext=(-20, -22),
            ha="left",
            size=11,
        )
        ax.annotate(
            f"Mean: {mean_value:.2f}",
            (mean_value, 1),
            textcoords="offset points",
            xytext=(18, -22),
            ha="right",
            size=11,
        )

    # Display the plot
    st.pyplot(plt)
    st.caption(
        "📊 The histogram and boxplot provide a visual summary of the data distribution and its central tendency. The histogram shows the frequency of different values, while the boxplot displays the median, quartiles, and potential outliers."
    )


def create_and_display_regression_plot(column_name):
    color = PLOT_COLORS2[df.columns.get_loc(column_name) % len(PLOT_COLORS2)]

    # Define the model
    formula = f"Sales ~ Q('{column_name}')"
    model = ols(formula, data=df).fit()

    # Create figure for matplotlib
    fig, ax = plt.subplots(figsize=(14, 6))
    plt.title(
        f"Sales Estimation Based on {column_name}",
        fontsize=21,
        color=color,
        pad=20,
    )

    # Create the regression plot
    sns.regplot(
        x=column_name,
        y="Sales",
        data=df,
        ax=ax,
        scatter_kws={"color": color},
        line_kws={"color": "#9893a5"},
    )
    ax.set_xlabel(column_name, fontsize=14)
    ax.set_ylabel("Sales", fontsize=14)

    if show_labels or show_interval:
        # Assuming functions to calculate smallest, average, and largest values are defined
        smallest_value = df[column_name].min()
        average_value = df[column_name].mean()
        largest_value = df[column_name].max()

        # Function to get the 95% prediction interval
        def get_prediction_interval(value):
            new_data = pd.DataFrame({column_name: [value]})
            prediction = model.get_prediction(new_data)
            return prediction.conf_int(alpha=0.05)  # 95% interval

        # Plotting points and intervals
        for value, label in zip(
            [smallest_value, average_value, largest_value],
            ["Smallest", "Average", "Largest"],
        ):
            estimated_sales = model.predict({column_name: value}).iloc[0]
            interval = get_prediction_interval(value)

            # Plot the point
            ax.scatter(value, estimated_sales, color="#56526e", zorder=3)
            if show_labels:
                ax.text(
                    value + (value * 0.003),
                    estimated_sales - (estimated_sales * 0.01),
                    f"{label}: {estimated_sales:.2f}",
                    color="#56526e",
                    fontsize=11,
                    bbox=dict(
                        boxstyle="round",
                        facecolor="white",
                        edgecolor="white",
                        alpha=0.4,
                    ),
                )

            # Plot the interval
            if show_interval:
                interval_df = pd.DataFrame(
                    interval, columns=["lower", "upper"]
                )
                ax.plot(
                    [value, value],
                    [interval_df.iloc[0, 0], interval_df.iloc[0, 1]],
                    color="#21202e",
                )

    if show_interval:
        interval_line = mlines.Line2D(
            [], [], color="#21202e", markersize=15, label="95th Percentile"
        )
    reg_line = mlines.Line2D(
        [], [], color="#9893a5", markersize=15, label="Regression Line"
    )
    handles = [reg_line]
    if show_interval:
        handles.append(interval_line)
    ax.legend(handles=handles, loc="upper left")

    # Display the plot
    st.pyplot(fig)
    st.caption(
        "📈 The regression plot shows the relationship between the selected variable and c-store sales. The line represents the estimated relationship, while the points and intervals provide a visual of the sales estimate and its 95% prediction interval."
    )

    if show_model:
        # Display the model summary
        st.write(model.summary())
        st.caption(
            "📊 The model summary is generated by the OLS Regression model. It provides detailed information about the model's performance, including the R-squared value, coefficients, and p-values."
        )


def multi_variable_regression_and_plots(selected_variables):
    if len(selected_variables) < 2:
        st.error(
            "Oops! 🙊 It seems like you need to select at least two variables to unleash the power of regression analysis. Let's choose some more variables for a thrilling data adventure! 🚀"
        )
    else:
        color1 = PLOT_COLORS[
            df.columns.get_loc(selected_variables[0]) % len(PLOT_COLORS)
        ]
        color2 = PLOT_COLORS[
            (df.columns.get_loc(selected_variables[0]) + 1) % len(PLOT_COLORS)
        ]
        color3 = PLOT_COLORS[
            (df.columns.get_loc(selected_variables[0]) + 2) % len(PLOT_COLORS)
        ]
        # Define the model formula based on selected variables
        formula = "Sales ~ " + " + ".join(
            [f"Q('{var}')" for var in selected_variables]
        )

        # Fit the model
        model = ols(formula, data=df).fit()

        # Create subplots
        fig, axs = plt.subplots(1, 3, figsize=(14, 6))

        # Coefficients plot
        coeffs = model.params.drop("Intercept")
        axs[0].bar(coeffs.index, coeffs.values, color=color1)
        axs[0].set_title("Coefficients", fontsize=20, color=color1, pad=10)
        axs[0].tick_params(axis="x", rotation=45)
        axs[0].set_ylabel("Coefficient Value", fontsize=14)

        # P-values plot
        p_values = model.pvalues.drop("Intercept")
        axs[1].bar(p_values.index, p_values.values, color=color2)
        axs[1].set_title("P-values", fontsize=20, color=color2, pad=10)
        axs[1].tick_params(axis="x", rotation=45)
        axs[1].set_ylabel("P-value", fontsize=14)
        axs[1].axhline(y=0.05, color="#DBB3BE", linestyle="--")
        axs[1].axhline(y=0.1, color="#b4637a", linestyle="--")

        # R-squared plot
        r_squared = model.rsquared
        axs[2].bar(["R-squared"], [r_squared], color=color3)
        axs[2].set_title("R-squared", fontsize=20, color=color3, pad=10)
        axs[2].set_ylabel("R-squared Value", fontsize=14)
        axs[2].axhline(y=0.85, color="#286983", linestyle="--")

        # Adjust layout and style
        plt.style.use("rose-pine-dawn")
        plt.rcParams["font.family"] = "Gill Sans"
        plt.rcParams["figure.dpi"] = 300
        fig.tight_layout(pad=3.0)

        # Add legend for p-value chart
        legend_lines = [
            mlines.Line2D(
                [],
                [],
                color="#b4637a",
                linestyle="--",
                label="10% Significance Level (Tyler's Rule)",
            ),
            mlines.Line2D(
                [],
                [],
                color="#DBB3BE",
                linestyle="--",
                label="5% Significance Level (Traditional)",
            ),
        ]
        axs[1].legend(handles=legend_lines, loc="upper left")

        # Display the plots
        st.pyplot(fig)
        st.caption(
            "📊 The multiplot above shows the bar charts of the results of the regression: Coefficients, P-values, and R-squared. The coefficients plot displays the estimated effect of each variable on c-store sales. The p-values plot shows the significance of each variable, and the R-squared plot indicates the model's goodness of fit."
        )
        if enable_calc:
            # Calculate the estimate for c-store sales with given variable values
            st.subheader("🔮 Sales Estimation 🔮")
            selected_fields = st.multiselect(
                "Choose some exciting fields to boost your data!", df.columns.drop("Sales")
            )
            received_values = {}
            for field in selected_fields:
                value = st.number_input(
                    f"Enter value for {field}", value=df[field].mean()
                )
                received_values[field] = [value]
            c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
            with c1:
                calculate_button = st.button(
                    "Unleash the Power of Regression Analysis 🚀"
                )
            with c2:
                get_interval = st.checkbox("Get 95% Interval 📏")
            with c3:
                get_net_impact = st.checkbox("Calculate Net Impact❗️")
            with c4:
                factor_enabled = st.checkbox("Activate Factor Boost! 🚀")
            with c2:
                factor_field = None
                if factor_enabled:
                    factor_field = st.selectbox(
                        "Select the field to supercharge! 🚀",
                        df.columns.drop("Sales"),
                    )
            with c3:
                if factor_enabled:
                    factor = st.number_input("Enter the Power Boost Factor 🚀", value=1)
            if calculate_button:
                # Calculate the estimate
                temp = []
                for field in selected_fields:
                    if field == factor_field and factor_enabled:
                        temp.append(
                            model.params[f"Q('{field}')"]
                            * received_values[field][0]
                            * factor
                        )
                    else:
                        temp.append(
                            model.params[f"Q('{field}')"]
                            * received_values[field][0]
                        )
                estimated_sales = model.params["Intercept"] + sum(temp)
                st.write(
                    "To estimate c-store sales, we will use the regression model coefficients obtained from the analysis. The equation derived from our model is:"
                )

                equation = f"Sales = \\beta_0 + \\beta_1 * column_1 + \\beta_2 * column_2 + ... + \\beta_n * column_n"
                if factor_enabled:
                    equation = equation.replace(
                        f"\\beta_1 * column_1",
                        f"\\beta_1 * {factor_field} * {factor}",
                    )
                st.latex(equation)

                st.write("Where:")
                st.write(
                    f"β_0 = is the intercept of the model (constant term)"
                )
                st.write(
                    f"β_1, β_2, ... β_n = are the coefficients of the variables in the model"
                )
                st.success(f"Estimated Sales: {estimated_sales:.2f} €")
                if get_interval:
                    predict_df = pd.DataFrame(received_values)
                    predict_df_with_const = sm.add_constant(predict_df)
                    prediction = model.get_prediction(predict_df_with_const)
                    interval = prediction.conf_int()
                    ci_lower, ci_upper = interval[0]
                    st.warning(
                        f"95% Prediction Interval ranges from {ci_lower:.2f} to {ci_upper:.2f} €"
                    )
                    if get_net_impact:
                        st.subheader("📉 Net Impact of Advertising 📈")
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            tv_grp_cost = st.number_input(
                                "Cost of TV Advertising (€)/GRP",
                                value=300,
                                disabled=True,
                            )
                        with c2:
                            radio_grp_cost = st.number_input(
                                "Cost of Radio Advertising (€)/GRP",
                                value=25,
                                disabled=True,
                            )
                        with c3:
                            profit_margin = st.number_input(
                                "Profit Margin on Advertising (%)",
                                value=0.30,
                                disabled=True,
                            )
                        with c4:
                            mf = st.number_input(
                                "Long term effect of Advertising (mf)",
                                value=3,
                                disabled=True,
                            )

                        tv_grps = received_values["TV"][0]
                        tv_advertising_cost = tv_grps * tv_grp_cost

                        if "Radio" in received_values:
                            radio_grps = received_values["Radio"][0]
                            radio_advertising_cost = (
                                radio_grps * radio_grp_cost
                            )
                            advertising_cost = (
                                tv_advertising_cost + radio_advertising_cost
                            )
                        else:
                            advertising_cost = tv_advertising_cost
                        # Calculate the overall revenue impact including the long-term effect
                        overall_revenue_impact = estimated_sales * mf

                        # Calculate the overall profit generated by c-store advertising
                        overall_profit = overall_revenue_impact * profit_margin

                        # Calculate the net impact of c-store advertising
                        net_impact = overall_profit - advertising_cost

                        st.write(
                            "The net impact of advertising is calculated as the difference between the overall profit generated by c-store advertising and the cost of advertising."
                        )
                        st.latex(
                            "\\text{Advertising Cost} = \\text{TV GRPs} * \\text{Cost of TV Advertising} + \\text{Radio GRPs} * \\text{Cost of Radio Advertising}"
                        )
                        st.latex(
                            "\\text{Overall Revenue Impact} = \\text{Estimated Sales} * \\text{Long term effect of Advertising}"
                        )
                        st.latex(
                            "\\text{Overall Profit} = \\text{Overall Revenue Impact} * \\text{Profit Margin on Advertising}"
                        )
                        st.latex(
                            "\\text{Net Impact} = \\text{Overall Profit} - \\text{Advertising Cost}"
                        )

                        st.info(f"Advertising Cost: {advertising_cost:.2f} €")
                        st.info(
                            f"Overall Revenue Impact: {overall_revenue_impact:.2f} €"
                        )
                        st.info(f"Overall Profit: {overall_profit:.2f} €")
                        st.info(f"Net Impact: {net_impact:.2f} €")

        if unleash_tech_summary:
            # Display model summary
            st.write(model.summary())
            st.caption(
                "📊 The model summary is generated by the OLS Regression model. It provides detailed information about the model's performance, including the R-squared value, coefficients, and p-values."
            )


##########################


######## MAIN LOGIC ########
if option == "1️⃣ | Explore Marseille 🗺️":
    st.subheader("🗺️ Exploring Marseille: Unleashing the Adventure! 🏞️")
    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=MARSEILLE_COORDS[0],
                longitude=MARSEILLE_COORDS[1],
                zoom=5,
                pitch=40,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=[{"position": MARSEILLE_COORDS, "size": 100}],
                    get_position="position",
                    get_color="[180, 0, 200, 140]",
                    get_radius="size",
                ),
            ],
        )
    )
    st.caption("Marseille, France")

    st.subheader("🌟 Fun Facts about Marseille! Let's Dive In!")
    facts = [
        "Marseille is the second-largest city in France, after Paris.",
        "It is the oldest city in France, founded by the Greeks around 600 BC.",
        "Marseille is known for its vibrant multicultural atmosphere and diverse cuisine.",
        "The city has a rich maritime history and is home to the largest port in France.",
        "Marseille is famous for its beautiful coastline and stunning calanques (narrow inlets).",
        "The local dish of Marseille is bouillabaisse, a traditional fish stew.",
        "The iconic Notre-Dame de la Garde basilica offers panoramic views of the city.",
        "Marseille was named the European Capital of Culture in 2013.",
        "The city has been featured in numerous films, including 'The French Connection' and 'Marius'.",
        "Marseille is a popular destination for outdoor activities like hiking, sailing, and diving.",
    ]
    for fact in facts:
        st.write(f"- " + fact)
elif option == "2️⃣ | Peek into the Data 👀":
    st.subheader("🔍 Exploring the Data: Unveiling the Secrets! 🕵️‍♂️")
    # Display DataFrame head
    st.subheader("How the Data Looks Like?")
    st.dataframe(df.head())  # Using dataframe to ensure proper formatting

    # Display DataFrame's descriptive statistics
    st.subheader("What Are the Data's Properties?")
    st.write(df.describe())

    # Column explanations
    st.subheader("What Do the Columns Mean?")
    column_explanations = [
        (
            "Week",
            "Running count of the week of the year. 1 = first week in January, and so on up to 52 = last week in December.",
        ),
        (
            "Sales",
            "The average convenience store sales (in €) per week per store location.",
        ),
        (
            "TV",
            "Total TV Gross Rating Points (GRPs), measuring the volume of message delivery to the target audience. One GRP represents 1% of the target audience reached.",
        ),
        (
            "Radio",
            "Total radio GRPs, with the average price of a radio GRP in Marseille being €25.",
        ),
        (
            "Fuel Volume",
            "Average volume of fuel sold per EuroPet facility in Marseille per week (in liters).",
        ),
        (
            "Fuel Price",
            "Average price of fuel in the market (in cents per liter).",
        ),
        (
            "Temp",
            "Average high temperature recorded in a given week in Marseille (in °C).",
        ),
        ("Prec", "Precipitation or rainfall (in millimeters)."),
        (
            "Holiday",
            "Dummy variable indicating whether there was a public or school holiday in a week (1) or not (0).",
        ),
        (
            "Visits (1 or 2)",
            "Percentage of survey respondents who reported shopping at a EuroPet convenience store 1–2 times in the past week.",
        ),
    ]

    st.table(
        pd.DataFrame(
            column_explanations, columns=["Column Name", "Description"]
        ).set_index("Column Name")
    )
elif option == "3️⃣ | Dive into the Numbers 📊":
    st.subheader("📊 Data Analysis: Riding the Data Wave! 🌊")
    col1, col2 = st.columns(2)

    with col1:
        selected_column = st.selectbox(
            "Choose a column to spice up 🌶️ your analysis with sales 📈",
            df.columns,
        )

    with col2:
        plot_type = st.selectbox(
            "Choose an exciting plot type 📊",
            ["Thrilling Histogram", "Adventurous Boxplot"],
        )
    create_and_display_plot(selected_column, plot_type)
elif option == "4️⃣ | Predict the Future Pt. 1 🎢":
    st.subheader(
        "🔮 Predicting the Future: Single Variable Sales Forecasting 📈"
    )
    selected_variable = st.selectbox(
        "Choose an exciting variable for regression analysis 🔴",
        df.columns.drop(["Sales"]),
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        show_labels = st.checkbox("Show Sales Numbers 🧮")
    with col2:
        show_interval = st.checkbox("Show Interval Widths 📏")
    with col3:
        show_model = st.checkbox("Unleash the Technical Summary 💣")

    create_and_display_regression_plot(selected_variable)
elif option == "5️⃣ | Predict the Future Pt. 2 🎢":
    st.subheader(
        "🔮 Predicting the Future: Multi-Variable Sales Forecasting 📈"
    )
    selected_variables = st.multiselect(
        "Choose some exciting variables for regression analysis!",
        df.columns.drop(["Sales"]),
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        enable_calc = st.checkbox("Enable Estimation 🔮")
    with c3:
        unleash_tech_summary = st.checkbox("Unleash Tech Summary 💣")
    multi_variable_regression_and_plots(selected_variables)
elif option == "6️⃣ | Predict the Future Pt. 3 🎢":
    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader(
            "🔮 Predicting the Future: Multi-Variable Sales Forecasting 📈"
        )
    with c2:
        week_magic = st.checkbox("Enable Magical Weeks 🪄")
    if week_magic:
        with c2:
            holiday_magic = st.checkbox("Enable Magical Holidays 🪄")
        df["Magic W07"] = (df["Week"] == 7).astype(int)
        df["Magic W21"] = (df["Week"] == 21).astype(int)
        df["Magic W49"] = (df["Week"] == 49).astype(int)
        if holiday_magic:
            with c2:
                temp_magic = st.checkbox("Enable Magical Temperatures 🪄")
            df["TV_Holiday_Magic"] = df["TV"] * df["Holiday"]
            df["Radio_Holiday_Magic"] = df["Radio"] * df["Holiday"]
            if temp_magic:
                df["Temp_Magic"] = df["Temp"] * df["Holiday"]

    selected_variables = st.multiselect(
        "Choose some exciting variables for regression analysis!",
        df.columns.drop(["Sales"]),
    )
    c1, c2 = st.columns([3, 1])
    with c1:
        enable_calc = st.checkbox("Enable Estimation 🔮")
    with c2:
        unleash_tech_summary = st.checkbox("Unleash Tech Summary 💣")
    multi_variable_regression_and_plots(selected_variables)
