import matplotlib.pyplot as plt


def plot_price(prediction):

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    ax.bar(
        ["Predicted Price"],
        [prediction]
    )

    ax.set_ylabel(
        "Price (Lakhs)"
    )

    ax.set_title(
        "Estimated House Price"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    return fig


def plot_property_features(
    bhk,
    school,
    hospital
):

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    ax.bar(
        ["BHK", "Schools", "Hospitals"],
        [bhk, school, hospital]
    )

    ax.set_title(
        "Nearby Facilities"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    return fig


def plot_price_comparison(
    prediction,
    market_price
):

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    ax.bar(
        ["Predicted", "Market"],
        [prediction, market_price]
    )

    ax.set_ylabel(
        "Price (Lakhs)"
    )

    ax.set_title(
        "Predicted vs Market Price"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    return fig


def plot_feature_importance(
    importance_df
):

    top = importance_df.head(10)

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.barh(
        top["Feature"],
        top["Importance"]
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Importance"
    )

    ax.set_title(
        "Top 10 Important Features"
    )

    return fig


def plot_investment_breakdown(
    prediction
):

    values = [
        prediction * 0.65,
        prediction * 0.35
    ]

    labels = [
        "Building Value",
        "Land Value"
    ]

    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title(
        "Estimated Investment Breakdown"
    )

    return fig


def plot_house_score(score):

    fig, ax = plt.subplots(
        figsize=(7, 1.8)
    )

    ax.barh(
        ["House Score"],
        [score]
    )

    ax.set_xlim(
        0,
        100
    )

    ax.set_xlabel(
        "Score"
    )

    return fig


def plot_history(df):

    if df.empty:
        return None

    if "Prediction" not in df.columns:
        return None

    recent = df.tail(10)

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.plot(
        range(1, len(recent) + 1),
        recent["Prediction"],
        marker="o"
    )

    ax.set_xlabel(
        "Prediction Number"
    )

    ax.set_ylabel(
        "Price (Lakhs)"
    )

    ax.set_title(
        "Recent Prediction Trend"
    )

    ax.grid(True, alpha=0.3)

    return fig