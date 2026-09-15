
import streamlit as st
import requests


# Page configuration
st.set_page_config(
    page_title="Research Paper Recommender",
    page_icon="📚"
)


# Title
st.title("📚 Research Paper Recommender")

st.write(
    "Search for research papers using semantic similarity."
)


# Search query
query = st.text_input(
    "Enter your research topic:",
    placeholder="e.g. graph neural networks"
)


# Number of recommendations
top_k = st.number_input(
    "Number of recommendations:",
    min_value=1,
    max_value=10,
    value=5
)


# Recommend button
if st.button("Recommend"):

    if not query.strip():

        st.warning("Please enter a research topic.")

    else:

        response = requests.post(
            "http://127.0.0.1:8000/recommend",
            json={
                "query": query,
                "top_k": top_k
            }
        )

        if response.status_code == 200:

            recommendations = response.json()

            # Save recommendations
            st.session_state["recommendations"] = recommendations

        else:

            st.error(
                f"API Error: {response.status_code}"
            )


# Get saved recommendations
recommendations = st.session_state.get(
    "recommendations",
    []
)


# Display recommendations
if recommendations:

    st.subheader("Recommended Papers")

    for paper in recommendations:

        st.markdown(
            f"### {paper['rank']}. {paper['titles']}"
        )

        st.write(
            f"Similarity: "
            f"{paper['similarity_score']:.3f}"
        )

        st.write(
            paper["abstracts"]
        )

        st.divider()


    # Paper selection
    st.subheader("Translate an Abstract")

    selected_rank = st.selectbox(
        "Select a paper:",
        [paper["rank"] for paper in recommendations]
    )


    # Find selected paper
    selected_paper = next(
        paper
        for paper in recommendations
        if paper["rank"] == selected_rank
    )


    # Translate button
    if st.button("Translate Abstract"):

        response = requests.post(
            "http://127.0.0.1:8000/translate",
            json={
                "text": selected_paper["abstracts"]
            }
        )

        if response.status_code == 200:

            translation = response.json()["translation"]

            st.subheader("Arabic Translation")

            st.write(translation)

        else:

            st.error(
                f"Translation API Error: "
                f"{response.status_code}"
            )

