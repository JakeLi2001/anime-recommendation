# Animendation

## Introduction

**Goal**: To build an anime recommendation engine and host on Streamlit.

**Motivation**: I love anime.

<img src='https://tenor.com/view/spy-x-family-anya-sparkling-eyes-anime-gif-25175073.gif' width=400 height=250 />

By valcrist via [Tenor](https://tenor.com/view/spy-x-family-anya-sparkling-eyes-anime-gif-25175073)

## Table of Contents

- [Quick Summary](#open_book-quick-summary)
- [Data Source](#mag_right-data-source-myanimelist-api)
- [Result](#dart-result)
- [Tech Stack](#hammer_and_wrench-tech-stack)

## :open_book: Quick Summary

1. Get data via the MyAnimeList API and export as CSV.
2. Exploratory data analysis and data cleaning.
3. Text preprocessing.
4. Build model using cosine similarity.

## :mag_right: Data Source

Pulled the top 10K most popular animes from [MyAnimeList API](https://myanimelist.net/apiconfig/references/api/v2#section/Authentication). 

## :dart: Result

Model deployed on Streamlit! ([Web app](https://animendation.streamlit.app/))

## :hammer_and_wrench: Tech Stack

**Language:** Python

**Libraries:** requests, re, Numpy, Pandas, NLTK, Scikit-learn

**Framework:** Streamlit

**Tool:** Jupyter Lab
