# Animendation (Anime Recommendation Engine)

**Goal**: To build an anime recommendation engine and host on Streamlit.

**Motivation**: I love anime.

## :label: Table of Contents
- [File/Code Structure](#open_file_folder-filecode-structure)
- [Project Description](#memo-project-description)
- [Data Source](#mag_right-data-source)
- [Summary](#open_book-summary)
- [Result](#dart-result)
- [Tech Stack](#hammer_and_wrench-tech-stack)

---
## :open_file_folder: File/Code Structure

```bash      
├── data                                # raw and cleaned anime data pulled from API
│   ├── anime_data.csv
│   └── cleaned_anime_data.csv
├── streamlit_app                       # contains Python files necessary for the web app
│    ├── gifs                           # contains gifs used for the web app and presentation
│    │   ├── ...
│    ├── homepage.py                    # streamlit app
│    └── requirements.txt               # requirements for the streamlit app
├── .gitignore
├── 1_data_collection.ipynb             # jupyter notebook for the data collection
├── 2_main.ipynb                        # jupyter notebook for building the model
├── README.md          
└── environment.yml                     # requirements for the project
```

[Back to top](#label-table-of-contents)

## :memo: Project Description
The idea is to create a simple anime recommendation engine using cosine similarity based on anime synopsis(plot) and genres. Cosine similarity measures how similar two vectors are to each other on a scale from 0 (opposite) to 1 (identical), and I want to use it to measure how similar animes are. In other words, I am going to measure how similar an anime is to every other anime in the dataset. Once I have the cosine similarity scores, I can use it to provide similar anime recommendations given the title.

[Back to top](#label-table-of-contents)

## :mag_right: Data Source

Pulled the top 10K most popular animes from [MyAnimeList API](https://myanimelist.net/apiconfig/references/api/v2#section/Authentication).

[Back to top](#label-table-of-contents)

## :open_book: Summary

1. Get data via the MyAnimeList API and export it as CSV.
2. Exploratory data analysis and data cleaning.
3. Text preprocessing.
4. Build a recommendation engine using cosine similarity.

[Back to top](#label-table-of-contents)

## :dart: Result

Model deployed on Streamlit! ([Web app](https://animendation.streamlit.app/))

[Back to top](#label-table-of-contents)

## :hammer_and_wrench: Tech Stack

**Language:** Python

**Libraries:** requests, re, Numpy, Pandas, NLTK, Scikit-learn

**Framework:** Streamlit

**Tool:** Jupyter Lab

[Back to top](#label-table-of-contents)
