# LLM-steam-review-generator

<p align="center">
  <img height=320 src="https://baldursgate3.game/wallpapers/thumbnails/wallpaper-01-thumb.jpg?raw=true" >
</p>

## Overview
This repository contains the final project for our Data Science course in Winter 2024. The project leverages state-of-the-art language models to generate and classify Steam reviews, specifically for the 2023 video game **"Baldur's Gate 3"**. 

Using GPT-2 from Hugging Face and the LoRA fine-tuning technique, two separate models were created: one for generating new reviews and another for classifying the sentiment of reviews as positive or negative.

The game industry receives a large amount of user-generated content, particularly in the form of reviews, which provides both a wealth of knowledge and a challenge in processing such large datasets. The study provides insight into GPT-2's ability to capture user feelings in game reviews, which has possible implications for natural language generation in the gaming industry. Using existing game reviews as training, the GPT-2 algorithm generates informative reviews that capture user sentiment. A subsequent GPT-2-based classifier classifies the generated reviews as either positive or negative.


## Features
- **Exploratory Data Analysis (EDA) and Visualization:** In-depth analysis and visualization of the dataset, offering insights into the review patterns.
- **Causal-based Review Generation:** A fine-tuned GPT-2 model that generates new Steam reviews for Baldur's Gate 3 based on a provided prompt.
- **Review Sentiment Classification:** A sequence classification model that determines whether a review is positive or negative.
- **Streamlit Web Application:** A user-friendly web interface to generate reviews and classify their sentiment.


## Project Structure

```
├── app.py            # The main application file.
├── model/            # Contains model weights for the fine-tuned GPT-2 models.
├── dataset/          # Preprocessed datasets
├── notebooks/        # Jupyter notebooks for EDA, fine-tuning, and evaluation.
└── report.pdf        # A comprehensive report detailing the project
```

Note: If you plan to run the notebooks, place your Kaggle token inside the `.kaggle` folder. Ensure it is named `kaggle.json`.


## Dataset
The dataset used in this project consists of all English-language Steam reviews for Baldur's Gate 3, which can be found [here](https://www.kaggle.com/datasets/harisyafie/baldurs-gate-3-steam-reviews). The dataset has been preprocessed and is available in the `dataset/` directory. 

Note that the raw dataset is not included due to size limitations, however running the `EDA - Preprocess.ipynb` notebook automatically downloads the full dataset, provided you have placed your Kaggle token.


## Installation
To run this project locally, follow these steps:
1. Clone the repository:
2. Install the required packages:
   ```bash
      pip install -r requirements.txt
    ```
3. Run the Streamlit app:
   ```bash
      streamlit run app.py
    ```

## Usage
Once the Streamlit app is running, you can interact with the following features:
- **Generate Review:** Input a prompt related to Baldur's Gate 3, and the model will generate a corresponding Steam review. Leave the prompt empty to generate a fully random review.
- **Classify Review:** Input a review, and the model will classify it as either positive or negative.


## Model Details
- **Causal-based Model:** A fine-tuned version of GPT-2 designed to generate coherent and contextually appropriate reviews.
- **Sequence Classification Model:** A model trained to predict the sentiment (positive or negative) of a given review.


## Exploratory Data Analysis (EDA)
Detailed analysis and visualizations were performed to understand the distribution of review sentiments, word frequency, and other insights. The EDA results are available in the notebooks provided.


## Results
The project successfully demonstrated the ability to generate and classify reviews for Baldur's Gate 3 using fine-tuned GPT-2 models. Detailed results and model performance metrics can be found in the `report.pdf` file.



## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.




