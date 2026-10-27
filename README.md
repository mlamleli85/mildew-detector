# Mildew Detector in Cherry Leaves

**Live dashboard:** _add your Heroku link here_

The Mildew Detector is a machine learning dashboard that tells, from a single photo of a cherry leaf, whether the leaf is **healthy** or has **powdery mildew**. It was built for Farmy & Foods so that farm staff no longer need to inspect every cherry tree by hand.

_Add a screenshot of the dashboard here._

## Table of contents

1. [Dataset Content](#dataset-content)
2. [Business Requirements](#business-requirements)
3. [Hypotheses and how to validate them](#hypotheses-and-how-to-validate-them)
4. [Rationale to map the business requirements to the Data Visualisations and ML tasks](#rationale-to-map-the-business-requirements-to-the-data-visualisations-and-ml-tasks)
5. [ML Business Case](#ml-business-case)
6. [Dashboard Design](#dashboard-design)
7. [CRISP-DM process](#crisp-dm-process)
8. [Project structure](#project-structure)
9. [Bugs](#bugs)
10. [Testing](#testing)
11. [Deployment](#deployment)
12. [Main data analysis and machine learning libraries](#main-data-analysis-and-machine-learning-libraries)
13. [Credits](#credits)

## Dataset Content

- The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). It contains photos of cherry leaves provided by Farmy & Foods, taken from their own crops.
- It holds a little over 4,000 images in two classes of roughly equal size: **healthy** leaves and leaves with **powdery mildew**, a fungal disease that shows as white, powdery patches.
- The images are 256 x 256 pixel RGB photos. For this project they are resized to 128 x 128.
- The cherry crop is one of Farmy & Foods' most valuable products, and the company is concerned about supplying the market with compromised produce.

## Business Requirements

Farmy & Foods currently checks each cherry tree by hand. An employee spends about **30 minutes per tree** taking leaf samples and checking them visually; if mildew is found, a compound is applied in about 1 minute. With thousands of trees on several farms, this process does not scale.

The IT team proposed a machine learning system that detects mildew instantly from a leaf photo. If it succeeds, the same approach could be used for pest detection in other crops.

1. **Business requirement 1:** the client is interested in conducting a study to visually differentiate a healthy cherry leaf from one that contains powdery mildew.
2. **Business requirement 2:** the client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.
3. **Business requirement 3:** deliver a dashboard that meets the above requirements.

## Hypotheses and how to validate them

| # | Hypothesis | Validation | Conclusion |
|---|---|---|---|
| 1 | Leaves with powdery mildew show white, powdery marks that visually set them apart from healthy leaves. | Image montages, average and variability images, and the difference between the average images. | _Fill in after running notebook 02._ |
| 2 | Because the mildew is white, infected leaves have a higher mean brightness than healthy leaves. | Mean pixel brightness per image, compared between the two classes with a Mann-Whitney U test (significance level α = 0.05). | _Fill in with the numbers from `brightness_test.json`._ |
| 3 | A convolutional neural network can classify unseen leaves as healthy or infected with at least 97% accuracy. | Accuracy on the test set, which is never used during training or model selection. | _Fill in with the test accuracy from notebook 03._ |

**Next steps if the hypotheses hold:** roll the detector out to farm staff, keep a manual check for low-confidence predictions, and collect images from other crops to extend the system.

## Rationale to map the business requirements to the Data Visualisations and ML tasks

### Business requirement 1: Data visualisation

- As a client, I want to see the average and variability images for healthy and infected leaves, so that I can visually tell the two apart.
  - _Action:_ compute the mean and standard deviation image per label (notebook 02) and show them on the Leaf Visualiser page.
- As a client, I want to see the difference between an average healthy leaf and an average infected leaf, so that I can see where they differ.
  - _Action:_ plot the two average images and their difference (notebook 02); show on the Leaf Visualiser page.
- As a client, I want to see a montage of real healthy and infected leaves, so that I can recognise the disease myself.
  - _Action:_ create an image montage per label (notebook 02); show on the Leaf Visualiser page with a label selector.

### Business requirement 2: Classification (ML task)

- As a client, I want an ML model that predicts whether a leaf is healthy or has powdery mildew, so that I can decide quickly whether a tree needs treatment.
  - _Action:_ train a binary classification CNN on the labelled images (notebook 03).
- As a client, I want to upload one or more leaf photos and get a prediction with a confidence level, so that I can check trees in the field.
  - _Action:_ build the Mildew Detector page with an image uploader, prediction and probability chart.
- As a client, I want to download a report of the predictions, so that I can keep a record of which trees were checked.
  - _Action:_ show a results table with a CSV download button.

### Business requirement 3: Dashboard

- As a client, I want a dashboard with clear navigation, so that I can find each answer easily.
  - _Action:_ a Streamlit app with a sidebar menu and five pages, deployed on Heroku.

## ML Business Case

### Mildew detection - classification

- **Aim:** predict whether a cherry leaf is healthy or has powdery mildew, so that infected trees can be treated without a 30-minute manual inspection.
- **Learning method:** supervised learning; binary image classification with a convolutional neural network (CNN).
- **Ideal outcome:** farm staff photograph a leaf and get a reliable diagnosis in seconds, so the company can inspect far more trees in the same time.
- **Success metric:** accuracy of **97% or higher** on the test set (agreed with the client).
- **Failure:** the model performs below 97% on the test set, or it frequently labels infected leaves as healthy, which would let the disease spread.
- **Model output:** a label (healthy / powdery mildew) and the probability of that label. A probability of 0.5 or more for powdery mildew means the leaf is classed as infected.
- **Relevance for the user:** staff can treat infected trees straight away and skip healthy ones, saving time and protecting the quality of the crop.
- **Heuristics:** the current heuristic is a manual visual check of leaf samples, which takes about 30 minutes per tree and does not scale.
- **Training data:** the Kaggle cherry leaves dataset, split 70% train, 10% validation and 20% test. Features are the image pixels; the target is the label.

## Dashboard Design

### Page 1: Project Summary

- The business problem and the time cost of the current manual process.
- A description of the dataset with a link to the source.
- The business requirements, and which page answers each one.
- A short guide to the menu.

### Page 2: Leaf Visualiser (business requirement 1)

- Checkbox 1: average and variability images for each label, with interpretation.
- Checkbox 2: the difference between the average healthy and average infected leaf, with interpretation.
- Checkbox 3: image montage with a selector for the label, with interpretation.

### Page 3: Mildew Detector (business requirement 2)

- A link to download sample images from Kaggle.
- An uploader that accepts one or more PNG/JPG images.
- For each image: the image, the diagnosis with its confidence, and an interactive bar chart of the class probabilities.
- A table of all results and a button to download it as a CSV report.

### Page 4: Project Hypotheses

- Each of the three hypotheses, how it was validated and the conclusion.
- Hypothesis 2 shows the brightness statistics, p-value and a box plot.
- Recommended next steps for the client.

### Page 5: ML Performance

- A clear statement at the top on whether the model met the 97% target.
- Interactive bar chart of the number of images per set and label.
- Comparison of the model versions trained.
- Learning curves (accuracy and loss) for training and validation.
- Test set results and confusion matrix.

## CRISP-DM process

1. **Business understanding:** the business requirements and ML business case above.
2. **Data understanding:** notebook 02 studies image size, averages, variability and montages.
3. **Data preparation:** notebook 01 downloads the data, removes non-image files and splits the sets; notebook 03 resizes images and applies augmentation.
4. **Modelling:** notebook 03 trains and compares two CNN versions.
5. **Evaluation:** notebook 03 checks the chosen model against the 97% target on the test set.
6. **Deployment:** the Streamlit dashboard on Heroku.

## Project structure

```
app.py                      # dashboard entry point
app_pages/                  # one file per dashboard page + the menu class
src/                        # helpers: loading outputs, predictions, evaluation
jupyter_notebooks/          # 01 data collection, 02 visualisation, 03 modelling
outputs/v1/                 # saved plots, model and evaluation for version 1
inputs/                     # image data (created by notebook 01, not in Git)
```

## Bugs

_List any bugs you found and how you fixed them. List any known unfixed bugs._

## Testing

### Manual testing

| Feature | Action | Expected result | Result |
|---|---|---|---|
| Navigation | Click each page in the sidebar | The page loads | |
| Leaf Visualiser | Tick each checkbox | Plots and interpretations appear | |
| Montage selector | Switch between labels | Montage changes | |
| Upload one image | Upload a healthy leaf | Prediction "Healthy" with probability chart | |
| Upload several images | Upload 3+ mixed leaves | One result per image and a report table | |
| Wrong file | Upload a non-image renamed to .jpg | Error message, other images still processed | |
| Download report | Click "Download report as CSV" | CSV file downloads | |
| ML Performance | Open the page | Success statement, charts and tables shown | |

### Code validation

All Python files were checked with a PEP 8 linter (for example the [CI Python Linter](https://pep8ci.herokuapp.com/)). _Add results._

## Deployment

### Heroku

- The app is live at: _add link_
- Python version is set in `runtime.txt`. Only the packages the dashboard needs are in `requirements.txt`; notebook-only packages are in `requirements-dev.txt` to keep the Heroku slug small.
- Large or unneeded files are excluded from Heroku with `.slugignore`.

Steps:

1. Log in to Heroku and create a new app.
2. Under the **Deploy** tab, choose **GitHub** and connect this repository.
3. Choose the `main` branch and click **Deploy Branch**.
4. When the build is complete, click **Open App**.

### Running locally

1. Clone the repository and create a virtual environment with Python 3.12.
2. `pip install -r requirements.txt -r requirements-dev.txt`
3. Put your `kaggle.json` in the repository root and run the notebooks in order (01, 02, 03).
4. `streamlit run app.py`

## Main data analysis and machine learning libraries

- **TensorFlow / Keras:** building, training and saving the CNN; loading images from folders; image augmentation.
- **NumPy:** image arrays, average and variability images.
- **pandas:** tables of image counts, training history and results.
- **Matplotlib / Seaborn:** static plots in the notebooks (averages, montages, learning curves, confusion matrix).
- **Plotly:** interactive charts on the dashboard.
- **scikit-learn:** confusion matrix and classification report.
- **SciPy:** Mann-Whitney U test for hypothesis 2.
- **Pillow:** opening and resizing images.
- **Streamlit:** the dashboard.
- **Kaggle API:** downloading the dataset.
- **joblib:** saving and loading small project files.

## Credits

### Content

- The project brief and dataset come from [Code Institute](https://codeinstitute.net/) and [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves).
- The project structure follows the approach taught in the Code Institute Predictive Analytics module.
- Information about powdery mildew: _add the source you used._

### Acknowledgements

- _Thank your mentor and anyone else who helped._
