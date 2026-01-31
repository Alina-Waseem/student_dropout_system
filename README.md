# Student Dropout Early Warning System

## Objective
Predict students at risk of dropping out early in the semester.

## Dataset
xAPI-Edu-Data (Kaggle)

## Data Cleaning
- Removed missing values
- Encoded categorical variables
- Scaled numerical features

## Features Used
- raisedhands
- VisITedResources
- AnnouncementsView
- Discussion
- Absence Days
- Parent Satisfaction
- Demographics

## Model
Random Forest Classifier

## Evaluation
- Accuracy
- Recall
- ROC-AUC

Focus on high recall to catch dropouts early.

## Risk Thresholds
- High: ≥ 0.7
- Medium: 0.4–0.7
- Low: < 0.4

## Explanation
Key reasons for dropout:
- Low participation
- High absences
- Low LMS usage
- Poor parent satisfaction

## Advisor Actions
- Counseling
- Academic support
- Attendance monitoring
- Parent meetings

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/518122d4-1d72-4356-b0ff-e47294494a66" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/0681de85-519e-4d6c-aa2a-b9cb13ea9f05" />
