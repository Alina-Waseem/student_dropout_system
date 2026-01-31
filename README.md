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
