import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# Load the datasets
prepared_games_path = 'MLB_Games_Prepared.xlsx'
prepared_averages_path = 'MLB_Averages.xlsx'
today_games_path = 'MLB_Games_Today.xlsx'

prepared_games_df = pd.read_excel(prepared_games_path)
prepared_averages_df = pd.read_excel(prepared_averages_path)
today_games_df = pd.read_excel(today_games_path)

# Remove empty rows in both datasets
prepared_games_df.dropna(how='all', inplace=True)
today_games_df.dropna(how='all', inplace=True)

# Aggregated stats for prepared_games_df (historical games)
prepared_games_df['H_SP_ERA'] = prepared_averages_df['SP ERA'].iloc[0] / np.where(prepared_games_df['A SP ERA'] == 0, 1, prepared_games_df['A SP ERA'])
prepared_games_df['A_SP_ERA'] = prepared_averages_df['SP ERA'].iloc[0] / np.where(prepared_games_df['H SP ERA'] == 0, 1, prepared_games_df['H SP ERA'])

prepared_games_df['H_SP_WHIP'] = prepared_averages_df['SP WHIP'].iloc[0] / np.where(prepared_games_df['A SP WHIP'] == 0, 1, prepared_games_df['A SP WHIP'])
prepared_games_df['A_SP_WHIP'] = prepared_averages_df['SP WHIP'].iloc[0] / np.where(prepared_games_df['H SP WHIP'] == 0, 1, prepared_games_df['H SP WHIP'])

# prepared_games_df['H_SP_1st_ERA'] = prepared_averages_df['SP 1st Inn ERA'].iloc[0] / np.where(prepared_games_df['A SP 1st Inn ERA'] == 0, 1, prepared_games_df['A SP 1st Inn ERA'])
# prepared_games_df['A_SP_1st_ERA'] = prepared_averages_df['SP 1st Inn ERA'].iloc[0] / np.where(prepared_games_df['H SP 1st Inn ERA'] == 0, 1, prepared_games_df['H SP 1st Inn ERA'])

# prepared_games_df['H_SP_1st_WHIP'] = prepared_averages_df['SP 1st Inn WHIP'].iloc[0] / np.where(prepared_games_df['A SP 1st Inn WHIP'] == 0, 1, prepared_games_df['A SP 1st Inn WHIP'])
# prepared_games_df['A_SP_1st_WHIP'] = prepared_averages_df['SP 1st Inn WHIP'].iloc[0] / np.where(prepared_games_df['H SP 1st Inn WHIP'] == 0, 1, prepared_games_df['H SP 1st Inn WHIP'])

prepared_games_df['H_Bat_AVG'] = prepared_games_df[[f'A Hitter{i}_AVG' for i in range(1, 10)]].sum(axis=1) / prepared_averages_df[[f'Hitter{i}_AVG' for i in range(1, 10)]].sum(axis=1).iloc[0]
prepared_games_df['A_Bat_AVG'] = prepared_games_df[[f'H Hitter{i}_AVG' for i in range(1, 10)]].sum(axis=1) / prepared_averages_df[[f'Hitter{i}_AVG' for i in range(1, 10)]].sum(axis=1).iloc[0]

prepared_games_df['H_Bat_OBP'] = prepared_games_df[[f'A Hitter{i}_OBP' for i in range(1, 10)]].sum(axis=1) / prepared_averages_df[[f'Hitter{i}_OBP' for i in range(1, 10)]].sum(axis=1).iloc[0]
prepared_games_df['A_Bat_OBP'] = prepared_games_df[[f'H Hitter{i}_OBP' for i in range(1, 10)]].sum(axis=1) / prepared_averages_df[[f'Hitter{i}_OBP' for i in range(1, 10)]].sum(axis=1).iloc[0]

prepared_games_df['H_Bat_SLG'] = prepared_games_df[[f'A Hitter{i}_SLG' for i in range(1, 10)]].sum(axis=1) / prepared_averages_df[[f'Hitter{i}_SLG' for i in range(1, 10)]].sum(axis=1).iloc[0]
prepared_games_df['A_Bat_SLG'] = prepared_games_df[[f'H Hitter{i}_SLG' for i in range(1, 10)]].sum(axis=1) / prepared_averages_df[[f'Hitter{i}_SLG' for i in range(1, 10)]].sum(axis=1).iloc[0]

prepared_games_df['H_Bat_OPS'] = prepared_games_df[[f'A Hitter{i}_OPS' for i in range(1, 10)]].sum(axis=1) / prepared_averages_df[[f'Hitter{i}_OPS' for i in range(1, 10)]].sum(axis=1).iloc[0]
prepared_games_df['A_Bat_OPS'] = prepared_games_df[[f'H Hitter{i}_OPS' for i in range(1, 10)]].sum(axis=1) / prepared_averages_df[[f'Hitter{i}_OPS' for i in range(1, 10)]].sum(axis=1).iloc[0]

prepared_games_df['H_Win_Mult'] = (prepared_games_df['A Wins'] / (prepared_games_df['A Wins'] + prepared_games_df['A Losses'])) * 2
prepared_games_df['A_Win_Mult'] = (prepared_games_df['H Wins'] / (prepared_games_df['H Wins'] + prepared_games_df['H Losses'])) * 2

prepared_games_df['H_Home_Mult'] = (prepared_games_df['A Home Wins'] / (prepared_games_df['A Home Wins'] + prepared_games_df['A Home Losses'])) * 2
prepared_games_df['A_Away_Mult'] = (prepared_games_df['H Away Wins'] / (prepared_games_df['H Away Wins'] + prepared_games_df['H Away Losses'])) * 2

prepared_games_df['H_L10_Mult'] = (prepared_games_df['A L10 Wins'] / (prepared_games_df['A L10 Wins'] + prepared_games_df['A L10 Losses'])) * 2
prepared_games_df['A_L10_Mult'] = (prepared_games_df['H L10 Wins'] / (prepared_games_df['H L10 Wins'] + prepared_games_df['H L10 Losses'])) * 2

prepared_games_df['H_Bullpen_ERA'] = prepared_averages_df['Bullpen ERA'].iloc[0] / np.where(prepared_games_df['A Bullpen ERA'] == 0, 1, prepared_games_df['A Bullpen ERA'])
prepared_games_df['A_Bullpen_ERA'] = prepared_averages_df['Bullpen ERA'].iloc[0] / np.where(prepared_games_df['H Bullpen ERA'] == 0, 1, prepared_games_df['H Bullpen ERA'])

prepared_games_df['H_9th_Bullpen_ERA'] = prepared_averages_df['9th Inn ERA'].iloc[0] / np.where(prepared_games_df['A 9th Inn ERA'] == 0, 1, prepared_games_df['A 9th Inn ERA'])
prepared_games_df['A_9th_Bullpen_ERA'] = prepared_averages_df['9th Inn ERA'].iloc[0] / np.where(prepared_games_df['H 9th Inn ERA'] == 0, 1, prepared_games_df['H 9th Inn ERA'])

# prepared_games_df['H_Bat_Runs_1st'] = prepared_games_df['A 1st Inn Runs/G'] / prepared_averages_df['1st Inn Runs/G'].iloc[0]
# prepared_games_df['A_Bat_Runs_1st'] = prepared_games_df['H 1st Inn Runs/G'] / prepared_averages_df['1st Inn Runs/G'].iloc[0]

# prepared_games_df['H_Bat_AVG_1st'] = prepared_games_df['A 1st AVG'] / prepared_averages_df['1st AVG'].iloc[0]
# prepared_games_df['A_Bat_AVG_1st'] = prepared_games_df['H 1st AVG'] / prepared_averages_df['1st AVG'].iloc[0]

# prepared_games_df['H_Bat_OBP_1st'] = prepared_games_df['A 1st OBP'] / prepared_averages_df['1st OBP'].iloc[0]
# prepared_games_df['A_Bat_OBP_1st'] = prepared_games_df['H 1st OBP'] / prepared_averages_df['1st OBP'].iloc[0]

# prepared_games_df['H_Bat_SLG_1st'] = prepared_games_df['A 1st SLG'] / prepared_averages_df['1st SLG'].iloc[0]
# prepared_games_df['A_Bat_SLG_1st'] = prepared_games_df['H 1st SLG'] / prepared_averages_df['1st SLG'].iloc[0]

# prepared_games_df['H_Bat_OPS_1st'] = prepared_games_df['A 1st OPS'] / prepared_averages_df['1st OPS'].iloc[0]
# prepared_games_df['A_Bat_OPS_1st'] = prepared_games_df['H 1st OPS'] / prepared_averages_df['1st OPS'].iloc[0]



# Create new features for today's games (MLB_Games_Today)
today_games_df['H_SP_ERA'] = prepared_averages_df['SP ERA'].iloc[0] / np.where(today_games_df['H SP ERA'] == 0, 1, today_games_df['H SP ERA'])
today_games_df['A_SP_ERA'] = prepared_averages_df['SP ERA'].iloc[0] / np.where(today_games_df['A SP ERA'] == 0, 1, today_games_df['A SP ERA'])

today_games_df['H_SP_WHIP'] = prepared_averages_df['SP WHIP'].iloc[0] / np.where(today_games_df['H SP WHIP'] == 0, 1, today_games_df['H SP WHIP'])
today_games_df['A_SP_WHIP'] = prepared_averages_df['SP WHIP'].iloc[0] / np.where(today_games_df['A SP WHIP'] == 0, 1, today_games_df['A SP WHIP'])

# today_games_df['H_SP_1st_ERA'] = prepared_averages_df['SP 1st Inn ERA'].iloc[0] / np.where(today_games_df['H SP 1st Inn ERA'] == 0, 1, today_games_df['H SP 1st Inn ERA'])
# today_games_df['A_SP_1st_ERA'] = prepared_averages_df['SP 1st Inn ERA'].iloc[0] / np.where(today_games_df['A SP 1st Inn ERA'] == 0, 1, today_games_df['A SP 1st Inn ERA'])

# today_games_df['H_SP_1st_WHIP'] = prepared_averages_df['SP 1st Inn WHIP'].iloc[0] / np.where(today_games_df['H SP 1st Inn WHIP'] == 0, 1, today_games_df['H SP 1st Inn WHIP'])
# today_games_df['A_SP_1st_WHIP'] = prepared_averages_df['SP 1st Inn WHIP'].iloc[0] / np.where(today_games_df['A SP 1st Inn WHIP'] == 0, 1, today_games_df['A SP 1st Inn WHIP'])

today_games_df['H_Bat_AVG'] = today_games_df[[f'H Hitter{i}_AVG' for i in range(1, 10)]].sum(axis=1) / prepared_averages_df[[f'Hitter{i}_AVG' for i in range(1, 10)]].sum(axis=1).iloc[0]
today_games_df['A_Bat_AVG'] = today_games_df[[f'A Hitter{i}_AVG' for i in range(1, 10)]].sum(axis=1) / prepared_averages_df[[f'Hitter{i}_AVG' for i in range(1, 10)]].sum(axis=1).iloc[0]

today_games_df['H_Bat_OBP'] = today_games_df[[f'H Hitter{i}_OBP' for i in range(1, 10)]].sum(axis=1) / prepared_averages_df[[f'Hitter{i}_OBP' for i in range(1, 10)]].sum(axis=1).iloc[0]
today_games_df['A_Bat_OBP'] = today_games_df[[f'A Hitter{i}_OBP' for i in range(1, 10)]].sum(axis=1) / prepared_averages_df[[f'Hitter{i}_OBP' for i in range(1, 10)]].sum(axis=1).iloc[0]

today_games_df['H_Bat_SLG'] = today_games_df[[f'H Hitter{i}_SLG' for i in range(1, 10)]].sum(axis=1) / prepared_averages_df[[f'Hitter{i}_SLG' for i in range(1, 10)]].sum(axis=1).iloc[0]
today_games_df['A_Bat_SLG'] = today_games_df[[f'A Hitter{i}_SLG' for i in range(1, 10)]].sum(axis=1) / prepared_averages_df[[f'Hitter{i}_SLG' for i in range(1, 10)]].sum(axis=1).iloc[0]

today_games_df['H_Bat_OPS'] = today_games_df[[f'H Hitter{i}_OPS' for i in range(1, 10)]].sum(axis=1) / prepared_averages_df[[f'Hitter{i}_OPS' for i in range(1, 10)]].sum(axis=1).iloc[0]
today_games_df['A_Bat_OPS'] = today_games_df[[f'A Hitter{i}_OPS' for i in range(1, 10)]].sum(axis=1) / prepared_averages_df[[f'Hitter{i}_OPS' for i in range(1, 10)]].sum(axis=1).iloc[0]

today_games_df['H_Win_Mult'] = (today_games_df['H Wins'] / (today_games_df['H Wins'] + today_games_df['H Losses'])) * 2
today_games_df['A_Win_Mult'] = (today_games_df['A Wins'] / (today_games_df['A Wins'] + today_games_df['A Losses'])) * 2

today_games_df['H_Home_Mult'] = (today_games_df['H Home Wins'] / (today_games_df['H Home Wins'] + today_games_df['H Home Losses'])) * 2
today_games_df['A_Away_Mult'] = (today_games_df['A Away Wins'] / (today_games_df['A Away Wins'] + today_games_df['A Away Losses'])) * 2

today_games_df['H_L10_Mult'] = (today_games_df['H L10 Wins'] / (today_games_df['H L10 Wins'] + today_games_df['H L10 Losses'])) * 2
today_games_df['A_L10_Mult'] = (today_games_df['A L10 Wins'] / (today_games_df['A L10 Wins'] + today_games_df['A L10 Losses'])) * 2

today_games_df['H_Bullpen_ERA'] = prepared_averages_df['Bullpen ERA'].iloc[0] / np.where(today_games_df['H Bullpen ERA'] == 0, 1, today_games_df['H Bullpen ERA'])
today_games_df['A_Bullpen_ERA'] = prepared_averages_df['Bullpen ERA'].iloc[0] / np.where(today_games_df['A Bullpen ERA'] == 0, 1, today_games_df['A Bullpen ERA'])

today_games_df['H_9th_Bullpen_ERA'] = prepared_averages_df['9th Inn ERA'].iloc[0] / np.where(today_games_df['H 9th Inn ERA'] == 0, 1, today_games_df['H 9th Inn ERA'])
today_games_df['A_9th_Bullpen_ERA'] = prepared_averages_df['9th Inn ERA'].iloc[0] / np.where(today_games_df['A 9th Inn ERA'] == 0, 1, today_games_df['A 9th Inn ERA'])

# today_games_df['H_Bat_Runs_1st'] = today_games_df['H 1st Inn Runs/G'] / prepared_averages_df['1st Inn Runs/G'].iloc[0]
# today_games_df['A_Bat_Runs_1st'] = today_games_df['A 1st Inn Runs/G'] / prepared_averages_df['1st Inn Runs/G'].iloc[0]

# today_games_df['H_Bat_AVG_1st'] = today_games_df['H 1st AVG'] / prepared_averages_df['1st AVG'].iloc[0]
# today_games_df['A_Bat_AVG_1st'] = today_games_df['A 1st AVG'] / prepared_averages_df['1st AVG'].iloc[0]

# today_games_df['H_Bat_OBP_1st'] = today_games_df['H 1st OBP'] / prepared_averages_df['1st OBP'].iloc[0]
# today_games_df['A_Bat_OBP_1st'] = today_games_df['A 1st OBP'] / prepared_averages_df['1st OBP'].iloc[0]

# today_games_df['H_Bat_SLG_1st'] = today_games_df['H 1st SLG'] / prepared_averages_df['1st SLG'].iloc[0]
# today_games_df['A_Bat_SLG_1st'] = today_games_df['A 1st SLG'] / prepared_averages_df['1st SLG'].iloc[0]

# today_games_df['H_Bat_OPS_1st'] = today_games_df['H 1st OPS'] / prepared_averages_df['1st OPS'].iloc[0]
# today_games_df['A_Bat_OPS_1st'] = today_games_df['A 1st OPS'] / prepared_averages_df['1st OPS'].iloc[0]

# Align today's game features with those from prepared_games_df
new_features = [
    'H_Bat_AVG', 'A_Bat_AVG', 
    'H_Bat_OBP', 'A_Bat_OBP', 
    'H_Bat_SLG', 'A_Bat_SLG', 
    'H_Bat_OPS', 'A_Bat_OPS', 
    'H_SP_ERA', 'A_SP_ERA', 
    'H_SP_WHIP', 'A_SP_WHIP', 
    # 'H_SP_1st_ERA', 'A_SP_1st_ERA', 
    # 'H_SP_1st_WHIP', 'A_SP_1st_WHIP', 
    'H_Win_Mult', 'A_Win_Mult', 
    'H_Home_Mult', 'A_Away_Mult', 
    'H_L10_Mult', 'A_L10_Mult', 
    'H_Bullpen_ERA', 'A_Bullpen_ERA', 
    'H_9th_Bullpen_ERA', 'A_9th_Bullpen_ERA', 
    # 'H_Bat_Runs_1st', 'A_Bat_Runs_1st', 
    # 'H_Bat_AVG_1st', 'A_Bat_AVG_1st', 
    # 'H_Bat_OBP_1st', 'A_Bat_OBP_1st', 
    # 'H_Bat_SLG_1st', 'A_Bat_SLG_1st', 
    # 'H_Bat_OPS_1st', 'A_Bat_OPS_1st'
]
today_features = today_games_df.reindex(columns=new_features, fill_value=0)

# Prepare the features and target for the model
features = prepared_games_df[new_features]
target = (prepared_games_df['H Score'] > prepared_games_df['A Score']).astype(int)  # 1 if home team wins, 0 otherwise

# Label Encoding for categorical variables (if any, but likely none for aggregated stats)
label_encoders = {}
for column in features.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    features[column] = le.fit_transform(features[column])
    today_features[column] = le.transform(today_features[column])
    label_encoders[column] = le

# Apply SMOTE for balancing the target classes
smote = SMOTE(random_state=42)
features_smote, target_smote = smote.fit_resample(features, target)

# Reset the index of 'today_games_df' to align properly after dropping empty rows
today_games_df.reset_index(drop=True, inplace=True)

# Simple holdout split for fast validation
X_train, X_val, y_train, y_val = train_test_split(
    features_smote,
    target_smote,
    test_size=0.2,
    random_state=42,
    stratify=target_smote
)

# Fast fixed Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

# Train model
model.fit(X_train, y_train)

# Validate model
y_val_pred = model.predict(X_val)
best_accuracy = accuracy_score(y_val, y_val_pred)

# Predict probabilities for today's games
today_probabilities = model.predict_proba(today_features)

confidence_threshold = 0.55
best_confident_predictions = []

for i, prob in enumerate(today_probabilities):
    if np.max(prob) >= confidence_threshold:
        prediction = np.argmax(prob)
        probability = np.max(prob)
        home_team = today_games_df.loc[i, 'H Team']
        away_team = today_games_df.loc[i, 'A Team']

        if prediction == 1:
            result = f"{home_team} ML {probability:.2%}"
        else:
            result = f"{away_team} ML {probability:.2%}"

        best_confident_predictions.append(result)
    else:
        best_confident_predictions.append("Uncertain")

# Print results
print(f"Validation Accuracy: {best_accuracy:.4f}")
print("Today's Confident Predictions:")
for prediction in best_confident_predictions:
    if prediction != "Uncertain":
        print(prediction)

#print(target.value_counts())     # Only if you want to know how many YRFI and NRFI there are