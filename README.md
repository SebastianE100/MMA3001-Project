# MMA3001-Project
Building Occupancy Forecasting (MMA3001 Project)

Predicts the hourly headcount of a university building one week ahead using a decision tree, to support heating and cooling scheduling. The model uses only inputs known in advance (day of week, time of day, and teaching/exam period), so no occupancy readings from the forecast week are needed.

Data source: Monash Smart Infrastructure Occupancy and Environmental Data, MMA3001 Project Datasets

Repository structure
File	Purpose
DataPrep.py	Cleans the raw sensor CSV and writes PeopleData.parquet
Baseline_Model.py	Tests two baselines (Mean Time, Mean Day) on chronological folds
Regression_Model.py	Alternative model: one quadratic polynomial in time for each day of the week
Complexity_Investigation.py	Compares decision trees using Time, Time + Day, and Time + Day + Context (class/exam flags) against the Mean Time baseline
Depth_Sweep.py	Sweeps tree depth (1 to 6)
Weight_Sweep.py	Sweeps the sample weight on empty (zero-headcount) hours and reports false positive and false negative rates

Setup

Python 3.10 or later. root_mean_squared_error needs scikit-learn 1.4 or later.

bash
pip install pandas numpy matplotlib scikit-learn pyarrow
Data
Raw input: a CSV with the columns collecteddate (ISO 8601 timestamp, read as UTC) and headcount (number of people). The original file is 5occupancySensor_MayToDec2024_9MRows.csv (about 9 million rows).
Processed output: PeopleData.parquet, created by DataPrep.py and read by every other script.
What DataPrep.py does
Reads collecteddate and headcount.
Removes rows with duplicate timestamps (keeps the first).
Converts timestamps to Australia/Melbourne local time.
Resamples to 60-minute intervals using the maximum headcount in each interval.
Sets intervals with no reading to 0, then removes calendar days with no readings at all (full-day sensor outages). Outages within a day are not detected and remain as zeros.
Adds Day, Date, Time (hour of day as a float), and boolean Class and Exam flags.
Writes PeopleData.parquet.

The Class and Exam flags come from teaching and exam period dates for 2024 to 2026 that are hard-coded in the script, taken from the Monash academic calendar.

Columns in PeopleData.parquet
Column	Type	Meaning
collecteddate	datetime (Australia/Melbourne)	Start of the hourly interval
headcount	number	Maximum people detected in the interval (0 or more)
Day	string	Monday to Sunday
Date	date	Calendar date
Time	float	Hour of day, 0 to 23
Class	bool	True during a teaching period
Exam	bool	True during an exam period
How to run

Run from the repository root, in this order:

bash
python DataPrep.py                 # 1. build PeopleData.parquet
python Baseline_Model.py           # 2. baselines
python Regression_Model.py         # 3. polynomial alternative
python Complexity_Investigation.py # 4. which inputs the tree needs
python Depth_Sweep.py              # 5. tree depth
python Weight_Sweep.py             # 6. weighting of empty hours
pytest                             # 7. tests (once added)

Before running: the scripts save figures to a hard-coded folder, C:/Users/Octob/Documents/Uni/2026/MMA3001 Phots/, and DataPrep.py reads the CSV from Documents/Uni/2026/.... Change these paths (for example to figures/ and data/), or the scripts will fail on another machine.

Validation method
Chronological cross-validation: TimeSeriesSplit with 5 folds on the hourly data. Each fold trains only on earlier data and validates on the next block, which matches real forecasting and avoids leaking the future into training.
Metrics: RMSE and R² (training and validation), plus false positive and false negative rates for the occupied/empty decision.
Occupied/empty: the building is empty when headcount is 2 or fewer. A false positive is an empty hour predicted as occupied (wasted energy). A false negative is an occupied hour predicted as empty (discomfort).
Baseline: Mean Time, the average training headcount for each time of day.
Fold 1 is excluded when comparing models, because its first training block is small and unrepresentative of its validation block.
Inputs and outputs of the final model
	Description
Inputs	dayNum (0 = Monday to 6 = Sunday), time (hours, 0 to under 24), class_flag, exam_flag
Output	Predicted hourly headcount (people, 0 or more)

Known limitations
One building and one period of data, so results may not transfer to other buildings or semesters.
Hourly maximum headcount can overstate occupancy within an hour.
Only full-day sensor outages are removed. Partial outages remain as zeros.
Inputs are calendar-based only: no weather, public holidays or events.
Teaching and exam periods are assumed known a week ahead.
A decision tree always returns a value, including for period/day combinations not seen in training, with no warning.
Validation R² is modest (roughly 0.3), so the model suits scheduling decisions, not exact counts.
Sensor headcount is treated as ground truth. Its accuracy is unverified.
Model settings were chosen using the same folds that report performance, so scores are slightly optimistic.

Licence
GNU GENERAL PUBLIC LICENSE
Author
Sebasitan Esposito