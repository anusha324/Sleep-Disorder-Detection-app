import numpy as np
import pandas as pd

np.random.seed(42)

# Load dataset
input_file = 'sleep_disorder_original.csv'
df = pd.read_csv(input_file)

# Features to jitter
num_cols = [
    'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
    'Stress Level', 'Heart Rate', 'Daily Steps'
]

# 1. Add noise (jitter) to all rows (20% of std)
for col in num_cols:
    std = df[col].std()
    noise = np.random.normal(0, 0.20 * std, size=len(df))
    df[col] += noise
    if col == 'Sleep Duration':
        df[col] = df[col].clip(2, 10)
    elif col == 'Quality of Sleep':
        df[col] = df[col].clip(1, 10)
    elif col == 'Stress Level':
        df[col] = df[col].clip(1, 10)
    elif col == 'Physical Activity Level':
        df[col] = df[col].clip(5, 100)
    elif col == 'Heart Rate':
        df[col] = df[col].clip(55, 110)
    elif col == 'Daily Steps':
        df[col] = df[col].clip(1000, 13000)

# 2. Random swapping of 15% of rows
swap_frac = 0.15
num_swap = int(len(df) * swap_frac)
swap_indices = np.random.choice(df.index, size=num_swap, replace=False)
for idx in swap_indices:
    this_class = df.at[idx, 'Sleep Disorder']
    other_classes = df[df['Sleep Disorder'] != this_class]
    if not other_classes.empty:
        swap_row = other_classes.sample(1, random_state=np.random.randint(0, 10000))
        for col in num_cols:
            df.at[idx, col] = swap_row.iloc[0][col]

# 3. Improve realistic relationships with more natural overlap
def improve_realistic_relationships(row):
    disorder = row['Sleep Disorder']

    # ✳️ STRONG RELATIONSHIPS — HIGH PRIORITY FEATURES
    if disorder == 'None':
        row['Stress Level'] = int(np.random.normal(3, 1.0))
        row['Quality of Sleep'] = int(np.random.normal(8.0, 1.0))
        row['Sleep Duration'] = int(np.random.normal(7.5, 1.0))
        row['Physical Activity Level'] = int(np.random.normal(75, 10))
    elif disorder == 'Insomnia':
        row['Stress Level'] = int(np.random.normal(8, 1.0))
        row['Quality of Sleep'] = int(np.random.normal(3.5, 1.0))
        row['Sleep Duration'] = int(np.random.normal(4.5, 1.0))
        row['Physical Activity Level'] = int(np.random.normal(35, 10))
    elif disorder == 'Sleep Apnea':
        row['Stress Level'] = int(np.random.normal(6.5, 1.0))
        row['Quality of Sleep'] = int(np.random.normal(4.5, 1.0))
        row['Sleep Duration'] = int(np.random.normal(5.5, 1.0))
        row['Physical Activity Level'] = int(np.random.normal(45, 10))

    # ✅ Moderate features
    if disorder == 'None':
        row['Heart Rate'] = int(np.random.normal(65, 6))
        row['Daily Steps'] = int(np.random.normal(10000, 1000))
    elif disorder == 'Insomnia':
        row['Heart Rate'] = int(np.random.normal(78, 6))
        row['Daily Steps'] = int(np.random.normal(6000, 1200))
    elif disorder == 'Sleep Apnea':
        row['Heart Rate'] = int(np.random.normal(82, 7))
        row['Daily Steps'] = int(np.random.normal(5500, 1000))

    # ✅ Softer age patterns by disorder
    if disorder == 'Sleep Apnea':
        row['Age'] = int(np.random.normal(50, 14))
    elif disorder == 'Insomnia':
        row['Age'] = int(np.random.normal(42, 16))
    else:
        row['Age'] = int(np.random.normal(36, 16))
    row['Age'] = np.clip(row['Age'], 18, 80)

    # ✅ BMI Category
    if disorder == 'Sleep Apnea':
        row['BMI Category'] = 'Overweight' if np.random.rand() < 0.7 else 'Normal'
    elif disorder == 'Insomnia':
        row['BMI Category'] = 'Overweight' if np.random.rand() < 0.5 else 'Normal'
    else:
        row['BMI Category'] = 'Normal' if np.random.rand() < 0.6 else 'Overweight'

    # ❌ Weaker gender effect
    row['Gender'] = np.random.choice(['Male', 'Female'])

    
    # ✅ Clip strongly related variables
    row['Sleep Duration'] = np.clip(row['Sleep Duration'], 2, 10)
    row['Quality of Sleep'] = np.clip(row['Quality of Sleep'], 1, 10)
    row['Stress Level'] = np.clip(row['Stress Level'], 1, 10)
    row['Heart Rate'] = np.clip(row['Heart Rate'], 55, 110)
    row['Daily Steps'] = np.clip(row['Daily Steps'], 1000, 13000)
    row['Physical Activity Level'] = np.clip(row['Physical Activity Level'], 5, 100)

    return row

df = df.apply(improve_realistic_relationships, axis=1)

# 4. Add small label noise (5%)
label_noise_frac = 0.05
num_noisy = int(len(df) * label_noise_frac)
noise_idx = np.random.choice(df.index, size=num_noisy, replace=False)

all_classes = df['Sleep Disorder'].unique()
for i in noise_idx:
    current = df.at[i, 'Sleep Disorder']
    new_class = np.random.choice([c for c in all_classes if c != current])
    df.at[i, 'Sleep Disorder'] = new_class

# 5. Round numerical columns to int
for col in num_cols:
    df[col] = df[col].round().astype(int)

# Save the new dataset
output_file = 'sleep_disorder_dataset.csv'
df.to_csv(output_file, index=False)
print(f'✅ Realistic dataset saved as {output_file}')
