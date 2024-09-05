import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from DataTransformation import LowPassFilter,  PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction 
from FrquencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans
import plotly.express as px

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed_chauevenets.pkl")

predictor_columns = df.columns[:6].tolist()

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20,5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

df[df['set'] == 35]['acc_y'].plot()

df.info()

for col in predictor_columns:
    df[col] = df[col].interpolate()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

duration = df[df['set']==1]['acc_y'].index[-1] - df[df['set']==1]['acc_y'].index[0]
duration.seconds

for set in df['set'].unique():
    start = df[df['set'] == set].index[0]
    end = df[df['set'] == set].index[-1]
    
    duration = end - start
    
    df.loc[(df['set'] == set), "duration"] = duration.seconds
    
    
duration_df = df.groupby(['category'])['duration'].mean()
    
duration_df.iloc[1] / 10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()

LowPass = LowPassFilter()

# Frequency

fs = 1000/200  
cutoff = 1.3
df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5 )



subset = df_lowpass[df_lowpass['set'] == 35]
print(subset['label'][0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
ax[0].plot(subset['acc_y'].reset_index(drop=True), label='raw data')
ax[1].plot(subset['acc_y_lowpass'].reset_index(drop=True), label='Filtered data')

ax[0].legend(loc="upper center", bbox_to_anchor= (0.5, 1.15), ncol=3, fancybox=True, shadow= True)
ax[1].legend(loc="upper center", bbox_to_anchor= (0.5, 1.15), ncol=3, fancybox=True, shadow= True)



for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass,col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col +"_lowpass"]
    del df_lowpass[col +"_lowpass"]
    
df_lowpass

len(predictor_columns)

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()

PCA = PrincipalComponentAnalysis()

pca_values = PCA.determine_pc_explained_variance(df_lowpass, predictor_columns)

plt.figure(figsize=(10,10))
plt.plot(range(1,len(predictor_columns)+1), pca_values)
plt.ylabel('explained variance')
plt.xlabel('Principal components')
plt.show()



df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)


subset = df_pca[df_pca['set'] == 23]
subset[['pca_1', 'pca_2', 'pca_3']].plot()

subset = df_pca[df_pca['set'] == 23]
subset[['acc_x', 'acc_y', 'acc_z']].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

acc_r = df_squared['acc_x']**2 + df_squared['acc_y']**2 + df_squared['acc_z']**2
gyr_r = df_squared['gyr_x']**2 + df_squared['gyr_y']**2 + df_squared['gyr_z']**2

df_squared['acc_r'] = np.sqrt(acc_r)
df_squared['gyr_r'] = np.sqrt(gyr_r)

subset = df_squared[df_squared['set'] == 14]
subset[['acc_r', 'gyr_r']].plot(subplots=True)


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()

numAbs = NumericalAbstraction()

predictor_columns += ['acc_r', 'gyr_r']

ws = int(1000/200)
 
for col in predictor_columns:
    df_temporal = numAbs.abstract_numerical(df_temporal, [col], ws, 'mean')
    df_temporal = numAbs.abstract_numerical(df_temporal, [col], ws, 'std')
    
df_temporal_list = []

for s in df_temporal['set'].unique():
    subset = df_temporal[df_temporal['set'] == s].copy()
    
    for col in predictor_columns:
        subset = numAbs.abstract_numerical(subset, [col], ws, 'mean')
        subset = numAbs.abstract_numerical(subset, [col], ws, 'std')
    
    df_temporal_list.append(subset)
    
df_temporal = pd.concat(df_temporal_list)

subset[['acc_y', 'acc_y_temp_mean_ws_5', 'acc_y_temp_std_ws_5']].plot()


subset[['gyr_y', 'gyr_y_temp_mean_ws_5', 'gyr_y_temp_std_ws_5']].plot()
 
# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index()

fs = int(1000/200)
# Average length of a repitition
ws = int(2800/200)

FreqAbs = FourierTransformation()

df_freq = FreqAbs.abstract_frequency(df_freq, ['acc_y'], ws, fs) 


subset = df_freq[df_freq['set'] == 25]
subset[['acc_y']].plot()

subset[
    [
        'acc_y_max_freq',
        'acc_y_freq_weighted',
        'acc_y_pse',
        'acc_y_freq_2.143_Hz_ws_14',
        'acc_y_freq_2.5_Hz_ws_14'
        ]
    ].plot()

df_freq_list = []

for s in df_freq['set'].unique():
    print(f'Applying Fourier Transformatiion to set {s}')
    subset = df_freq[df_freq['set'] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq = df_freq.dropna()
# Reduces the risk of Overfitting
df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()

cluster_columns = ['acc_x', 'acc_y', 'acc_z']
k_values = range(2,10)
inertias = []

for  k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=1)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10,10))
plt.plot(k_values, inertias)
plt.ylabel('Sum of Squared Distances')
plt.show()


kmeans = KMeans(n_clusters=5, n_init=20, random_state=1)
subset = df_cluster[cluster_columns]
df_cluster['cluster'] = kmeans.fit_predict(subset)

df_cluster['cluster'].unique()


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster['cluster'].unique():
    subset = df_cluster[df_cluster['cluster'] == c]
    ax.scatter(subset['acc_x'], subset['acc_y'], subset['acc_z'], label=c)
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
plt.legend()
plt.show()


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection="3d")
for l in df_cluster['label'].unique():
    subset = df_cluster[df_cluster['label'] == l]
    ax.scatter(subset['acc_x'], subset['acc_y'], subset['acc_z'], label=l)
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
plt.legend()
plt.show()





fig = px.scatter_3d(df_cluster, x = 'acc_x', y='acc_y', z='acc_z',
                 labels={
                     "x": "Feature 1",
                     "y": "Feature 2",
                     "z": "Feature 3"
                 },
                 opacity=1,color = 'label')

# Change chart background color
fig.update_layout(dict(plot_bgcolor = 'white'))

# Update axes lines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                 showline=True, linewidth=1, linecolor='black')

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                 showline=True, linewidth=1, linecolor='black')

# Set figure title
fig.update_layout(title_text="Scatter plot for labels")

# Update marker size
fig.update_traces(marker=dict(size=1.5))

fig.show()




fig = px.scatter_3d(df_cluster, x = 'acc_x', y='acc_y', z='acc_z',
                 labels={
                     "x": "Feature 1",
                     "y": "Feature 2",
                     "z": "Feature 3"
                 },
                 opacity=1,color = 'cluster')

# Change chart background color
fig.update_layout(dict(plot_bgcolor = 'white'))

# Update axes lines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                 showline=True, linewidth=1, linecolor='black')

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                 showline=True, linewidth=1, linecolor='black')

# Set figure title
fig.update_layout(title_text="Scatter for clusters")

# Update marker size
fig.update_traces(marker=dict(size=1.5))

fig.show()

# ------------------------------ --------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")