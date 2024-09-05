import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib as mpl
from cycler import cycler


# --------------------------------------------------------------
# Read the pickle file 
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

set_df = df[df['set'] == 1]

plt.plot(set_df['acc_y'])

plt.plot(set_df['acc_y'].reset_index(drop=True))


# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

for label in df["label"].unique():
    subset = df[df["label"] == label] 
    fig, ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()
    
    
for label in df["label"].unique():
    subset = df[df["label"] == label] 
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()
    
    
# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------

mpl.style.use("seaborn-v0_8-darkgrid")
mpl.rcParams["figure.figsize"] = (20, 5)

# Helps export figure with good resolution 
mpl.rcParams["figure.dpi"] = 100  
   

print(plt.style.available)

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

df.head()

category_df = df.query("label == 'squat'").query("participant == 'E'").reset_index()  

fig, ax = plt.subplots()
category_df.groupby(["category"])['acc_y'].plot()
ax.set_xlabel("Samples")
ax.set_ylabel("acc_y")
plt.legend()
plt.show()



category_df = df.query("label == 'ohp'").query("participant == 'B'").reset_index()  

fig, ax = plt.subplots()
category_df.groupby(["category"])['acc_y'].plot()
ax.set_xlabel("Samples")
ax.set_ylabel("acc_y")
plt.legend()
plt.show()


# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

participant_df = df.query("label == 'bench'").sort_values("participant").reset_index()
fig, ax = plt.subplots()
participant_df.groupby(["participant"])['acc_y'].plot()
ax.set_xlabel("Samples")
ax.set_ylabel("acc_y")
plt.legend()
plt.show()


participant_df = df.query("label == 'ohp'").sort_values("participant").reset_index()
fig, ax = plt.subplots()
participant_df.groupby(["participant"])['acc_y'].plot()
ax.set_xlabel("Samples")
ax.set_ylabel("acc_y")
plt.legend()
plt.show()


# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

label = 'squat'
participant = 'A'

all_axis_df =  df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()
fig, ax = plt.subplots()
all_axis_df[['acc_x', 'acc_y', 'acc_z']].plot(ax=ax)
ax.set_xlabel("Samples")
ax.set_ylabel("acc_x | acc_y | acc_z ")
plt.legend()
plt.show()



label = 'squat'
participant = 'C'

all_axis_df =  df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()
fig, ax = plt.subplots()
all_axis_df[['acc_x', 'acc_y', 'acc_z']].plot(ax=ax)
ax.set_xlabel("Samples")
ax.set_ylabel("acc_x | acc_y | acc_z ")
plt.legend()
plt.show()


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

labels = df['label'].unique()
partipants = df['participant'].unique()

for label in labels:
    for participant in partipants:
        all_axis_df =  (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'").reset_index()
        )
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[['acc_x', 'acc_y', 'acc_z']].plot(ax=ax)
            ax.set_xlabel("Samples")
            ax.set_ylabel("acc_x | acc_y | acc_z ")
            plt.legend()
            plt.title(f"{label} : {participant}")
            plt.show()


for label in labels:
    for participant in partipants:
        all_axis_df =  (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'").reset_index()
        )
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[['gyr_x', 'gyr_y', 'gyr_z']].plot(ax=ax)
            ax.set_xlabel("Samples")
            ax.set_ylabel("gyr_x | gyr_y | gyr_z ")
            plt.legend()
            plt.title(f"{label} : {participant}")
            plt.show()


# --------------------------------------------------------------
# Combine plots in one figure   
# --------------------------------------------------------------

label = 'row'
particpant = 'A'

combined_plot_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'").reset_index()
)
fig, ax  = plt.subplots(nrows=2, figsize=(20,10))
combined_plot_df[['acc_x', 'acc_y', 'acc_z']].plot(ax=ax[0])
combined_plot_df[['gyr_x', 'gyr_y', 'gyr_z']].plot(ax=ax[1])

ax[0].legend(loc="upper center", bbox_to_anchor= (0.5, 1.15), ncol=3, fancybox=True, shadow= True)
ax[0].legend(loc="upper center", bbox_to_anchor= (0.5, 1.15), ncol=3, fancybox=True, shadow= True)
ax[1].set_xlabel("samples")


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df['label'].unique()
participants = df['participant'].unique()

for label in labels:
    for participant in participants:
        combined_plot_df = (
        df.query(f"label == '{label}'")
        .query(f"participant == '{participant}'")
        .reset_index()
        )
        if len(combined_plot_df) > 0:
            
            fig, ax  = plt.subplots(nrows=2, figsize=(20,10))
            combined_plot_df[['acc_x', 'acc_y', 'acc_z']].plot(ax=ax[0])
            combined_plot_df[['gyr_x', 'gyr_y', 'gyr_z']].plot(ax=ax[1])

            ax[0].legend(loc="upper center", bbox_to_anchor= (0.5, 1.15), ncol=3, fancybox=True, shadow= True)
            ax[1].legend(loc="upper center", bbox_to_anchor= (0.5, 1.15), ncol=3, fancybox=True, shadow= True)
            ax[1].set_xlabel("samples")
            
            plt.savefig(f"../../reports/figures/{label.title()}  ({participant}).png")
            plt.show()


 