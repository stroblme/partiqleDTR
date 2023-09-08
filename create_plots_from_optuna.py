
import optuna
from optuna.trial import TrialState
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np

path = "studies/partiqledtr.db"
name = "OptunaOptimization#017"

study = optuna.load_study(  study_name=name,
                            storage=f"sqlite:///{path}"
)



qLearningRate = []
learningRate = []

objective_distance = {'obj_value':[], 'distance':[]}

for trial in study.trials:
    if trial.state == TrialState.PRUNED or trial.state == TrialState.FAIL:
        continue
    

    learningRate.append(trial.params['learning_rate'])
    qLearningRate.append(trial.params['quantum_learning_rate'])

    objective_distance['obj_value'].append(trial.value)
    objective_distance['distance'].append(np.abs(trial.params['learning_rate'] - trial.params['quantum_learning_rate']))

# qLearningRate = pd.DataFrame(qLearningRate)
# learningRate = pd.DataFrame(learningRate)

lr_combined =pd.DataFrame(dict(
    series=np.concatenate((["classical"]*len(learningRate), ["quantum"]*len(qLearningRate))), 
    learning_rate  =np.concatenate((learningRate,qLearningRate))
))

# fig = px.histogram(lr_combined, x="learning_rate", color="series", barmode="overlay", nbins=40, marginal="violin", range_x=[0,0.04], color_discrete_sequence=px.colors.qualitative.Dark2)

# fig.update_layout(
#     title_text='Sampling Distribution Learning Rate', # title of plot
#     xaxis_title_text='Learning Rate', # xaxis label
#     yaxis_title_text='Count', # yaxis label,
#     bargap=0.2, # gap between bars of adjacent location coordinates
#     bargroupgap=0.1 # gap between bars of the same location coordinates
# )

# fig.show()

# pass

# del fig

fig = px.density_heatmap(pd.DataFrame(objective_distance), x="obj_value", y="distance", nbinsx=10, nbinsy=10)

fig.update_layout(
    title_text='Distribution Objective Value against Learning Rate Distance', # title of plot
    xaxis_title_text='Objective Value', # xaxis label
    yaxis_title_text='Learning Rate Distance', # yaxis label,
)

# fig = px.histogram(x=objective_distance['obj_value'], y=objective_distance['distance'])

fig.show()