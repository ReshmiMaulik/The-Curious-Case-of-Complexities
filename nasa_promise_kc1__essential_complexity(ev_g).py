
!pip install git+https://github.com/microsoft/dowhy.git
import dowhy
from dowhy import CausalModel

import numpy as np
import pandas as pd
import graphviz
import networkx as nx

np.set_printoptions(precision=3, suppress=True)
np.random.seed(0)

"""**Utility Function**"""

def make_graph(adjacency_matrix, labels=None):
    idx = np.abs(adjacency_matrix) > 0.01
    dirs = np.where(idx)
    d = graphviz.Digraph(engine='dot')
    names = labels if labels else [f'x{i}' for i in range(len(adjacency_matrix))]
    for name in names:
        d.node(name)
    for to, from_, coef in zip(dirs[0], dirs[1], adjacency_matrix[idx]):
        d.edge(names[from_], names[to], label=str(coef))
    return d

def str_to_dot(string):
    '''
    Converts input string from graphviz library to valid DOT graph format.
    '''
    graph = string.strip().replace('\n', ';').replace('\t','')
    graph = graph[:9] + graph[10:-2] + graph[-1] # Removing unnecessary characters from string
    return graph

"""**Load Data**"""

# Import data from Github

url="https://raw.githubusercontent.com/ReshmiMaulik/NASA-promise-dataset-repository/main/kc1.csv"


#df = pd.read_csv(url, sep="\t", header= None)
df = pd.read_csv(url, sep=",")

df = pd.read_csv(url, sep=",")

df

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
df_scaled = ss.fit_transform(df)

df.head()

df.columns

"""Causal Discovery with causal-learn

We first try the PC algorithm with default parameters.
"""

from causallearn.search.ConstraintBased.PC import pc

labels = [f'{col}' for i, col in enumerate(df.columns)]
data = df_scaled

cg = pc(data)

# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io

pyd = GraphUtils.to_pydot(cg.G, labels=labels)
pyd.write_png('img_causal_PC_original.png')
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.axis('off')
plt.imshow(img)
plt.show()

"""GES algorithm"""

# For Cyclomatic complexity
df1 = df[['defects','lOCode', 'v(g)','d','i','e']]
df1['defects'] = df1['defects'].astype(int)
df1.rename(columns={'v(g)': 'v_g'}, inplace=True)

# For Essential complexity
df1 = df[['defects','lOCode', 'ev(g)','d','i','e']]
df1['defects'] = df1['defects'].astype(int)
df1.rename(columns={'ev(g)': 'ev_g'}, inplace=True)

# For Design complexity
df1 = df[['defects','lOCode', 'iv(g)','d','i','e']]
df1['defects'] = df1['defects'].astype(int)
df1.rename(columns={'iv(g)': 'iv_g'}, inplace=True)

#df1.rename(columns={'iv(g)': 'iv_g'}, inplace=True)
#df1.rename(columns={'ev(g)': 'ev_g'}, inplace=True)

"""defects: Target variable — number of known defects in the software module. Often binary (0 = no defect, 1 = defect), but can be count-based in some versions.
loc- Lines of Code — total number of lines in the module, including code, comments, and blanks.
loCode-Lines of Code (actual) — lines containing executable code only.
locomment-Lines of Comments — lines that contain comments/documentation.
v(g)- Cyclomatic Complexity — number of independent paths through the code. Higher values indicate more complex logic.

n: Halstead's length metric.
v: Halstead's volume metric. High v indicates large codebase size—may be harder to maintain or understand.
l: Halstead's program length estimate.
d: Halstead's difficulty metric. High d suggests the code is complex and potentially error-prone.
i: Halstead's intelligence content.
e: Halstead's effort metric. High e implies more cognitive effort is needed to write or understand the code.


from causallearn.search.ScoreBased.GES import ges

# default parameters
Record = ges(df1.values)

# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io

labels = [f'{col}' for i, col in enumerate(df1.columns)]
pyd = GraphUtils.to_pydot(Record['G'], labels=labels)
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.axis('off')
plt.imshow(img)
plt.show()

# or save the graph
pyd.write_png('kc1_GES_df1_DC.png')

import graphviz
!apt install libgraphviz-dev
!pip install pygraphviz

#Final graph for cyclomatic complexity
# Define the causal graph using dowhy's graph notation
# The graph is represented as a string in DOT format
causal_graph = """digraph {
    v_g -> e;
    v_g -> i;
    v_g -> defects;
    v_g -> d;
    lOCode -> d;
    d -> defects;
    v_g -> lOCode;
    e -> lOCode;
    i -> lOCode;
    i -> defects;

}"""

# Final graph for Essential Complexity
# Define the causal graph using dowhy's graph notation
# The graph is represented as a string in DOT format
causal_graph = """digraph {

    lOCode -> e;
    lOCode -> i;
    lOCode -> d;
    lOCode -> ev_g;
    d -> e;
    d -> ev_g;
    d -> defects ;
    e -> ev_g;
    i -> e;
    i -> ev_g;
    i -> defects;
    ev_g -> defects;
}"""

# Final graph for Design Complexity
# Define the causal graph using dowhy's graph notation
# The graph is represented as a string in DOT format
causal_graph = """digraph {

    lOCode -> e;
    lOCode -> i;
    lOCode -> d;
    lOCode -> iv_g;
    d -> e;
    d -> iv_g;
    d -> defects ;
    e -> iv_g;
    i -> e;
    i -> iv_g;
    i -> defects;

}"""


# For Cyclomatic complexity
model=CausalModel(
        data = df1,
        treatment='v_g',
        outcome='defects',
        graph=causal_graph,
        )
model.view_model(layout="dot")
model.view_model(file_name="causal_model_CC.png") # Save the plot to a file

# For Essential complexity
model=CausalModel(
        data = df1,
        treatment='e',
        outcome='defects',
        graph=causal_graph,
        )
model.view_model(layout="dot")
model.view_model(file_name="causal_model_final_EC") # Save the plot to a file

# For Design complexity
model=CausalModel(
        data = df1,
        treatment='iv_g',
        outcome='defects',
        graph=causal_graph,
        )
model.view_model(layout="dot")
model.view_model(file_name="causal_model_final_DC") # Save the plot to a file

import matplotlib.pyplot as plt
import seaborn as sns
# plot heatmap for feature variables
corr = df1.corr()
plt.figure(figsize=[12,10])
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,annot_kws={"size":15})
#plt.xticks(rotation=60)
import matplotlib.pyplot as plt
import seaborn as sns
# plot heatmap for feature variables
corr = df1.corr()
plt.figure(figsize=[12,10])
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,annot_kws={"size":15})
#plt.xticks(rotation=60)
#plt.title("Heatmap of Correlation Coefficient for Bug Feature Variables", size=10);
plt.xticks(rotation=45,size=15)
plt.yticks(rotation=0, size=15)
#plt.title("Heatmap of Correlation Coefficient for Bug Feature Variables", size=14);

plt.savefig('corrplot-kc1.png', bbox_inches='tight', pad_inches=0.0)

# Compute correlation matrix
corr_matrix = df1.corr()

# Save to CSV
corr_matrix.to_csv('correlation_matrix-kc1-DC.csv')

# Or display as a table
print(corr_matrix)

#kc1 only
 #Convert 'problems' column to numerical (1 for 'yes', 0 for 'no')
df['defects'] = df['defects'].map({'yes': 1, 'no': 0})

# Compute correlation matrix
corr_matrix = df.corr(numeric_only=True)

# Save to CSV
corr_matrix.to_csv('correlation_matrix-kc1.csv')

# Or display as a table
print(corr_matrix)

"""Identification

Halstead Metrics (n, v, l, d, i, e, b, t). These metrics are derived from Halstead's software science, which quantifies software complexity based on operators and operands in the code. They're commonly used in defect prediction models to estimate software quality and maintainability.

Strategic Applications
Defect Prediction: Use metrics like b, e, and d to train models that predict defect-prone modules.

Code Review Prioritization: Focus reviews on files with high v, d, or low lOComment."""


identified_estimand = model.identify_effect()
print(identified_estimand)

estimate= model.estimate_effect(
 identified_estimand,
 method_name='backdoor.linear_regression',
 confidence_intervals=True,
  test_significance=True
)

print(f'Estimate of causal effect: {estimate}')

"""### Estimand : 1 for i on defects
Estimand name: backdoor
Estimand expression:
 d                     
────(E[defects|lOCode])
d[i]                   
Estimand assumption 1, Unconfoundedness: If U→{i} and U→defects then P(defects|i,lOCode,U) = P(defects|i,lOCode)

## Realized estimand
b: defects~i+lOCode+i*d
Target units:

## Estimate
Mean value: 0.0035908464605419466
p-value: [0.]
95.0% confidence interval: (np.float64(0.0014848487485671436), np.float64(0.005288320451726725))
### Conditional Estimates
__categorical__d
(-0.001, 1.5]      0.003170
(1.5, 2.5]         0.003260
(2.5, 5.56]        0.003382
(5.56, 11.008]     0.003674
(11.008, 53.75]    0.004543

For Design Complexity
Estimand expression:
   d                   
───────(E[defects|d,i])
d[iv_g]                
Estimand assumption 1, Unconfoundedness: If U→{iv_g} and U→defects then P(defects|iv_g,d,i,U) = P(defects|iv_g,d,i)

## Realized estimand
b: defects~iv_g+d+i
Target units: ate

## Estimate
Mean value: -0.01322177964093027
p-value: [0.001]
95.0% confidence interval: [[-0.021 -0.006]]

# For Essential Complexity
### Estimand : 1
Estimand name: backdoor
Estimand expression:
   d                   
───────(E[defects|d,i])
d[ev_g]                
Estimand assumption 1, Unconfoundedness: If U→{ev_g} and U→defects then P(defects|ev_g,d,i,U) = P(defects|ev_g,d,i)

## Realized estimand
b: defects~ev_g+d+i
Target units: ate

## Estimate
Mean value: -0.02285483839510266
p-value: [0.]
95.0% confidence interval: [[-0.032 -0.014]]

### Estimand : 1
Estimand name: backdoor
Estimand expression:
  d               
──────(E[defects])
d[v_g]            
Estimand assumption 1, Unconfoundedness: If U→{v_g} and U→defects then P(defects|v_g,,U) = P(defects|v_g,)

## Realized estimand
b: defects~v_g
Target units: ate

## Estimate
Mean value: 0.02739818324879416
p-value: [0.]
95.0% confidence interval: [[0.024 0.031]]
"""

#Textual Interpreter
interpretation = estimate.interpret(method_name="textual_effect_interpreter")

"""Refutation"""

refute_results = model.refute_estimate(identified_estimand, estimate,
                                       method_name="random_common_cause")
print(refute_results)

"""Refute: Add a random common cause for Design Complexity
Estimated effect:-0.01322177964093027
New effect:-0.01323065072753803
p value:1.0

Refute: Add a random common cause for essential complexity
Estimated effect:-0.02285483839510266
New effect:-0.0228354615951731
p value:0.94

Refute: Add a random common cause for cyclomatic complexity
Estimated effect:0.02739818324879416
New effect:0.02739747508616771
p value:0.96
"""

refutel_common_cause=model.refute_estimate(identified_estimand,estimate,"data_subset_refuter")
print(refutel_common_cause)

"""Refute: Use a subset of data v_g
Estimated effect:0.008116542975618833
New effect:0.008093236071439011
p value:0.98
"""

refutation = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=100)
print(refutation)

"""Refute: Use a Placebo Treatment v_g
Estimated effect:0.008116542975618833
New effect:0.0005137592023183447
p value:0.8600000000000001

"""
