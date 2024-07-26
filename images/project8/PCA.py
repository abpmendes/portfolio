from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters
PC = 3  # Number of principal components
file_path = 'Data.xlsx'

# Import data
df = pd.read_excel(file_path, index_col=0, header=0)

# Helper function to ensure at least one variable from each group is retained
def ensure_variables_per_group(loadings_df, variables_to_keep, desired_prefixes):
    variables_per_group = {prefix: [] for prefix in desired_prefixes}
    for var in variables_to_keep:
        prefix = var[0]  # Assumes the group is determined by the first letter
        if prefix in variables_per_group:
            variables_per_group[prefix].append(var)

    # Ensure there is at least one variable from each group
    for prefix, vars_group in variables_per_group.items():
        if len(vars_group) == 0:
            all_vars_group = [col for col in loadings_df.index if col.startswith(prefix)]
            var_with_highest_loading = loadings_df.loc[all_vars_group].abs().max(axis=1).idxmax()
            variables_to_keep.append(var_with_highest_loading)

    return list(set(variables_to_keep))  # Remove duplicates and convert back to list

# Data standardization
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# PCA
pca = PCA(n_components=PC)
principal_components = pca.fit_transform(data_scaled)

# Create a new DataFrame for principal components
pca_df_3 = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
pca_df_3.to_excel('PCA_Data_3.xlsx', index=False)

# Explained variance
print("Explained variance by each component:", pca.explained_variance_ratio_)
print("Total explained variance:", sum(pca.explained_variance_ratio_))

# Calculating component loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Creating a DataFrame for component loadings
loadings_df_3 = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3'], index=df.columns)
loadings_df_3.to_excel('PCA_Loadings_3.xlsx')

# Variable selection criteria
min_explained_variance = 0.8
min_loading = 0.4
correlation_threshold = 0.8

# Calculate cumulative variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
selected_components = np.argmax(cumulative_variance >= min_explained_variance) + 1

# Select variables with high loadings on the selected components
variables_to_keep = []
for i in range(selected_components):
    variables_to_keep.extend(loadings_df_3.index[abs(loadings_df_3.iloc[:, i]) >= min_loading].tolist())

variables_to_keep = ensure_variables_per_group(loadings_df_3, variables_to_keep, ['A', 'F', 'I', 'T', 'H'])
print(f"Variables to keep: {variables_to_keep}")

# Calculate explained variance for different numbers of components
variances = []
for n in range(1, 16):
    pca = PCA(n_components=n)
    pca.fit(data_scaled)
    variances.append(pca.explained_variance_ratio_[-1])

plt.figure(figsize=(8, 4))
plt.plot(range(1, 16), variances, 'bo-', markersize=8, linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance per Component')
plt.xticks(range(1, 16))
plt.yticks(np.around(np.linspace(0, max(variances), 10), decimals=2))
plt.grid(True, linestyle='--', color='lightgrey', alpha=0.7)
plt.savefig('screeplot.png', format='png', dpi=300, bbox_inches='tight')

# Create Biplot
plt.figure(figsize=(10, 8))
for i, (loading, var_name) in enumerate(zip(loadings, df.columns)):
    plt.arrow(0, 0, loading[0], loading[1], color='r', alpha=0.5)
    plt.text(loading[0], loading[1], var_name, color='black', ha='center', va='center', fontsize=9)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')
plt.xlim(min(loadings[:, 0]) * 1.1, max(loadings[:, 0]) * 1.1)
plt.ylim(min(loadings[:, 1]) * 1.1, max(loadings[:, 1]) * 1.1)
plt.axhline(0, color='grey', lw=1, linestyle='--')
plt.axvline(0, color='grey', lw=1, linestyle='--')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Component Loadings for PC1 and PC2')
plt.savefig('biplot.png', format='png', dpi=300, bbox_inches='tight')

# Create bar plot for component loadings
fig, ax = plt.subplots(figsize=(12, 6))
positions = np.arange(loadings.shape[0])
width = 0.25
ax.bar(positions - width, loadings[:, 0], width, label='PC1')
ax.bar(positions, loadings[:, 1], width, label='PC2')
ax.bar(positions + width, loadings[:, 2], width, label='PC3')
ax.set_xticks(positions)
ax.set_xticklabels(df.columns, rotation=90)
ax.set_title('Principal Component Loadings (PC1, PC2, PC3)')
ax.set_xlabel('Variables')
ax.set_ylabel('Component Loadings')
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.savefig('barplot.png', format='png', dpi=300, bbox_inches='tight')

# Biplot with different colors for variables to keep and remove
plt.figure(figsize=(10, 8))
for i, (loading, var_name) in enumerate(zip(loadings, df.columns)):
    color = 'b' if var_name in variables_to_keep else 'r'
    plt.arrow(0, 0, loading[0], loading[1], color=color, alpha=0.5)
    plt.text(loading[0], loading[1], var_name, color='black', ha='center', va='center', fontsize=9)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')
plt.xlim(min(loadings[:, 0]) * 1.1, max(loadings[:, 0]) * 1.1)
plt.ylim(min(loadings[:, 1]) * 1.1, max(loadings[:, 1]) * 1.1)
plt.axhline(0, color='grey', lw=1, linestyle='--')
plt.axvline(0, color='grey', lw=1, linestyle='--')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Component Loadings for PC1 and PC2')
plt.plot([], [], 'bo', label='Variables to Keep')
plt.plot([], [], 'ro', label='Variables to Remove')
plt.legend()
plt.savefig('biplot1.png', format='png', dpi=300, bbox_inches='tight')
plt.show()
