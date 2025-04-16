# GPR model with sea temp, wave power, period and height averages for each year - GenAI noted below
import numpy as np
import pandas as pd
import torch
import gpytorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import StandardScaler

# Load the data
ndvi_file = "D:/FYP/Data/updated_data.csv"
data = pd.read_csv(ndvi_file)

data['STRATCODE'] = data['STRATCODE'].fillna('')
data.fillna(data.mean(numeric_only=True), inplace=True)

features = data.drop(columns=['OBJECTID', 'STRATCODE', 'NDVI_2024H', 'NDVI_2024L'])
features = features.select_dtypes(include=['number'])
target = data['NDVI_2024H']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features = pd.DataFrame(features_scaled, index=features.index, columns=features.columns)

indices = data.index
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
#torches
X_train = torch.tensor(features.loc[train_idx].values, dtype=torch.float32)
X_test  = torch.tensor(features.loc[test_idx].values, dtype=torch.float32)
y_train = torch.tensor(target.loc[train_idx].values, dtype=torch.float32)
y_test  = torch.tensor(target.loc[test_idx].values, dtype=torch.float32)

test_STRATCODE = data.loc[test_idx, 'STRATCODE']

# sweet spot
num_inducing = 500
inducing_idx = np.random.choice(X_train.shape[0], size=num_inducing, replace=False)
inducing_points = X_train[inducing_idx]

# Define a sparse GP model using variational inference (GenAI assisted)
class SparseGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(SparseGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        # Use a ScaleKernel with an RBF kernel for smooth functions (GenAI to solve errors)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

model = SparseGPModel(inducing_points)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

likelihood.noise = 1e-2 #GenAI

model.train()
likelihood.train()

# Define optimizer and loss (Variational ELBO) (GenAI helped)
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.01)
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_train.size(0))

training_iterations = 100
for i in range(training_iterations):
    optimizer.zero_grad()
    # Use increased jitter (e.g., 1e-4) during the Cholesky decomposition (GenAI)
    with gpytorch.settings.cholesky_jitter(1e-4):
        output = model(X_train)
    loss = -mll(output, y_train)
    loss.backward()
    print(f'Iteration {i+1}/{training_iterations} - Loss: {loss.item()}')
    optimizer.step()

print("GPyTorch model training complete.")

model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-4):
    preds = likelihood(model(X_test))
    mean_preds = preds.mean
    lower, upper = preds.confidence_region()

# Print a sample of predictions along with actual NDVI_2024H values and STRATCODE for identification
print("\nFirst 10 Predictions on Test Set:")
for i in range(10):
    stratcode_val = test_STRATCODE.iloc[i]
    actual_val = y_test[i].item()
    pred_val = mean_preds[i].item()
    lower_val = lower[i].item()
    upper_val = upper[i].item()
    print(f"STRATCODE: {stratcode_val}, Actual NDVI_2024H: {actual_val:.4f}, Predicted: {pred_val:.4f}, 95% CI: [{lower_val:.4f}, {upper_val:.4f}]")

# Compute evaluation metrics
mean_preds_np = mean_preds.cpu().numpy()
y_test_np = y_test.cpu().numpy()
mse = mean_squared_error(y_test_np, mean_preds_np)
r2 = r2_score(y_test_np, mean_preds_np)
mae = mean_absolute_error(y_test_np, mean_preds_np)
explained_var = explained_variance_score(y_test_np, mean_preds_np)

print("\nEvaluation Metrics:")
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")
print(f"Mean Absolute Error: {mae}")
print(f"Explained Variance: {explained_var}")
