import numpy as np
import pandas as pd
import torch
import gpytorch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score


#Custom White Noise Kernel suggested from GenAI to solve errors i was experiencing
# This simple kernel returns a fixed noise value on the diagonal and 0 off-diagonally.
class WhiteNoiseKernel(gpytorch.kernels.Kernel):
    def __init__(self, noise=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.noise = noise

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            return self.noise * torch.ones(x1.size(0), device=x1.device)
        else:
            # White noise is zero off-diagonally.
            return self.noise * torch.zeros(x1.size(0), x2.size(0), device=x1.device)


ndvi_file = "D:/FYP/Data/updated_data_with_tide_averages.csv"
data = pd.read_csv(ndvi_file)

data = data.drop(columns=['FID'])

# nulls replaced with averages
numeric_cols = data.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].mean())

# ndvi24 used as target
features = data.drop(columns=['OBJECTID', 'NDVI_2024H', 'NDVI_2024L'])
target = data['NDVI_2024H']

# Split data into training and testing sets
indices = data.index
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_train_np = features.loc[train_idx].values
X_test_np  = features.loc[test_idx].values
y_train_np = target.loc[train_idx].values
y_test_np  = target.loc[test_idx].values

# standardise the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled  = scaler.transform(X_test_np)

# Convert standardised data to torch tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test  = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)
y_test  = torch.tensor(y_test_np, dtype=torch.float32)

# number of inducing points
num_inducing = 500
inducing_idx = np.random.choice(X_train.shape[0], size=num_inducing, replace=False)
inducing_points = X_train[inducing_idx]

class SparseGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(SparseGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        # Combine RBF with the custom WhiteNoiseKernel (from GenAI)
        base_kernel = gpytorch.kernels.RBFKernel() + WhiteNoiseKernel(noise=1e-4)
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Instantiate model
model = SparseGPModel(inducing_points)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

model.train()
likelihood.train()

optimiser = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.01)

mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_train.size(0))

training_iterations = 100  #same as gpr.py
for i in range(training_iterations):
    optimiser.zero_grad()
    with gpytorch.settings.cholesky_jitter(1e-3):
        output = model(X_train)
        loss = -mll(output, y_train)
    loss.backward()
    print(f"Iteration {i+1}/{training_iterations} - Loss: {loss.item():.4f}")
    optimiser.step()

print("\nGPyTorch model training complete.")


model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    preds = likelihood(model(X_test))
    mean_preds = preds.mean
    lower, upper = preds.confidence_region()

print("\nFirst 10 Predictions on Test Set:")
for i in range(10):
    actual_val = y_test[i].item()
    pred_val = mean_preds[i].item()
    lower_val = lower[i].item()
    upper_val = upper[i].item()
    print(f"Actual NDVI_2024H: {actual_val:.4f}, Predicted: {pred_val:.4f}, 95% CI: [{lower_val:.4f}, {upper_val:.4f}]")


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
