import numpy as np
import pandas as pd
import torch
import gpytorch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

ndvi_file = "D:/FYP/Data/updated_data_with_tide_averages.csv"
data = pd.read_csv(ndvi_file)

data['STRATCODE'] = data['STRATCODE'].fillna('')

# Define NDVI columns (for 2016 to 2023, including H and L measurements)
ndvi_years = ['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
ndvi_cols = [f'NDVI_{year}H' for year in ndvi_years] + [f'NDVI_{year}L' for year in ndvi_years]

wave_years = list(range(2013, 2024))
wave_features = ['WaveHeight', 'WavePeriod', 'WavePower', 'SeaTemp']
wave_cols = [f'{feature}{year}' for year in wave_years for feature in wave_features]

# Combine all numeric feature columns
numeric_cols = ndvi_cols + wave_cols

#replace nulls with means
for col in numeric_cols:
    if col in data.columns:
        data[col] = data[col].fillna(data[col].mean())
    else:
        print(f"Warning: Column {col} not found in data.")

features_numeric = data[numeric_cols]
scaler = StandardScaler()
features_numeric_scaled = scaler.fit_transform(features_numeric)
features_numeric_scaled = pd.DataFrame(features_numeric_scaled,
                                       index=features_numeric.index,
                                       columns=features_numeric.columns)

features_categorical = pd.get_dummies(data['STRATCODE'], prefix='STRATCODE')
features = pd.concat([features_numeric_scaled, features_categorical], axis=1)
features = features.astype(np.float32)

#  NDVI_2024H as the target
target = data['NDVI_2024H']

indices = data.index
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_train = torch.tensor(features.loc[train_idx].values, dtype=torch.float32)
X_test  = torch.tensor(features.loc[test_idx].values, dtype=torch.float32)
y_train = torch.tensor(target.loc[train_idx].values, dtype=torch.float32)
y_test  = torch.tensor(target.loc[test_idx].values, dtype=torch.float32)

test_STRATCODE = data.loc[test_idx, 'STRATCODE']


# Custom White Noise Kernel for additional stability (GenAI - used to solve errors I was having)
class WhiteNoiseKernel(gpytorch.kernels.Kernel):
    def __init__(self, noise=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.noise = noise

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            return self.noise * torch.ones(x1.size(0))
        else:
            return self.noise * torch.zeros(x1.size(0), x2.size(0))

# Sparse GP Model using variational inference (GenAI)
class SparseGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        # Combine an RBF kernel with the custom White Noise kernel (GenAI)
        base_kernel = gpytorch.kernels.RBFKernel() + WhiteNoiseKernel(noise=1e-4)
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        self.mean_module = gpytorch.means.ConstantMean()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# number of points used, i found 500 was the sweet spot with accuracy vs computation time
num_inducing = 500
inducing_idx = np.random.choice(X_train.shape[0], size=num_inducing, replace=False)
inducing_points = X_train[inducing_idx]

# Instantiate model and likelihood
model = SparseGPModel(inducing_points)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.01)

mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_train.size(0))

training_iterations = 200
for i in range(training_iterations):
    optimizer.zero_grad()
    with gpytorch.settings.cholesky_jitter(1e-3):
        output = model(X_train)
        loss = -mll(output, y_train)
    loss.backward()
    print(f"Iteration {i+1}/{training_iterations} - Loss: {loss.item():.4f}")
    optimizer.step()

print("GPyTorch model training complete.")

model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    preds = likelihood(model(X_test))
    mean_preds = preds.mean
    lower, upper = preds.confidence_region()

print("First 10 Predictions on Test Set:")
for i in range(10):
    stratcode_val = test_STRATCODE.iloc[i]
    actual_val = y_test[i].item()
    pred_val = mean_preds[i].item()
    lower_val = lower[i].item()
    upper_val = upper[i].item()
    print(f"STRATCODE: {stratcode_val}, Actual NDVI_2024H: {actual_val:.4f}, "
          f"Predicted: {pred_val:.4f}, 95% CI: [{lower_val:.4f}, {upper_val:.4f}]")


mean_preds_np = mean_preds.cpu().numpy()
y_test_np = y_test.cpu().numpy()

mse = mean_squared_error(y_test_np, mean_preds_np)
r2 = r2_score(y_test_np, mean_preds_np)
mae = mean_absolute_error(y_test_np, mean_preds_np)
explained_var = explained_variance_score(y_test_np, mean_preds_np)

print("Evaluation Metrics:")
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")
print(f"Mean Absolute Error: {mae}")
print(f"Explained Variance: {explained_var}")
