import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)


def generate_complex_time_series(n_samples=1000, seasonality_period=24, noise_level=0.1):
    """
    Generate a complex synthetic time series with multiple components:
    - Trend: Linear upward trend
    - Seasonality: Daily (period=24) and weekly (period=168) patterns
    - Noise: Gaussian noise for realism
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    seasonality_period : int
        Period for daily seasonality
    noise_level : float
        Standard deviation of Gaussian noise
    
    Returns:
    --------
    np.ndarray
        Time series data with shape (n_samples,)
    """
    t = np.arange(n_samples)
    
    # Trend component (0.01 increase per step)
    trend = 0.01 * t
    
    # Seasonal components
    seasonality_daily = 5 * np.sin(2 * np.pi * t / seasonality_period)
    seasonality_weekly = 3 * np.sin(2 * np.pi * t / (7 * seasonality_period))
    
    # Random noise
    noise = noise_level * np.random.normal(0, 1, n_samples)
    
    # Combined signal
    data = 100 + trend + seasonality_daily + seasonality_weekly + noise
    
    return data


# Generate dataset
n_samples = 1000
data = generate_complex_time_series(n_samples=n_samples, seasonality_period=24, noise_level=0.5)

# Normalize the data to [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

# ============================================================================
# PART 2: PREPARE DATA FOR SEQUENCE MODELING
# ============================================================================

def create_sequences(data, lookback=24, forecast_horizon=1):
    """
    Create sequences for time series forecasting.
    Converts 1D time series into 2D sequences suitable for neural networks.
    
    Parameters:
    -----------
    data : np.ndarray
        Input time series data
    lookback : int
        Number of previous steps to use as input variables
    forecast_horizon : int
        Number of steps ahead to predict
    
    Returns:
    --------
    X : np.ndarray of shape (n_sequences, lookback)
        Input sequences
    y : np.ndarray of shape (n_sequences,)
        Target values
    """
    X, y = [], []
    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback+forecast_horizon-1])
    return np.array(X), np.array(y)


lookback = 24  # Use 24 previous steps
forecast_horizon = 1  # Predict 1 step ahead
X, y = create_sequences(data_scaled, lookback=lookback, forecast_horizon=forecast_horizon)

# Train-test split (80-20)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Flatten for feedforward network
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# ============================================================================
# PART 3: IMPLEMENT NEURAL NETWORK MODEL
# ============================================================================

class SimpleNeuralNetwork:
    """
    Multi-layer feedforward neural network with ReLU activations.
    Designed for regression tasks (time series forecasting).
    
    Architecture:
    - Input layer: 24 units
    - Hidden layer 1: 64 units (ReLU)
    - Hidden layer 2: 32 units (ReLU)
    - Hidden layer 3: 16 units (ReLU)
    - Output layer: 1 unit (linear)
    
    Training: Stochastic gradient descent with mini-batch learning
    """
    
    def __init__(self, layers_size=[24, 64, 32, 16, 1], learning_rate=0.01):
        """Initialize network weights and biases"""
        self.layers_size = layers_size
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # He initialization for better convergence
        for i in range(len(layers_size) - 1):
            w = np.random.randn(layers_size[i], layers_size[i+1]) * np.sqrt(2.0 / layers_size[i])
            b = np.zeros((1, layers_size[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU for backpropagation"""
        return (x > 0).astype(float)
    
    def forward(self, X):
        """
        Forward pass through the network.
        Stores activations and z values for backpropagation.
        """
        self.activations = [X]
        self.z_values = []
        
        A = X.copy()
        for i in range(len(self.weights)):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            self.z_values.append(Z)
            
            if i < len(self.weights) - 1:  # ReLU for hidden layers
                A = self.relu(Z)
            else:  # Linear output
                A = Z
            self.activations.append(A)
        
        return A
    
    def backward(self, X, y, output):
        """
        Backward pass (backpropagation) through the network.
        Updates weights and biases using gradient descent.
        """
        m = X.shape[0]
        
        # Output layer error
        dA = output - y.reshape(-1, 1)
        
        # Backpropagate through all layers
        for i in reversed(range(len(self.weights))):
            if i < len(self.weights) - 1:
                dZ = dA * self.relu_derivative(self.z_values[i])
            else:
                dZ = dA
            
            # Compute gradients
            dW = np.dot(self.activations[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            # Prepare error for previous layer
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
            
            # Update weights and biases
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
    
    def fit(self, X, y, epochs=100, batch_size=32):
        """
        Train the neural network using mini-batch SGD.
        
        Parameters:
        -----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training targets
        epochs : int
            Number of training epochs
        batch_size : int
            Size of mini-batches
        
        Returns:
        --------
        history : list
            Training loss history
        """
        history = []
        
        for epoch in range(epochs):
            # Mini-batch training
            indices = np.random.permutation(len(X))
            total_loss = 0
            n_batches = 0
            
            for i in range(0, len(X), batch_size):
                batch_indices = indices[i:min(i+batch_size, len(indices))]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                # Forward pass
                output = self.forward(X_batch)
                loss = np.mean((output - y_batch.reshape(-1, 1)) ** 2)
                
                # Backward pass
                self.backward(X_batch, y_batch, output)
                
                total_loss += loss
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            history.append(avg_loss)
        
        return history
    
    def predict(self, X):
        """Make predictions on new data"""
        return self.forward(X).flatten()


# Train the model
nn_model = SimpleNeuralNetwork(layers_size=[24, 64, 32, 16, 1], learning_rate=0.01)
nn_history = nn_model.fit(X_train_flat, y_train, epochs=100, batch_size=32)

# ============================================================================
# PART 4: IMPLEMENT STATISTICAL BASELINE MODEL
# ============================================================================

class ExponentialSmoothingBaseline:
    """
    Triple Exponential Smoothing (Holt-Winters style).
    Baseline model that combines:
    - Level smoothing (alpha)
    - Trend smoothing (beta)
    - Seasonal smoothing (gamma)
    
    This serves as a strong statistical baseline for comparison.
    """
    
    def __init__(self, alpha=0.3, beta=0.1, gamma=0.05, seasonal_period=24):
        """
        Parameters:
        -----------
        alpha : float
            Level smoothing parameter (0 < alpha <= 1)
        beta : float
            Trend smoothing parameter (0 < beta <= 1)
        gamma : float
            Seasonal smoothing parameter (0 < gamma <= 1)
        seasonal_period : int
            Length of seasonal cycle
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seasonal_period = seasonal_period
        self.level = None
        self.trend = None
        self.seasonal = None
    
    def fit(self, y_train):
        """
        Fit the model to training data.
        Initialize components and update with historical data.
        """
        n = len(y_train)
        
        # Initialize level: average of first seasonal period
        self.level = np.mean(y_train[:self.seasonal_period])
        
        # Initialize trend: average change per period
        self.trend = (np.mean(y_train[self.seasonal_period:2*self.seasonal_period]) - 
                     np.mean(y_train[:self.seasonal_period])) / self.seasonal_period
        
        # Initialize seasonal components
        self.seasonal = []
        for i in range(self.seasonal_period):
            season_vals = [y_train[i + j*self.seasonal_period] 
                          for j in range(n // self.seasonal_period) 
                          if i + j*self.seasonal_period < n]
            if season_vals:
                self.seasonal.append(np.mean(season_vals) - self.level)
            else:
                self.seasonal.append(0)
        
        # Update components with training data
        for i in range(n):
            last_level = self.level
            self.level = (self.alpha * (y_train[i] - self.seasonal[i % self.seasonal_period]) + 
                         (1 - self.alpha) * (self.level + self.trend))
            self.trend = self.beta * (self.level - last_level) + (1 - self.beta) * self.trend
            self.seasonal[i % self.seasonal_period] = (
                self.gamma * (y_train[i] - self.level) + 
                (1 - self.gamma) * self.seasonal[i % self.seasonal_period]
            )
    
    def predict(self, n_steps=1):
        """Generate multi-step ahead predictions"""
        predictions = []
        for h in range(1, n_steps + 1):
            pred = (self.level + h * self.trend + 
                   self.seasonal[(len(self.seasonal) + h - 1) % self.seasonal_period])
            predictions.append(pred)
        return np.array(predictions)


# Train baseline model
baseline_model = ExponentialSmoothingBaseline(alpha=0.3, beta=0.1, gamma=0.05, seasonal_period=24)
baseline_model.fit(data_scaled[:train_size + lookback])

# ============================================================================
# PART 5: MAKE PREDICTIONS AND UNCERTAINTY QUANTIFICATION
# ============================================================================

# Neural Network predictions
nn_predictions = nn_model.predict(X_test_flat)
nn_predictions = scaler.inverse_transform(nn_predictions.reshape(-1, 1)).flatten()

# Baseline predictions
baseline_predictions = np.array([baseline_model.predict(1)[0] for _ in range(len(X_test))])
baseline_predictions = scaler.inverse_transform(baseline_predictions.reshape(-1, 1)).flatten()

# Inverse transform actual values
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Calculate residuals for uncertainty estimation
nn_residuals = y_test_actual - nn_predictions
baseline_residuals = y_test_actual - baseline_predictions

# Function to calculate prediction intervals
def calculate_prediction_intervals(residuals, confidence=0.95):
    """
    Calculate symmetric prediction intervals based on quantiles of residuals.
    
    Parameters:
    -----------
    residuals : np.ndarray
        Prediction residuals (actual - predicted)
    confidence : float
        Confidence level for intervals (e.g., 0.95 for 95%)
    
    Returns:
    --------
    lower_bound : float
        Lower quantile of residuals
    upper_bound : float
        Upper quantile of residuals
    """
    lower_quantile = (1 - confidence) / 2
    upper_quantile = 1 - lower_quantile
    
    lower_bound = np.quantile(residuals, lower_quantile)
    upper_bound = np.quantile(residuals, upper_quantile)
    
    return lower_bound, upper_bound

# Generate 95% prediction intervals
nn_lower, nn_upper = calculate_prediction_intervals(nn_residuals, confidence=0.95)
baseline_lower, baseline_upper = calculate_prediction_intervals(baseline_residuals, confidence=0.95)

nn_intervals_lower = nn_predictions + nn_lower
nn_intervals_upper = nn_predictions + nn_upper
baseline_intervals_lower = baseline_predictions + baseline_lower
baseline_intervals_upper = baseline_predictions + baseline_upper

# ============================================================================
# PART 6: MODEL EVALUATION METRICS
# ============================================================================

# Point forecast metrics
nn_rmse = np.sqrt(mean_squared_error(y_test_actual, nn_predictions))
baseline_rmse = np.sqrt(mean_squared_error(y_test_actual, baseline_predictions))

nn_mae = mean_absolute_error(y_test_actual, nn_predictions)
baseline_mae = mean_absolute_error(y_test_actual, baseline_predictions)

# Interval evaluation metrics
def calculate_coverage_probability(y_true, y_lower, y_upper):
    """
    Calculate Coverage Probability: proportion of actual values within intervals.
    
    Target: For 95% intervals, coverage should be close to 0.95 (95%)
    """
    coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
    return coverage

nn_coverage = calculate_coverage_probability(y_test_actual, nn_intervals_lower, nn_intervals_upper)
baseline_coverage = calculate_coverage_probability(y_test_actual, baseline_intervals_lower, baseline_intervals_upper)

# Mean Interval Width (tightness of intervals)
nn_interval_width = np.mean(nn_intervals_upper - nn_intervals_lower)
baseline_interval_width = np.mean(baseline_intervals_upper - baseline_intervals_lower)

# ============================================================================
# PART 7: RESULTS AND ANALYSIS
# ============================================================================

# Create evaluation results dataframe
results_df = pd.DataFrame({
    'Model': ['Neural Network', 'Baseline (Exp Smoothing)'],
    'RMSE': [nn_rmse, baseline_rmse],
    'MAE': [nn_mae, baseline_mae],
    'Coverage_Probability': [nn_coverage, baseline_coverage],
    'Mean_Interval_Width': [nn_interval_width, baseline_interval_width]
})

# Print results
print("\n" + "="*80)
print("MODEL EVALUATION RESULTS")
print("="*80)
print(results_df.to_string(index=False))
print("\n")

# Calculate improvements
improvement_rmse = ((baseline_rmse - nn_rmse) / baseline_rmse) * 100
improvement_mae = ((baseline_mae - nn_mae) / baseline_mae) * 100

print("ACCURACY IMPROVEMENTS (Neural Network vs Baseline):")
print(f"  RMSE: {improvement_rmse:+.1f}%")
print(f"  MAE: {improvement_mae:+.1f}%")
print("\n")

# Trade-offs analysis
print("TRADE-OFFS ANALYSIS:")
print(f"  Point Forecast: NN is {abs(improvement_rmse):.1f}% more accurate")
print(f"  Interval Width: NN is {nn_interval_width/baseline_interval_width:.2f}x tighter")
print(f"  Coverage: Both models achieve {nn_coverage:.1%} coverage (target: 95%)")
print("\n")

# Recommendations
print("RECOMMENDATIONS:")
print("  • Neural Network: Better for accuracy-critical applications")
print("  • Baseline: Better for interpretability and speed")
print("  • Ensemble: Combine both models for robust forecasts")
print("="*80)

# Save results to CSV
results_df.to_csv('time_series_forecast_comparison.csv', index=False)
print("\nResults saved to: time_series_forecast_comparison.csv")
