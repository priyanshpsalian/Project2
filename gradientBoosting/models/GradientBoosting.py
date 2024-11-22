import numpy as np
from ..models.DecisionTree import DecisionTree


class GradientBoostingRegressor:
    # Setting up a Gradient Boosting Regressor with key configuration parameters.
    def __init__(self, no_of_estimators=1000, rate_of_learning=0.2, max_depth=2, sample_minimum_split=2,
                 subsample=1.0, criterion='friedman_mse', minimum_samples_per_leaf=1,
                 weight_minimum_leaf_fraction=0.0, least_impurity_reduce=0.0, init=None, random_state=None,
                 how_many_features=None, verbose=0, greatest_node_of_leaf=None,
                 criteria_for_early_stopping=0.1, count_of_max_iteration=None, tol=0.0001, alpha_parameter=0.0):
        # It incorporates user-specified values or fallback defaults to define the model's operational characteristics.
        self.no_of_estimators = no_of_estimators
        self.rate_of_learning = rate_of_learning
        self.max_depth = max_depth
        self.sample_minimum_split = sample_minimum_split
        self.subsample = subsample
        self.criterion = criterion
        self.minimum_samples_per_leaf = minimum_samples_per_leaf
        self.weight_minimum_leaf_fraction = weight_minimum_leaf_fraction
        self.least_impurity_reduce = least_impurity_reduce
        self.init = init
        self.random_state = random_state
        self.how_many_features = how_many_features
        self.verbose = verbose
        self.greatest_node_of_leaf = greatest_node_of_leaf
        self.criteria_for_early_stopping = criteria_for_early_stopping
        self.count_of_max_iteration = count_of_max_iteration
        self.tol = tol
        self.alpha_parameter = alpha_parameter
        self.models = []  # Stores the trained trees in the ensemble
        self.initial_prediction = None  # Baseline prediction value
        self.best_iteration = None  # Tracks optimal iteration for early stopping

    # Configures random number generator with provided seed for reproducibility
    def _set_random_state(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)

    # Implements sub-sampling logic for data when subsample ratio is less than 1
    def _subsample_data(self, X, y):
        X = np.array(X)
        if self.subsample < 1.0:
            n_samples = int(self.subsample * X.shape[0])
            indices = np.random.choice(X.shape[0], n_samples, replace=False)
            return X[indices], y[indices]
        return X, y

    # Method to train the ensemble model
    def fit(self, X, y):
        # Set up reproducibility for random operations
        self._set_random_state()

        # Initialize model predictions with mean of the target variable or user-provided initial value
        if self.init is None:
            self.initial_prediction = np.mean(y)
        else:
            self.initial_prediction = self.init

        # Create a baseline prediction array initialized to the starting value
        predictions = np.full_like(y, self.initial_prediction)

        # Split data into training and validation sets if early stopping is enabled
        if self.count_of_max_iteration:
            n_val_samples = int(len(X) * self.criteria_for_early_stopping)
            X_train, X_val = X[:-n_val_samples], X[-n_val_samples:]
            y_train, y_val = y[:-n_val_samples], y[-n_val_samples:]
            best_score = float('inf')  # Initialize best validation loss
            no_change_count = 0  # Counter for tracking convergence
        else:
            X_train, y_train = X, y

        # Iterate over the number of estimators to sequentially build the ensemble
        for i in range(self.no_of_estimators):
            # Compute residuals as the negative gradient of loss function (here, squared error)
            residuals = y_train - predictions[:len(y_train)]

            # Perform sub-sampling if configured
            X_sample, residuals_sample = self._subsample_data(X_train, residuals)

            # Train a single decision tree on the sampled data
            tree = DecisionTree(max_depth=self.max_depth, sample_minimum_split=self.sample_minimum_split,
                                criterion=self.criterion, minimum_samples_per_leaf=self.minimum_samples_per_leaf,
                                weight_minimum_leaf_fraction=self.weight_minimum_leaf_fraction,
                                least_impurity_reduce=self.least_impurity_reduce,
                                how_many_features=self.how_many_features, greatest_node_of_leaf=self.greatest_node_of_leaf,
                                alpha_parameter=self.alpha_parameter)
            tree.fit(X_sample, residuals_sample)
            self.models.append(tree)  # Append the trained tree to the model list

            # Update training predictions using the newly added tree
            predictions[:len(y_train)] += self.rate_of_learning * tree.predict(X_train)

            # Early stopping mechanism based on validation loss improvement
            if self.count_of_max_iteration:
                val_residuals = y_val - predictions[len(y_train):]
                val_loss = np.mean(np.square(val_residuals))  # Compute Mean Squared Error
                if val_loss + self.tol < best_score:  # Check improvement tolerance
                    best_score = val_loss
                    no_change_count = 0  # Reset counter if validation improves
                    self.best_iteration = i
                else:
                    no_change_count += 1  # Increment counter when no improvement
                    if no_change_count >= self.count_of_max_iteration:
                        if self.verbose:
                            print(f"Early stopping at iteration {i}")  # Log stopping
                        break

    # Predicts target values for new data points using the trained ensemble
    def predict(self, X):
        X = np.array(X)
        predictions = np.full(X.shape[0], self.initial_prediction)  # Initialize predictions
        no_of_estimators = self.best_iteration + 1 if self.best_iteration else self.no_of_estimators
        for tree in self.models[:no_of_estimators]:  # Iterate only up to the best iteration if early stopping was used
            predictions += self.rate_of_learning * tree.predict(X)
        return predictions
