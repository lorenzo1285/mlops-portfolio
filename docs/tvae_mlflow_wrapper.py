import mlflow
import torch
import numpy as np
import pandas as pd

class GenerativeCrashClassifier(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc Wrapper for a 3-TVAE Pure Generative Classifier.
    Predicts crash severity by selecting the class whose TVAE yields the lowest reconstruction loss.
    """
    
    def load_context(self, context):
        """Loads the three trained TVAE models from the MLflow artifact context."""
        import joblib
        
        # Load the fitted CTGAN TVAE models
        self.tvae_pdo = joblib.load(context.artifacts["pdo_model"])
        self.tvae_injury = joblib.load(context.artifacts["injury_model"])
        self.tvae_fatal = joblib.load(context.artifacts["fatal_model"])
        
        self.models = {
            "PDO": self.tvae_pdo,
            "Injury": self.tvae_injury,
            "Fatal": self.tvae_fatal
        }

    def _calculate_reconstruction_loss(self, model, X_df):
        """Calculates the MSE reconstruction loss for a given TVAE model."""
        # 1. Transform raw dataframe using the model's internal Variational Gaussian Mixture transformer
        transformed_np = model._transformer.transform(X_df)
        
        # 2. Convert to PyTorch tensor and send to the correct device
        device = next(model._model.parameters()).device
        tensor_data = torch.tensor(transformed_np.to_numpy() if isinstance(transformed_np, pd.DataFrame) else transformed_np, dtype=torch.float32).to(device)
        
        # 3. Forward pass through the neural network
        model._model.eval() # Ensure we are in evaluation mode (turns off dropout)
        with torch.no_grad():
            recon, mu, logvar = model._model(tensor_data)
            
            # 4. Calculate Mean Squared Error across the feature dimension
            mse_loss = torch.mean((recon - tensor_data) ** 2, dim=1)
            
            # Note: We use purely MSE here for stability, but you could add KL Divergence
            # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            
        return mse_loss.cpu().numpy()

    def predict(self, context, model_input):
        """
        Runs the input through all three models and returns the class with the lowest loss.
        model_input: A pandas DataFrame containing new crash records.
        """
        # Calculate loss for each class
        losses = {}
        for class_name, model in self.models.items():
            losses[class_name] = self._calculate_reconstruction_loss(model, model_input)
            
        # Convert dictionary of arrays to a DataFrame: 
        # Columns = ['PDO', 'Injury', 'Fatal'], Rows = each crash record
        loss_df = pd.DataFrame(losses)
        
        # The predicted class is the column name with the minimum loss for each row
        predictions = loss_df.idxmin(axis=1).values
        
        return predictions
