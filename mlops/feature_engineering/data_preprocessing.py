import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class DataPreprocessing:
    """
    Class for performing data cleaning, normalization, and feature extraction (PCA).
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def clean_data(self):
        """
        Method for cleaning the data.
        - Drop the first row if it contains metadata or description.
        - Rename columns based on provided considerations
        - Drop 'ID' column
        - Convert to appropriate data types (categorical, numerical)
        """
        # Drop the first row if it's a description of the data (metadata)
        self.df = self.df.drop(index=0)

        # Rename columns
        self.df.columns = ['ID', 'MV', 'SR', 'NR', 'TR', 'VR', 'SUR1', 'SUR2', 'SUR3', 'UR', 'FR',
                           'OR', 'RR', 'BR', 'MR', 'CR', 'Green frogs', 'Brown frogs',
                           'Common toad', 'Fire-bellied toad', 'Tree frog',
                           'Common newt', 'Great crested newt']

        # Drop the 'ID' column
        self.df = self.df.drop(['ID'], axis=1)

        # Convert numerical columns
        numerical_columns = ['SR', 'NR', 'OR', 'RR', 'BR']
        for col in numerical_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Convert categorical columns
        categorical_columns = ['MV', 'TR', 'VR', 'SUR1', 'SUR2', 'SUR3', 'UR', 'FR', 'MR', 'CR',
                               'Green frogs', 'Brown frogs', 'Common toad', 'Fire-bellied toad',
                               'Tree frog', 'Common newt', 'Great crested newt']
        for col in categorical_columns:
            self.df[col] = self.df[col].astype('category')

        return self.df

    def normalize_data(self):
        """
        Method for normalizing the data using StandardScaler.
        """
        # Select all features for normalization except for the categorical ones
        x = self.df.select_dtypes(include=['float64', 'int64']).values  # Select numerical values
        x = StandardScaler().fit_transform(x)  # Standardize the data

        # Create normalized DataFrame with renamed feature columns
        feat_cols = ['feature' + str(i) for i in range(x.shape[1])]
        normalised_df = pd.DataFrame(x, columns=feat_cols)

        return normalised_df

    def apply_pca(self, X):
        """
        Apply PCA to reduce the dimensionality of the features.
        :param X: DataFrame of features
        """

        n_components = min(10, X.shape[0], X.shape[1])  # Min of 10, number of samples, or number of features

        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(X)

        return principal_components

    def preprocess(self):
        """
        Full preprocessing pipeline including:
        - Data cleaning (renaming columns, dropping columns, handling types)
        - Normalization (scaling features)
        - Feature extraction (PCA)
        """
        # Step 1: Clean the data
        self.clean_data()

        # Step 2: Normalize the data
        normalized_df = self.normalize_data()

        # Step 3: Select features and labels
        X = normalized_df.iloc[:, :21]  # Select the first 21 features
        y = self.df[['Green frogs', 'Brown frogs', 'Common toad', 'Fire-bellied toad', 'Tree frog', 'Common newt',
                     'Great crested newt']]

        # Step 4: Apply PCA to reduce the dimensionality of X
        X_pca = self.apply_pca(X)

        # Return the processed features (X_pca) and labels (y)
        return X_pca, y
