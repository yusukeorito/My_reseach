 def preprocessing(self) -> np.ndarray:
        X_train = self.X_train
        X_test = self.X_test
        
        X_train_ = X_train.reshape(X_train.shape[0], -1)
        X_test_ = X_test.reshape(X_test.shape[0], -1)

        X_train_ = X_train_ / 255
        X_test_ = X_test_ / 255
        # Use 32-bit instead of 64-bit float
        X_train_ = X_train_.astype("float32")
        X_test_ = X_test_.astype("float32")
        return X_train_, X_test_