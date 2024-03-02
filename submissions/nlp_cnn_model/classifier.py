import numpy as np
import pandas as pd
from scipy.special import softmax
from tqdm import tqdm

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms


# the input of both the NLP and the CNN classifiers are the same : a dataframe
# with one column for the images, two columns for the types (targets), and the
# other columns for the names in different languages


class Transform_dataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = torch.FloatTensor(self.labels[idx])
        return sample, label
    

class Data :

    def get_data_XNLP(df : pd.DataFrame) -> pd.DataFrame:
        X = df[['French', 'English', 'German', 'Kanas', 'Registered']]
        return X


    def get_data_yNLP(df : pd.DataFrame) ->  np.ndarray:
        # to one hot encode the target
        mlb = MultiLabelBinarizer()

        targets = ["Type 1", "Type 2"]
        y = df[targets].apply(lambda x: [x['Type 1']] if pd.isna(x['Type 2']) else [x['Type 1'], x['Type 2']], axis=1)

        y = mlb.fit_transform(y)

        return y
    

    # to be modified : this function should return the dataloader given df
    # a transformation should be applied to get each image of dimension sizeX x sizeY
    def get_data_CNN(df : pd.DataFrame, batch_size : int = 8, sizeX : int = 256, sizeY : int = 256) -> pd.DataFrame:

        # probably doesn't work
        X = df['Image'].values
        y = df[["Type 1", "Type 2"]]

        # probably doesn't work
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        dataset = Transform_dataset(dataset, transforms.Resize((sizeX, sizeY)))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataloader
    



class NLPClassifier(BaseEstimator) :

    def __init__(self, max_features : int = 1000, max_iter : int = 1000, class_weight : str = 'balanced') :

        self.num_classes = 18
        # the minimum size of the n-grams
        self.min_gram_range = 1
        # the maximum size of the n-grams
        self.max_gram_range = 3

        # consider the top 'max_features' for NLP model
        self.max_features = max_features
        # the maximum number of iterations for the logistic regression
        self.max_iter = max_iter
        # balance for the logistic regression
        self.class_weight = class_weight



    def set_model(self) :

        transformer_element = lambda name,column : (name, TfidfVectorizer(analyzer="char", ngram_range=(self.min_gram_range, self.max_gram_range), max_features=self.max_features), column)

        transformer = ColumnTransformer(
            transformers=[
                transformer_element("en", "English"),
                transformer_element("de", "German"),
                transformer_element("fr", "French"),
                transformer_element("ka", "Kanas"),
                transformer_element("re", "Registered"),
            ],
            remainder="drop",
            sparse_threshold=0,
        )

        model = MultiOutputClassifier(LogisticRegression(max_iter=self.max_iter, class_weight=self.class_weight))

        self.model = make_pipeline(transformer, model)


    def fit(self, X : pd.DataFrame, y : np.ndarray) :

        if not isinstance(X, pd.DataFrame) :
            raise TypeError(f"The input should be a pandas Dataframe, found {type(X)} instead")
        
        if not isinstance(y, np.ndarray) :
            raise TypeError(f"The input should be a pandas Dataframe, found {type(y)} instead")
        
        self.set_model()

        self.model.fit(X, y)

        self.is_fitted_ = True


    def predict_proba(self, X : pd.DataFrame) :
        """
        Return the raw probabilities of each class (transformations may be applied
        afterwards when calling the final classifier)
        """
        if not isinstance(X, pd.DataFrame) :
            raise TypeError(f"The input should be a pandas Dataframe, found {type(X)} instead")
        
        check_is_fitted(self)

        pred_probas = self.model.predict_proba(X)
        pred_probas = np.array(pred_probas)[:,:,1].T

        return pred_probas


    
    



class CNN(nn.Module):

    def __init__(self, img_height, img_width, channels, n_labels):

        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 150, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(150)

        self.fc1 = nn.Linear(150 * (img_height // 32) * (img_width // 32), 64)
        self.fc2 = nn.Linear(64, n_labels)


    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    


class CNNClassifier :

    def __init__(
        self, learning_rate : float = 0.001, epochs : int = 10, batch_size : int = 8, sizeX : int = 256, sizeY : int = 256,
    ):
        self.num_classes = 18
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.model = CNN(self.sizeX, self.sizeY, channels=4, n_labels=self.num_classes)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)


    def fit(self, train_loader : DataLoader):
        
        self.model.to(self.device)

        self.model.train()

        # Training loop
        for epoch in tqdm(range(self.epochs)):

            for inputs, labels in train_loader:

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()
        
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")



    def predict_proba(self, test_loader : DataLoader) -> np.ndarray:

        self.model.eval()

        probabilities = []

        with torch.no_grad():

            for inputs,_ in test_loader :
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probabilities.extend(outputs.cpu())

        return torch.sigmoid(probabilities).numpy()




class Classifier(BaseEstimator):

    def __init__(self, normalisation : str | None = None, weightNLP : float = 1., threshold : float = .07, **kwargs):

        self.num_classes = 18
        self.modelNLP = NLPClassifier(**kwargs)
        self.modelCNN = CNNClassifier(**kwargs)

        # how to normalise the probabilities
        normalisation_methods = {None, 'softmax', 'sum'}
        if not normalisation in normalisation_methods:
            raise ValueError(f"The normalisation method {normalisation} is not in {normalisation_methods}")
        self.normalisation = normalisation

        # this factor indicated the weight of the probability
        # of the NLP model
        if (weightNLP < 0.) or (weightNLP > 1.) :
            raise ValueError(f"The weight of the NLP model should be in [0,1], but found {weightNLP}")
        self.weightNLP = weightNLP

        # the threshold above which a second type is predicted
        # (to be modulated with the normalisation)
        self.threshold = threshold


    def fit(self, df : pd.DataFrame):

        if not isinstance(df, pd.DataFrame) :
            raise TypeError(f"The input should be a pandas Dataframe, found {type(df)} instead")
        
        XNLP = Data.get_data_XNLP(df)
        yNLP = Data.get_data_yNLP(df)
        train_loaderCNN = Data.get_data_CNN(df, self.modelCNN.batch_size, self.modelCNN.sizeX, self.modelCNN.sizeY)
        

        self.modelNLP.fit(XNLP, yNLP)
        self.modelCNN.fit(train_loaderCNN)

        self.is_fitted_ = True

        return self
    

    def predict_proba(self, df : pd.DataFrame) -> np.ndarray:

        check_is_fitted(self)

        if not isinstance(df, pd.DataFrame) :
            raise TypeError(f"The input should be a pandas Dataframe, found {type(df)} instead")
        
        XNLP = Data.get_data_X_NLP(df)
        test_loaderCNN = Data.get_data_CNN(df, self.modelCNN.batch_size, self.modelCNN.sizeX, self.modelCNN.sizeY)

        probasNLP = self.modelNLP.predict_proba(XNLP)
        probasCNN = self.modelCNN.predict_proba(test_loaderCNN)

        averaged_probas = self.weightNLP*probasNLP + (1- self.weightNLP)*probasCNN

        if self.normalisation == 'softmax' :
            averaged_probas = softmax(averaged_probas, axis=1)
        elif self.normalisation == "mean" :
            row_sums = averaged_probas.sum(axis=1).reshape(-1, 1)
            averaged_probas /= row_sums
        # if normalisation == None do nothing

        return averaged_probas
    
    
    def predict(self, df : pd.DataFrame) -> np.ndarray:

        pred_probs = self.predict_proba(df)

        preds = np.zeros_like(pred_probs)

        for i,pred_proba in enumerate(pred_probs) :

            t = np.argsort(pred_proba)

            # no matter what, the biggest probability is considered
            preds[i, t[-1]] = 1

            second_best = t[-2]
            # the second best is considered only if higher than the threshold
            if pred_proba[second_best] > self.threshold:
                preds[i, second_best] = 1

        return preds


if __name__ == "__main__" :

    path_df = "Data/Table/Pokemon_name_and_type.parquet"

    df = pd.read_parquet(path_df)

    clf = Classifier()
    clf.fit(df)
    print(clf.predict(df))