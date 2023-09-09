#TODO
#1- create many model 
#user choose model to use
#user gets the output in two forms, visualization, or files
import joblib as jl
import pandas as pd
import numpy as np
import zipfile as zf
from attr_adder import CombinedAttributesAdder
with zf.ZipFile('./models/preprocessingPipeline.zip') as preprocess:
    pipeline = jl.load(preprocess.open('preprocessingPipeline.joblib'))
    
with zf.ZipFile('./models/fine_tuned_random_forest_v_1_0_0.zip') as modelForest:
    model = jl.load(modelForest.open('fine_tuned_random_forest_v_1_0_0.joblib'))
def preprocessing(housing_features):
    """
    returns the X preprocessed and ready for the algorithm
    """
    return pipeline.transform(housing_features)

def predictor(csv_filepath):
    features = pd.read_csv(csv_filepath)
    features_preprocessed = preprocessing(features)
    features["predictions"] = (pd.DataFrame(model.predict(features_preprocessed), columns={'predictions'}))["predictions"]
    return features.iloc[:,1:]


def prediction_file(inputpath, outpath):
    predictor(inputpath).to_csv(outpath,  encoding='utf-8')

