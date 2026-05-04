import os
from pathlib import Path
from dotenv import load_dotenv
from joblib import dump
import logging


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit  
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report
)




def training():
    try:
        load_dotenv()

        PROJECT_FOLDER=Path(os.getenv("PROJECT_FOLDER")).resolve()

        DATASET_PATH=PROJECT_FOLDER / os.getenv("DATASET_DIR") / os.getenv("DATASET_FILE")
        LOG_PATH=PROJECT_FOLDER / os.getenv("LOG_DIR") / os.getenv("LOG_FILE")
        MODEL_PATH=PROJECT_FOLDER / os.getenv("MODEL_DIR")  / os.getenv("MODEL_NAME")
        TARGET_COL=os.getenv("TARGET_COL")
        RANDOM_STATE=int(os.getenv("RANDOM_STATE"))
        TEST_SIZE=float(os.getenv("TEST_SIZE"))

        MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
        LOG_PATH.parent.mkdir(parents=True,exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LOG_PATH)
            ]
        )
        df=pd.read_csv(DATASET_PATH)
        logging.info(f"DataFrame loaded successfully {df.shape}")

        X=df.drop(columns=TARGET_COL)
        y=df[TARGET_COL]
        
        row_signature=pd.util.hash_pandas_object(X, index=False)

        gss=GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_idx, test_idx=next(gss.split(X,y,groups=row_signature))

        X_train, X_test =X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test =y.iloc[train_idx], y.iloc[test_idx]

        logging.info(f"X_train.shape={X_train.shape}  |  X_test.shape={X_test.shape}")
        model=RandomForestClassifier(
            n_estimators=824,
            max_depth=9,
            min_samples_split=20,
            min_samples_leaf=2,
            max_samples=0.744,
            max_features=0.25,
            bootstrap=True,
            ccp_alpha=np.float64(7.2755034093985316e-06),
            random_state=RANDOM_STATE
        )
        model.fit(X_train,y_train)
        
        logging.info("Model trained successfully")
        logging.info(f"Train Metrics:\n{classification_report(y_train,model.predict(X_train))}")
        logging.info(f"Test Metrics:\n{classification_report(y_test,model.predict(X_test))}")

        dump(model,filename=MODEL_PATH)
        logging.info(f"MODEL saved at:{MODEL_PATH}")
        logging.info(f"Training Script completed")

    except Exception as e:
        print(f"Training failed: {e}")
        logging.exception(f"Training Script failed {e}")
        raise
        

if __name__ == "__main__":
    training()
