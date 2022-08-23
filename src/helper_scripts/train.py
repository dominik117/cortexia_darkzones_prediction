from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning
import helper_scripts.data_processor as data_processor
import time

def train_poisson_model(df, output):
    """     Poisson Algorithm
    One Hot Encode for categorical features
    Robust Scaler for numerical features
    With hyperparameter tunning

    Parameters
    ----------
    df : pandas.dataframe
        Cleaned and aggegated Dataframe with added features all from data_processor script
    output : string
        The numerical value of the desired litter to train ML model

    Returns
    -------
    pipeline_poisson : list
        List with the column names that where umerical values
    X_test : pandas.dataframe
        Dataframe with X_test
    y_test : pandas.dataframe
        Dataframe with y_test
    time2train : float
        Simply the time it took in minutes to fit the model
    """
    start_time = time.time()
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
    output = str(output)
    columns_to_drop = ['total_litter', 'total_litter_ratio']
    columns_to_drop.extend(data_processor.get_litter_columns(df))
    X = df.drop(columns=columns_to_drop, errors='ignore')
    y = df[output]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features = X_train.select_dtypes(include=['int', 'float']).columns.tolist()
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
    numeric_transformer = Pipeline(steps=[("scaler", RobustScaler())])
    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features),
                                                   ("cat", categorical_transformer, categorical_features)])
    model_poisson = PoissonRegressor(alpha=1e-12, max_iter=500)
    pipeline_poisson = Pipeline(steps=[("pre_process", preprocessor), ("poisson_model", model_poisson)])
    pipeline_poisson.fit(X_train, y_train)
    y_pred_poisson = pipeline_poisson.predict(X_test)
    y_pred_poisson = y_pred_poisson.astype(int)
    time2train = round((time.time() - start_time)/60, 1)
    print(f"The fitting took: {time2train} minutes")
    score = pipeline_poisson.score(X_test, y_test)
    print(f'Litter {output} D2 Score: {round(score, 4)}')
    print(f'#################################')
    return pipeline_poisson, X_test, y_test, time2train

def get_litter_labels():
    litter_labels = [['1', 'Cigarette'],['2', 'Leaf'],['3', 'Leaves'],['4', 'Paper/Carton'],['5', 'CAN'],['7', 'Glass bottle'],['8', 'PET'],['9', 'Carton drink'],
                ['10', 'FF Cup'],['11', 'FF Foam Polystrene'],['12', 'Other Foam Polystrene'],['13', 'Food packaging'],['14', 'Newspaper'],['15', 'Small bag'],
                ['16', 'Glass Splinter'],['17', 'Syringe'],['18', 'Organic food littering'],['19', 'Dog fouling'],['21', 'Garbage bags'],['22', 'Sand/Grit/Granulate'],
                ['23', 'Chewing- gum'],['24', 'Vomit'],['25', 'FF Cup'],['26', 'FF Lid'],['27', 'FF Straw'],['28', 'FF Fries cartin'],['29', 'Unclear bottles'],
                ['30', 'FF Burger Box'],['31', 'FF Paper'],['32', 'FF Other Paper'],['33', 'iQos'],['34', 'Confettis (pile)'],['35', 'Medium/big stain'],
                ['36', 'Transparent plastic'],['37', 'Opaque plastic'],['38', 'Fabric'],['39', 'Unrecognizable'],['40', 'Capsule'],['41', 'Carcass'],['42', 'Furniture'],
                ['43', 'Tag'],['44', 'Poster'],['45', 'Waste bin stain'],['46', 'Waste bin tag'],['47', 'Waste bin sticker'],['48', 'Waste bin Ouverture'],['49', 'Waste bin'],
                ['50', 'Cigarette white'],['51', 'Cigarette rolled'],['52', 'Cigarette unknown'],['53', 'Waste container too full'],['54', 'Illegal advertising poster'],
                ['55', 'Illegal advertising poster (influenceable)'],['56', 'Illegal litters'],['57', 'Spray painting, graffiti'],['58', 'Spray painting, graffiti (influenceable)'],
                ['59', 'Feuille mouillée'],['60', 'Poubelles remplies'],['61', 'Robydog'],['62', 'Wooden or plastic crate'],['63', 'Mask'],['total_litter', 'Total Litter']]
    return litter_labels


def make_models(df, litters):
    """     Creates a dictionary that stores the ML prediction model for later use or for exporting
    Parameters
    ----------
    df : pandas.dataframe
        Cleaned and aggegated Dataframe with added features all from data_processor script
    litters : list
        List containig the numerical values of the desired litters to train the ML models

    Returns
    -------
    models : dictionary
    Each key correspond to a type of litter, which in turn conaints the following:
        'model' : The ML algorithm
        'y_test' : Dataframe with y_test
        'score' : The deviance squared score for the Poisson Model
        'time2train' : Simply the time it took in minutes to fit the model
    """
    models = {}
    for litter in litters:
        litter = str(litter)
        model, X_test, y_test, time2train = train_poisson_model(df, litter)
        score = round(model.score(X_test, y_test), 4)
        models[litter] = [model, y_test, score, time2train]
    litter_labels = get_litter_labels()
    for key, value in models.items():
        for item in litter_labels:
            if key == item[0]:
                models[key].append(item[1])
    return models



