from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline 


preproceso = ColumnTransformer(transformers=[],
                               remainder='passthrough')

pipeline = Pipeline([
    ('preproceso', preproceso),
    ('balanceo', None),
    ('escalado', None),
    ('modelo', None)])




pipeline_clustering = Pipeline([
    ('preproceso', preproceso),
    ('escalado', None),
    ('clustering',None)])


pipeline_pca = Pipeline([
    ('preproceso', preproceso),
    ('escalado', None),
    ('pca', None)])