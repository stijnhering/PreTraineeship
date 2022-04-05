import mlflow
from IPython.display import display
# predictor.py
class Predictor:
    def load(self):
        # from transformers import pipeline
        # self.model = pipeline(task="sentiment-analysis")
        logged_model = 'runs:/df5b50e3e4c14917a528245d7ce38593/model'

        # Load model as a PyFuncModel.
        self.model = mlflow.pyfunc.load_model(logged_model)

    async def predict(self, request):
        # We know we are going to use the `predict_dict` method, so we use
        # the request.payload pattern
        req = request.payload
        display(req)
        return self.model(req)