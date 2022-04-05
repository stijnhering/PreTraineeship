from predictor import Predictor
import budgetml
# add your GCP project name here.
budgetml = budgetml.BudgetML(project='TEST_PROJECT')

# launch endpoint
budgetml.launch_local()