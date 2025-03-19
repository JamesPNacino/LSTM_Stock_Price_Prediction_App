from myproject import db

#neural_network table - used to list all the created neural network models
class neural_network(db.Model):
	
	id = db.Column(db.Integer, primary_key=True)
	ticker = db.Column(db.Text)
	pattern = db.Column(db.Text)
	days_out_parameter = db.Column(db.Integer)
	percent_increase_parameter = db.Column(db.Integer)
	total_observations_train_and_val = db.Column(db.Integer)
	accuracy_test = db.Column(db.Integer)
	most_frequent_class_test_pct = db.Column(db.Integer)
	data_start_date = db.Column(db.Text)
	data_end_date = db.Column(db.Text)

    #e.g. an instance of this class must have a ticker for the instance to be generated (instance meaning a new observation)
	#def __init__(self, ticker):
	#	self.ticker = ticker
	#	self.pattern = pattern
	
	def __repr__(self):
		return f"Neural network ticker: {self.ticker}. model_id: {self.id}"
