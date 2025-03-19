from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField, SelectField, RadioField

class RunTickerModel_Form(FlaskForm):

	ticker_query = StringField('Input stock ticker:')
	pattern = SelectField('Input candlestick pattern:', choices=[('Random', 'Random'), ('Hammer', 'Hammer'), ('Inverted Hammer', 'Inverted Hammer')])
	days_out = SelectField('Input days out parameter:', choices=[('1', '1'), ('3', '3'), ('5', '5'), ('10', '10'), ('15', '15')], default='10')
	pct_increase = SelectField('Input percent increase parameter:', choices=[('1.0', '1.0'), ('1.01', '1.01'), ('1.02', '1.02')], default='1.01')
	data_end_date = StringField('Ending date for gathering data - must format as yyyy-mm-dd (i.e. 2020-01-23):')
	submit = SubmitField('Submit')

class Predict_Form(FlaskForm):

	submit = SubmitField('Submit')

class SearchCandlestick_Form(FlaskForm):

	ticker_query = StringField('Input stock ticker:')
	submit = SubmitField('Submit')
