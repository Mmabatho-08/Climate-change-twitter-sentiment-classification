"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "EDA", "Model Explanation", "About Us"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "EDA" page
	if selection == "EDA":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

	if selection == "Model Explanation":
		st.info("Insert explanations of the models we used")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

	if selection == "About Us":
		st.info("About the team")
		st.image("resources/imgs/Trendsetters_Analytics_Company_Logo.png", use_column_width = True)
		# You can read a markdown file from supporting resources folder
		st.markdown("**Edna Mosima Kobo**: Team leader and Project Manager - Oversees the project, coordinates team members, and ensures project goals are achieved.")
		st.markdown("**Donald Nkabinde**: Vice team leader and Data Analyst - Assists the team leader, contributes to data analysis, and provides insights and recommendations.")
		st.markdown("**Mmabatho Mojapelo**: Time Management Specialist - Manages project timelines, deadlines, and task prioritization for efficient project progress.")
		st.markdown("**Makosha Elizabeth Lekganyane**: Quality Control Analyst - Ensures accuracy, reliability, and quality of data, models, and outcomes.")
		st.markdown("**Khutso Madiga**: Data Engineer - Responsible for data acquisition, preprocessing, integration, and storage for high-quality data analysis.")
		st.markdown("**Hawert Tshepo Hobyane**: Feature Engineer - Identifies and designs relevant features to enhance model performance.")
		st.markdown("**Mack Thabo Ramalatso**: Data Scientist - Applies advanced analytics, develops and trains machine learning models, and extracts insights for predictions.")

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
