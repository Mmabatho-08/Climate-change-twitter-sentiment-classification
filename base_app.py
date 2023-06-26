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

from streamlit_option_menu import option_menu # need to pip install streamlit-option-menu to use this

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/vect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.set_page_config(layout="wide")
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	# with st.sidebar:
	selection = option_menu(
		menu_title = None,
		options = ["Info", "Predict", "EDA", "Models", "About Us"],
		icons = ["info-square", "twitter", "bar-chart-line", "book", "envelope"],
		default_index = 0,
		menu_icon = "house",
		orientation = "horizontal",
		styles = {
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "lightskyblue"},
    })
	# selection = st.sidebar.radio("Choose Option", options)

	# Building out the "EDA" page
	if selection == "Info":
		st.info("🤷‍♂️ Purpose of the app")
		st.markdown("Welcome to the digital age, where advancements in social media have given rise to the significance of sentiment analysis. Powered by artificial intelligence and natural language processing, sentiment analysis allows us to delve into the realm of public perspectives and emotional insights. By harnessing the vast ocean of social media data, particularly platforms like Twitter, we can uncover the collective sentiment surrounding various topics, products, events, or trends. This understanding enables businesses to adapt their strategies, refine their offerings, and align their messaging with the prevailing sentiments, leading to data-driven decisions, enhanced brand reputation management, and improved customer satisfaction in the dynamic digital landscape.")
		st.markdown("Despite the vast potential of leveraging Twitter data for marketing purposes, marketers face challenges in effectively interpreting and utilizing sentiment analysis. While sentiment analysis provides insights into public sentiment, marketers often express concerns about the accuracy and contextual understanding of sentiment analysis algorithms. The nuances of human expression, sarcasm, or cultural variations in language can be challenging to capture accurately. Marketers also need to consider the limitations of automated sentiment analysis, as it may not fully capture the complexities of consumer emotions. Addressing these concerns and improving the accuracy and contextual understanding of sentiment analysis algorithms is crucial for marketers to confidently leverage Twitter data for actionable insights and informed decision-making in their marketing strategies.")
		st.markdown("This app, therefore, showcases how machine learning can be utilised to solve this problem.")

		st.info("💡 How to use this app")
		st.markdown("This app requires the user to input text, ideally a tweet relating to climate change, and will classify it into one of the four classes given below. In the section below, you will find information about the data source and a brief description of the data. To make a prediction, navigate to **Predict** on the menu at the top. Once there you should insert the tweet you would like to classify into the textbox. Furthermore, you can select a classification model of your choosing on this page. Explanations of the models used are given in the **Models** menu. A deep dive into the data used to train the models is given in the **EDA** menu.")

		st.info("📚 Data description")
		st.markdown("The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo. This dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were annotated. Each tweet is labelled independently by 3 reviewers. This dataset only contains tweets that all 3 reviewers agreed on (the rest were discarded). Each tweet is labelled as one of the following classes:")
		st.markdown("- 2 (News): the tweet links to factual news about climate change")
		st.markdown("- 1 (Pro): the tweet supports the belief of man-made climate change")
		st.markdown("- 0 (Neutral): the tweet neither supports nor refutes the belief of man-made climate change")
		st.markdown("- -1 (Anti): the tweet does not believe in man-made climate change")

		st.info("📈 To view the raw data, select the checkbox below.")
		# st.subheader("Raw Twitter data and label")
		if st.checkbox("Show raw data"): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the "EDA" page
	if selection == "EDA":
		st.info("🕵️‍♂️ This page explains information gathered from the data. To expand the image, hover over it and click the expand arrows that appear.")
		
		st.markdown("---")

		st.markdown("The figures below were extracted during the exploritory data analysis (EDA) part of this project. Some additional information is provided alongside the figures.")
		st.markdown("---")
		
		col1, col2 = st.columns(2)

		with col1:
			st.image("resources/imgs/figure_1.png")

		with col2:
			st.image("resources/imgs/figure_2.png")
		
		st.markdown("The previous two plots indicate that our dataset is not balanced. There are more entries belonging to people who have the belief of man-made climate change. Over 50% of the data is comprised of such tweets.")
		st.markdown("---")
		#-------------------------------------------------------------
		### Hacky way to centre the images
		col1, col2, col3 = st.columns(3)

		with col1:
			st.write(" ")

		with col2:
			st.image("resources/imgs/figure_3.png")

		with col3:
			st.write(" ")

		st.markdown("The length of the text is determined by the character limit of each tweet. The character limit used to be 140, however in [late 2017](https://www.forbes.com/sites/nicholasreimann/2023/02/08/twitter-boosts-character-limit-to-4000-for-twitter-blue-subscribers/?sh=689ca7825ab8), it was expanded to 280 characters. All sentiment classes have outliers, with the only exception being the `Neutral` class.")
		st.markdown("---")
		#-------------------------------------------------------------
		col1, col2, col3 = st.columns(3)

		with col1:
			st.write(" ")

		with col2:
			st.image("resources/imgs/figure_4.png")

		with col3:
			st.write(" ")

		st.markdown("The following bullet points relate to the figure above:")
		st.markdown("- `#climate` and `#climatechange` are expected to be the most popular as they are our key identifier in tweets.")
		st.markdown("- `#BeforeTheFlood` surfaced after a [documentary](https://en.wikipedia.org/wiki/Before_the_Flood_(film)) about environmental degradation that leads to global warming and suggestions on how to reduce it, narrated by Leonardo DiCaprio. This hashtag is most popular in tweets belonging in the `Pro` and `Neutral` categories.")
		st.markdown("- `#trump` is one of the top hashtags for the Anti class. This could possibly be due to Donald Trump calling climate change a [\"hoax\"](https://www.motherjones.com/environment/2016/12/trump-climate-timeline/) multiple times.")
		st.markdown("- The `#cop22` represents the [United Nations Climate Change Conference](https://unfccc.int/event/cop-22) that took place in 2016.")
		st.markdown("---")
		#-------------------------------------------------------------
		col1, col2, col3 = st.columns(3)

		with col1:
			st.write(" ")

		with col2:
			st.image("resources/imgs/figure_5.png")

		with col3:
			st.write(" ")

		st.markdown("Other than news outlets, politicians are the most engaged individuals in climate change discussions. This has shifted this issue from being a scientific issue into a political one.")
		# #-------------------------------------------------------------
		# col1, col2, col3 = st.columns(3)

		# with col1:
		# 	st.write(" ")

		# with col2:
		# 	st.image("resources/imgs/figure_6.png")

		# with col3:
		# 	st.write(" ")
		# #-------------------------------------------------------------

	# Building out the predication page
	if selection == "Predict":
		output = {-1: "Anti", 0: "Neutral", 1: "Pro", 2: "News"}
		st.info("🗳️ Which model would you like to use?")
		# Creating a text box for user input
		model = st.radio(" ", 
		   ("LinearSVC", "Logistic Regression", "Stochastic Gradient Descent (SGD)"), horizontal=True)

		if model == "LinearSVC":
			tweet_text = st.text_area("Enter text (Replace text below)", "CHECK OUT THESE WEATHER STORIES https://t.co/LwVzcPO30e Do Not believe the Global warming climate change stories sold by UN, Vatican &amp; Obama")
			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/final_lsvc.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				st.success("Text Categorized as: {}".format(output[prediction[0]]))

		if model == "Logistic Regression":
			tweet_text = st.text_area("Enter text (Replace text below)", "CHECK OUT THESE WEATHER STORIES https://t.co/LwVzcPO30e Do Not believe the Global warming climate change stories sold by UN, Vatican &amp; Obama")
			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/final_logistic.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				st.success("Text Categorized as: {}".format(output[prediction[0]]))

		if model == "Stochastic Gradient Descent (SGD)":
			tweet_text = st.text_area("Enter text (Replace text below)", "CHECK OUT THESE WEATHER STORIES https://t.co/LwVzcPO30e Do Not believe the Global warming climate change stories sold by UN, Vatican &amp; Obama")
			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/final_SGD.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				# st.success("Text Categorized as: {}".format(prediction[0]))
				st.success("Text Categorized as: {}".format(output[prediction[0]]))

		st.write("Note: Our models have ~75% accuracy. So not all classifications will be correct.")

	if selection == "Models":
		st.info("🤖 This section provides explanations of the three models we used. We also provide advantages and disadvantages of the models.")
		st.markdown("The figure below shows the performances of the 11 models we trained. For this app, only the three best models were chosen. The three models are explained below.")
		st.image("resources/imgs/figure_7.png")
		st.markdown("## 1️ LinearSVC (Support Vector Classifier)")
		st.markdown("LinearSVC is a linear classification model based on Support Vector Machines (SVM). It aims to find a hyperplane that separates the data into different classes with the maximum margin.")
		st.markdown("#### ✅ Advantages:")
		st.markdown("- Effective in high-dimensional spaces: LinearSVC performs well in datasets with a large number of features or dimensions.")
		st.markdown("- Memory-efficient: LinearSVC uses a subset of training points (support vectors) in the decision function, making it memory-efficient compared to other SVM variants.")
		st.markdown("- Suitable for large datasets: LinearSVC scales well with large datasets, making it computationally efficient.")
		st.markdown("#### ❌ Disadvantages:")
		st.markdown("- Requires feature scaling: LinearSVC is sensitive to feature scaling, and it is generally recommended to scale the input features before training the model.")
		st.markdown("- Limited to linearly separable data: LinearSVC is not suitable for datasets that are not linearly separable unless the data is transformed into a higher-dimensional space using kernel methods.")
		st.markdown("- Less effective with noisy data: LinearSVC may struggle to classify noisy data or data with overlapping classes.")

		st.markdown("---")

		st.markdown("## 2️ Logistic Regression")
		st.markdown("Logistic Regression is a statistical model used for binary classification. It estimates the probability of an instance belonging to a particular class based on its features.")
		st.markdown("#### ✅ Advantages:")
		st.markdown("- Simple and interpretable: Logistic regression is a straightforward model that provides interpretable coefficients, allowing us to understand the impact of each feature on the predicted probability.")
		st.markdown("- Efficient training: Logistic regression can be trained efficiently on large datasets compared to more complex models.")
		st.markdown("- Works well with linearly separable data: Logistic regression performs well when the data can be separated by a linear decision boundary.")
		st.markdown("#### ❌ Disadvantages:")
		st.markdown("- Assumes linearity: Logistic regression assumes a linear relationship between the features and the log-odds of the target variable. It may not capture complex non-linear relationships.")
		st.markdown("- Not suitable for non-linear data: Logistic regression may struggle to capture non-linear decision boundaries, leading to lower performance on datasets with complex relationships.")
		st.markdown("- Prone to overfitting with high-dimensional data: When the number of features is large compared to the number of instances, logistic regression can be prone to overfitting.")

		st.markdown("---")

		st.markdown("## 3️ Stochastic Gradient Descent (SGD)")
		st.markdown("Stochastic Gradient Descent (SGD) is an optimisation algorithm commonly used for training linear classifiers and regressors. It updates the model parameters based on the gradients computed on small random subsets of the training data. SGD can be used with different loss functions, making it versatile for different types of models. ")
		st.markdown("#### ✅ Advantages:")
		st.markdown("- Efficiency with large datasets: SGD is computationally efficient and can handle large datasets, as it updates the model parameters on small subsets of the data.")
		st.markdown("- Suitable for online learning: SGD can be used for online learning, where the model is updated incrementally as new data arrives.")
		st.markdown("- Versatility in loss functions: SGD supports various loss functions, making it adaptable to different types of models and tasks.")
		st.markdown("#### ❌ Disadvantages:")
		st.markdown("- Requires careful hyperparameter tuning: The learning rate and regularisation parameters in SGD need to be tuned carefully to achieve optimal performance.")
		st.markdown("- Sensitive to feature scaling: Like other linear models, SGD benefits from feature scaling to ensure fair treatment of different features.")
		st.markdown("- May converge to suboptimal solutions: SGD can converge to suboptimal solutions if the learning rate is too high or the training data is noisy. It requires careful parameter selection and appropriate data preprocessing.")

		st.markdown("---")

		st.markdown("## 🔚 Conclusion")
		st.markdown("Our best models are linear classifiers. Text classification problems are typically high dimensionality, and high dimensionality problems are likely to be linearly separable. Thus, linear classifiers perform well because they help to avoid over-fitting by separating the patterns of each class by large margins.")


	if selection == "About Us":
		st.info("😎 About the team")
		#-------------------------------------------------------------
		col1, col2, col3 = st.columns(3)

		with col1:
			st.write(" ")

		with col2:
			st.image("resources/imgs/Trendsetters_Analytics_Company_Logo.png")

		with col3:
			st.write(" ")
		#-------------------------------------------------------------
		st.markdown("The Trendsetters Analytics logo embodies our commitment to discovering, analysing, and leading trends. With a sleek and dynamic design, it represents our expertise in providing insightful analytics solutions. The logo features the name \"Trendsetters Analytics\" alongside our empowering slogan \"Discover, Analyze, Lead...\" This combination reflects our dedication to helping businesses stay ahead of the curve by unlocking trend insights. Below is a list of the team members making this possible:")
		st.markdown("**[Edna Mosima Kobo](https://github.com/EdnaM06)**: Team leader and Project Manager - Oversees the project, coordinates team members, and ensures project goals are achieved.")
		st.markdown("**[Donald Nkabinde](https://github.com/khulu2)**: Vice team leader and Data Analyst - Assists the team leader, contributes to data analysis, and provides insights and recommendations.")
		st.markdown("**[Mmabatho Mojapelo](https://github.com/Mmabatho-08)**: Time Management Specialist - Manages project timelines, deadlines, and task prioritisation for efficient project progress.")
		st.markdown("**Makosha Elizabeth Lekganyane**: Quality Control Analyst - Ensures accuracy, reliability, and quality of data, models, and outcomes.")
		st.markdown("**Khutso Madiga**: Data Engineer - Responsible for data acquisition, preprocessing, integration, and storage for high-quality data analysis.")
		st.markdown("**[Hawert Tshepo Hobyane](https://github.com/HawertHobyane)**: Feature Engineer - Identifies and designs relevant features to enhance model performance.")
		st.markdown("**Mack Thabo Ramalatso**: Data Scientist - Applies advanced analytics, develops and trains machine learning models, and extracts insights for predictions.")

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
