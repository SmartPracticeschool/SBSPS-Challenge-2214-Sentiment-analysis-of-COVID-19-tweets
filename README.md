# SBSPS-Challenge-2214-Sentiment-analysis-of-COVID-19-tweets
**Problem Statement**:

**Title**: 
Sentiment Analysis of COVID-19 Tweets – Visualization Dashboard

Description:
The sentiment analysis of Indians after the extension of lockdown announcements to be analyzed with the relevant #tags on twitter and build a predictive analytics model to understand the behavior of people if the lockdown is further extended.  Also develop a dashboard with visualization of people reaction to the govt announcements on lockdown extension.

Expected Solutions:
Develop a twitter sentiment analysis model to understand the following
1. Get to know people’s sentiment towards the epidemic.
2. Understand the sentiments of people on govt. decision to extend the lockdown.

Technologies and Tools: Python 2 or 3, IBM Watson Studio, IBM Cloud for Deployment, Any Web frameworks

**Our Solution**:
Our project will show the different sentiments in the form of histogram.
After each live telecast about the status of lockdown, the histogram will be updated according to the live tweets.
A comparison of the old and updated histograms will be shown side by side.
On clicking each part of the graph, more details about the tweets will be shown.

Upon giving a keyword, we can majorly obtain two graphs. The graphs being:
- Live Sentiment.
- Long Term Sentiment.

Live Sentiment graph shows the sentiment or the feedback of the general public in a live manner while Long Term Sentiment graph shows people's feedback cumulatively. The feedback for a longer period of time is added up and shown in this case. This allows the user to see the difference in actual sentiment of the public over a period of time.
The tweets corresponding to the keyword "corona" are obtained and sentimental analysis is performed and the output is shown in the form of a pie chart. The chart has 2 side : positive and negative depicted by the different colored sides of the chart. 
Each tweet is given a sentiment point with which the percentage of the pie chart

Suppose we use the keyword "covid" :
Keyword : covid

The tweets for "covid" are retrieved and the sentiment analysis is performed.
Similarly any keyword can be entered in the search bar provided at the top left corner of the screen and the tweets corresponding to that topic is retrieved and the sentiment of the public about that particular topic is then displayed as shown above.

**Quick Start**:
1. Clone the Repo.
2. Requirements.txt needs to be installed using pip install -r requirements.txt
3. Fill in your Twitter App credentials to twitter_stream.py . Go to apps.twitter.com to set that up if you need to.
4. To construct a database, run twitter_stream.py .
5. You can run this application with dev_server.py script if you’re using this locally. If you wish to deploy this to the webserver, do checkout my video on deploying Dash application tutorial.
(you might need the current version of sqLite.)

[YOUTUBE VIDEO](https://www.youtube.com/watch?v=A4etA_z-_Ew&feature=youtu.be) 
