📊 Social Media K-Means Clustering Dashboard
#---------------------------
An interactive Streamlit dashboard for analyzing and clustering social media post engagement data using the K-Means Clustering algorithm.
The project provides interactive visualizations, cluster analysis, PCA visualization, the Elbow Method, and a filtered data explorer.
🚀 Features
- 📊 Interactive dashboard with KPI cards
- 🎯 K-Means clustering with adjustable number of clusters
- 📈 Social media engagement analysis
- 🔵 2D PCA visualization of clusters
- 📐 Elbow Method for selecting K
- 📊 Cluster distribution and cluster profiles
- 🔥 Status Type vs Cluster heatmap
- 🔎 Filter data by cluster and status type
- 📋 Interactive dataset explorer
- ⬇️ Download filtered clustered data as CSV
- 📁 Upload a compatible CSV dataset directly through the dashboard
- 🎨 Professional Streamlit interface
- 📊 Interactive Plotly charts
🧠 Machine Learning Workflow
The project follows this workflow:
Raw Dataset
     ↓
Data Cleaning
     ↓
Remove Unnecessary Columns
     ↓
Encode Status Type
     ↓
Handle Missing Values
     ↓
Min-Max Scaling
     ↓
K-Means Clustering
     ↓
Cluster Evaluation
     ↓
PCA Visualization
     ↓
Interactive Dashboard
📂 Project Structure
K-Means Clustering/
│
├── app.py
├── Live.csv
├── Day 71_K means Clustering_Project.ipynb
└── README.md
Files
File	Description
app.py	Main Streamlit dashboard application
Live.csv	Dataset used for clustering
Day 71_K means Clustering_Project.ipynb	Jupyter Notebook containing the K-Means analysis
README.md	Project documentation


🗃️ Dataset
The dashboard is designed for a social-media engagement dataset containing information such as:
- status_id
- status_type
- status_published
- num_reactions
- num_comments
- num_shares
- num_likes
- num_loves
- num_wows
- num_hahas
- num_sads
- num_angrys
The exact columns may vary depending on the dataset provided.
⚙️ Data Preprocessing
Before applying K-Means, the application performs the following steps:
1. Remove unnecessary columns
Identifier/date-related columns and redundant columns are removed where available.
2. Handle missing values
Missing numerical values are replaced using the median of the corresponding feature.
3. Encode categorical data
The status_type column is converted into numerical form using LabelEncoder.
4. Feature Scaling
MinMaxScaler is used to scale the features into a common range.
5. K-Means Clustering
The processed data is grouped using K-Means clustering.
🎯 K-Means Clustering
K-Means divides the dataset into K clusters based on feature similarity.
The dashboard allows the user to change:
- Number of clusters (K)
- Random state
The default number of clusters is 4, but it can be adjusted from the sidebar.
📐 Elbow Method
The Elbow Method is included to help determine a suitable value of K.
The dashboard calculates K-Means inertia for different values of K.
K = 2
K = 3
K = 4
...
K = 10
The point where the decrease in inertia starts becoming slower can be considered a possible optimal K.
📊 Silhouette Score
The dashboard also calculates the Silhouette Score to evaluate cluster quality.
The score generally ranges from:
-1 to +1
A higher score generally indicates that the clusters are better separated.
🔵 PCA Visualization
Principal Component Analysis (PCA) is used to reduce the scaled feature space to two dimensions.
This makes it possible to visually observe the K-Means clusters in a 2D scatter plot.
The PCA visualization displays:
- Principal Component 1
- Principal Component 2
- Cluster
- Status Type
Hovering over points provides additional information.
📈 Dashboard Sections
1. Overview
Displays:
- Total Posts
- Number of Clusters
- Total Reactions
- Total Comments
- Silhouette Score
- Posts by Status Type
- Engagement distributions
- Overall engagement metrics
2. Cluster Analysis
Displays:
- Cluster distribution
- Cluster share
- Average engagement by cluster
- Cluster profiles
- Status Type vs Cluster heatmap
3. PCA Visualization
Displays:
- 2D PCA scatter plot
- Cluster separation
- Cluster centroids
4. Elbow Method
Displays:
- K-Means inertia
- Elbow curve
- Current K
- Current inertia
- Silhouette Score
5. Data Explorer
Allows users to:
- Filter by cluster
- Filter by status type
- View clustered records
- Download filtered data
🛠️ Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Jupyter Notebook
💻 Installation
Make sure Python is installed on your system.
Step 1: Clone or download the project
Place the project files in a folder such as:
K-Means Clustering
Step 2: Open the terminal
Navigate to the project folder:
cd "K-Means Clustering"
Step 3: Install required libraries
pip install streamlit pandas numpy scikit-learn plotly
▶️ Run the Dashboard
Run:
streamlit run app.py
Streamlit will start a local server, usually at:
http://localhost:8501
Open that address in your browser.
📤 Using Your Own Dataset
You can upload a CSV file using the Upload CSV Dataset option in the sidebar.
For the full dashboard functionality, the dataset should contain a status_type column and relevant numerical engagement features.
📥 Export Results
After applying filters in the Data Explorer, click:
⬇️ Download Filtered CSV
The dashboard generates:
clustered_social_media_data.csv
containing the filtered records and their assigned cluster.
📊 Example Insights
The dashboard can help identify:
- Which posts have higher engagement
- Which status types dominate the dataset
- How posts are distributed among clusters
- Which clusters receive more reactions
- Which clusters receive more comments or shares
- Similarity patterns among social media posts
- Potential high-engagement and low-engagement groups
⚠️ Important Notes
- K-Means is an unsupervised learning algorithm.
- The value of K can significantly affect the clustering results.
- PCA is used mainly for visualization and does not replace the original feature space used for clustering.
- Results may change if the dataset or preprocessing method changes.
- The dashboard is intended for educational, analytical, and project demonstration purposes.
👩‍💻 How to Run in VS Code
Open the project folder in VS Code.
Open the integrated terminal and run:
pip install streamlit pandas numpy scikit-learn plotly
Then:
streamlit run app.py
🌐 Streamlit Deployment
The application can be deployed using Streamlit Community Cloud.
Typical deployment steps:
1. Upload the project to GitHub.
2. Make sure app.py and Live.csv are available in the repository.
3. Create a requirements.txt file containing:
streamlit
pandas
numpy
scikit-learn
plotly
4. Connect the GitHub repository to Streamlit Community Cloud.
5. Select app.py as the main application file.
6. Deploy the application.
📜 License
This project is intended for educational and learning purposes.
⭐ Project Summary
Social Media K-Means Clustering Dashboard combines machine learning and interactive data visualization to discover meaningful groups in social media engagement data.
It demonstrates a complete workflow from data preprocessing → feature scaling → K-Means clustering → evaluation → PCA visualization → interactive dashboard.
Built With ❤️ Using Python, Streamlit & Machine Learning
