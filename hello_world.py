from pandas import read_csv
from matplotlib import pyplot 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the dataset
name="Iris.csv"
data=read_csv(name)
print(data.head())

#show the shape of the data
print("shape of data:",data.shape)
data.hist()
pyplot.savefig("iris_histogram.png")
pyplot.close()

data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
pyplot.savefig("iris_density.png")
pyplot.close()

# Verify dataset structure
print(data.head())
print(data.columns)

# Extract features and target
array = data.values
X = array[:, 1:5]  # Adjust based on dataset structure
Y = array[:, 5]    # Adjust based on dataset structure

# Encode target labels if necessary
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)

# Split the dataset into training and testing sets
test_size=0.33
seed=7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# Train the model
model=LogisticRegression(max_iter=200) 
model.fit(X_train,Y_train)

# Evaluate the model
result=model.score(X_test,Y_test)   
print("Accuracy: {:.2f}%".format(result*100))

# Save the model
joblib.dump(model,"iris_model.pkl")

