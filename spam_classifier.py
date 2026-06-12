# Spam Email Classifier using TF-IDF and Naive Bayes

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Sample dataset
messages = [
    "Congratulations! You won a free lottery prize",
    "Free entry in a contest, claim your reward now",
    "You have won a cash prize click here",
    "Meeting is scheduled at 10 AM tomorrow",
    "Please send the project report today",
    "Let's have lunch together",
    "Your mobile bill is generated",
    "Call me when you are free",
    "You got a free gift voucher",
    "Important: Your account statement is ready"
]


# Labels
# spam = spam message
# ham = normal message

labels = [
    "spam",
    "spam",
    "spam",
    "ham",
    "ham",
    "ham",
    "ham",
    "ham",
    "spam",
    "ham"
]


# Convert text into numbers using TF-IDF

vectorizer = TfidfVectorizer()


X = vectorizer.fit_transform(messages)


# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.3,
    random_state=1
)


# Create Naive Bayes model

model = MultinomialNB()


# Train model

model.fit(
    X_train,
    y_train
)


# Test accuracy

prediction = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    prediction
)


print("Model Accuracy:", accuracy)


# Test new message

while True:

    text = input("\nEnter message: ")

    if text == "exit":
        break


    # Convert message to TF-IDF

    new_message = vectorizer.transform(
        [text]
    )


    result = model.predict(
        new_message
    )


    print(
        "Prediction:",
        result[0]
    )