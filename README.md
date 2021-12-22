# Example
Initialize class
```python
from chatbot import chatbot

model = chatbot()
```
Add topics like this
```python
model.add_topic(
  topic_name="greeting",
  patterns=["Hi", "Hello", "how are you", "hey", "Good morning", "Good afternoon"],
  responses=["Hi", "Hello"]
)

model.add_topic(
  topic_name="goodbye",
  patterns=["See you later", "Goodbye", "I am Leaving", "Have a Good day"],
  responses=["Goodbye"]
)
```
or like this
```python
model.data = {
    "greeting": {
      "patterns": ["Hi", "Hello", "how are you", "hey", "Good morning"],
      "responses": ["Hi", "Hello"]
  },
    "goodbye": {
      "patterns": ["See you later", "Goodbye", "I am Leaving", "Have a Good day"],
      "responses": ["Goodbye!"]
  }
}
```
Train network
```python
model.train_model("model")
```
Wait and test it
```python
print(model.get_result("Oh, hello, nice to see you"))
```
# Installation
- Install python 3.8.6 (no problems with tensorflow)
- ```[Environment]::SetEnvironmentVariable("python3.8", "C:\path\to\python3.8.6", "User")``` in PowerShell 
- Install nltk, tensorflow, tflearn
