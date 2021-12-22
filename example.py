from chatbot import chatbot

model = chatbot()
'''
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
'''
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

model.train_model("model")
print(model.get_result("Oh, hello, nice to see you"))
print(model.get_result("Goodbye, I need to go"))
