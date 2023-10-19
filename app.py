from flask import Flask, render_template, request, session, redirect,url_for
import openai
from langchain import OpenAI, LLMChain, PromptTemplate, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize


app = Flask(__name__)
# Set your OpenAI GPT-3 API key here
api_key = "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"
app.secret_key = 'afnan'

@app.route('/', methods=['GET', 'POST'])
def index():
    user_text = None
    ai_response = None

    if request.method == 'POST':
        user_text = request.form['user_input']

        # Process user_text using your OpenAI function (open_ai)
        ai_response = open_ai(user_text)

        # Store the user's input and the AI response in the session
        session['user_text'] = user_text
        session['ai_response'] = ai_response

        return redirect(url_for('index'))  # Redirect to prevent form resubmission

    user_text = session.get('user_text')
    ai_response = session.get('ai_response')

    # Clear the session variables when rendering the template
    session.pop('user_text', None)
    session.pop('ai_response', None)

    return render_template('index.html', user_text=user_text, ai_response=ai_response)

def open_ai(topic):

        # Specify the engine (e.g., "text-davinci-002" for the completions engine)
    engine = "text-davinci-002"

    # Set up the OpenAI API client
    openai.api_key = api_key


    # Create a prompt with emotions and detailed descriptions
    prompt = f"keep the personality hidden in first few lines starting with some plot describes this topic {topic} and  all the details of almost 500 words and some human-like pauses and tone change also, if it is related to some place then start it from the history of that land how essential is that land and if it is related to some building then start describing the most important things of the building and the reason for popularity but in a way that someone is telling the story with emotions and every line divides into line describing one thing only "

    # Generate the description
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=1048  # Adjust the maximum number of tokens as needed
    )

    # Print the generated description
    a = response.choices[0].text.strip()
    paragraph = a

    # Tokenize the paragraph into sentences
    sentences = sent_tokenize(paragraph)
    lines = []

    # Print each sentence on a separate line
    for sentence in sentences:
        # i = 1
        lines.append(sentence.strip())

    return lines


if __name__ == '__main__':
    app.run(debug=True)
