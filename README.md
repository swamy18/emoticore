Emotion-Aware Text Summarizer 🧠✨
Hey there! This is a super handy Python tool I've put together. Its main job? To summarize your text, but with a cool twist: it tries its best to keep the original emotional vibe of your writing. I built it to be really straightforward and reliable – no over-engineering here, just getting the job done right.

🌟 What It Does
Keeps the Feeling: This is the core! It first checks if your text sounds positive, negative, or neutral, then aims to make the summary reflect that same emotion.

Smart Summaries: Under the hood, it uses a powerful model called BART (specifically, the large CNN version) to create those high-quality summaries.

Quick Emotion Check: For the sentiment analysis, I've used TextBlob because it's fast and light, perfect for a quick emotional read.

Flexible Text Input: You can feed it text right from your command line, or point it to a file – whatever works best for you.

Handles Big Texts: Got a really long article? No problem! It cleverly breaks it down into smaller pieces, summarizes them, and then summarizes those summaries to keep everything coherent.

Faster with GPUs: If you've got a CUDA-enabled graphics card, it'll automatically use it to speed things up. Nice!

Easy to Use: The command-line interface is pretty clean. Just a few simple arguments to tell it what to summarize and how.

🛠️ Getting Started (Installation)
Ready to give it a spin? Here's how to get it set up:

Grab the Code: If you haven't already, clone this repository to your local machine:

# git clone <your-repo-url>
# cd <your-repo-directory>

Install the Essentials: Open your terminal and run this command to install all the necessary Python libraries:

pip install torch transformers textblob tqdm

🚀 How to Use It
Using the summarizer is super simple right from your command line.

Summarize a File

Let's say you have an article in my_article.txt and want a 150-word summary saved to summary.txt:

python your_script_name.py --input my_article.txt --output summary.txt --max-length 150

Summarize Text Directly

Or, if you just have a quick paragraph you want to summarize:

python your_script_name.py --text "This is an amazing product! I am so incredibly happy with its performance and features. It truly exceeds all expectations." --max-length 80

Quick Look at the Options:

-i, --input: This is where you put the path to your text file.

-t, --text: Or, use this if you want to type your text directly. (You can't use both --input and --text at the same time, though!)

-o, --output: Want to save the summary to a file? Provide the path here. If you skip this, it'll just print the summary right in your console.

-l, --max-length: Sets the maximum word count for your summary. (Default is 150 words).

✅ A Few Quick Checks (Manual Testing)
Before you rely on it heavily, here are some quick ways to manually test it out:

What happens with empty input?

python your_script_name.py --text ""
# It should politely tell you: "Input text is empty. Nothing to summarize."

Summarizing really short text?

python your_script_name.py --text "This is a very short sentence. It has very few words."
# For super short texts (less than 30 words), it'll just return the original text.

Feeding it a huge file?

# Try making a large_file.txt (e.g., by copying a long book a few times)
python your_script_name.py --input large_file.txt
# It's designed to stop if the file is over 5MB, giving you a warning about the size.

Non-English text?

python your_script_name.py --text "La vida es bella y el sol brilla. Amo este día."
# It shouldn't crash! It'll try to summarize, but just a heads-up, the emotion analysis might not be super accurate for non-English.

🚧 Things to Keep in Mind (Limitations)
Like any project, there are a few areas where this tool has its quirks:

Emotion Detection is Simple: The TextBlob sentiment analysis is pretty basic. It might miss tricky things like sarcasm or really nuanced feelings. (Something to FIXME later!)

No Fancy GUI (Yet!): Right now, it's purely a command-line tool. I'm thinking about adding a simple graphical interface using Tkinter or PyQt down the road. (TODO in code!)

Best for English: It's primarily trained and works best with English text. Other languages might not get the same quality.

Big Texts Can Be Slow: While it handles long documents, if your text is really massive (like over 2000 words), it can still take a bit of time to process.

🤝 Want to Contribute?
I'm always open to improvements! If you have ideas for new features, spot a bug, or just want to make things better, feel free to open an issue or send in a pull request. Your contributions are super welcome!

📄 License
This project is open-sourced under the MIT License.
