# GPT-based Chatbot demo
This GPT-based chatbot demo uses OpenAI API for answering questions - based on your own data. You can add your book, and start chatting with it. As this code is made for demonstration purposes, it's working, but unfinished. I have added some "TODO" comments on how you can further develop it.

## How to use
 * Add your OpenAI API key to the .env file (copy the .env.example file, and remove the .example part, then replace the my-openai-api-key with your real one - as well as the organization id - you can also add None)
 * Add your txt and pdf files to a subfolder under ./data/01_raw/ (in the example, I added book_robot with a few examples).
 * Add the path to the demo_openai_bot.yaml file (path_raw_folder, path_clean_folder, path_kb)
 * Run the notebook - and start chatting.