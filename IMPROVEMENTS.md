# Word Embedding vector dimension reduction
Qdrant had a nice summary on quantization methods for its DB, i've used Binary Quantization.
Recall was used as the main metric in that article.
https://qdrant.tech/documentation/guides/quantization/

It would be nice to write a script measuring the change of recommended results made with the quantization method compared to the original one functioning as GT.

# UI
synonym uses tags from create_dummy_tag.py file, it should be loaded once the dropdown list was selected, but wasn't familiar enough with gradio library to achieve that

# Unit tests
Unit tests for synonym functionality could be useful

# Maintenance

In case of supervised dimension reduction techniques one shoulsd re-train the dimension reduction method if new tags would be added

# Additional comments

I didn't do the warmup task, as i did not have access to gpu when writing this homework.
I was considering about adding a script that uses requests library and does load testing of the API, but the task specifically requested to measure difference of GPU/CPU, so i skipped this task. 

repo is still private, Tasks.md didnt mention to put it to public, but i don't have the reviewer's github account name, so i can't invite them as collaborators to a private repository :(

https://github.com/csuzs/tagsearch_hw