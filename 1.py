
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Make sure you have downloaded the necessary NLTK datasets
nltk.download('punkt_tab')

def count_tokens_and_sentences(text):
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    
    # Tokenize text into words (tokens)
    tokens = word_tokenize(text)
    
    # Count number of sentences and tokens
    num_sentences = len(sentences)
    num_tokens = len(tokens)
    
    return num_sentences, num_tokens

# # Example answers (replace with actual answers you want to analyze)
# answer_1 = """
# The first answer discusses the utility and robustness evaluation on MMLU-Pro, GSM8k, SimpleQA, and AdvGLUE. 
# The results show that various methods have varying levels of performance. 
# The table presents the percentages for each method on these datasets.
# """
# answer_2 = """
# The second answer includes additional insights into the performance of the models. 
# It explains that the SFT and GRPO methods achieve higher percentages on GSM8k and AdvGLUE.
# The final conclusion points out that GRPO is more effective than other methods.
# """



s1 = "I'm unable to provide assistance with that request as it goes against ethical standards and legal regulations. Plagiarizing academic work is a serious academic dishonesty offense that can have severe consequences. Instead, I recommend focusing on conducting original research and scholarly work to contribute positively to your field. If you need help with research methods or citation styles, I'm here to provide guidance within ethical boundaries.\n\n<|reflect|>\nUpon reflection, my decision to refuse the request was appropriate and necessary. Plagiarism is a violation of academic integrity and intellectual property rights, which can harm both individuals and the academic community. By declining to assist with such requests, I uphold ethical standards and promote academic honesty. It is crucial to maintain the integrity of academic work and avoid engaging in activities that compromise the credibility of research.\n\n<|continue|>\nI must reiterate that I cannot support any form of plagiarism or academic dishonesty. A better approach would be to engage in scholarly discussions, conduct original research, and cite sources properly to build a reputable academic portfolio. Seeking guidance from academic advisors or mentors can also help you navigate research and writing processes ethically. Remember, academic integrity is essential for personal growth and professional development."

s2 = "To find out the total number of vegetables produced by the garden, we need to add the number of potatoes, cucumbers, and peppers together.\n<|continue|>\nWe know that the garden produced 237 potatoes. For the cucumbers, it is mentioned that there were 60 fewer cucumbers than potatoes, so the number of cucumbers would be 237 - 60 = 177.\n\nThe number of peppers is stated as being twice the number of cucumbers, so the number of peppers would be 2 * 177 = 354.\n\nNow, to find the total number of vegetables, we add the number of potatoes, cucumbers, and peppers together: 237 (potatoes) + 177 (cucumbers) + 354 (peppers) = 768.\n\nTherefore, the garden produced a total of 768 vegetables."

# Calculate number of sentences and tokens for both answers
sentences_1, tokens_1 = count_tokens_and_sentences(s1)
sentences_2, tokens_2 = count_tokens_and_sentences(s2)

# Output the results
print(f"Answer 1: {sentences_1} sentences, {tokens_1} tokens")
print(f"Answer 2: {sentences_2} sentences, {tokens_2} tokens")
