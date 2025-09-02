from IPython.display import HTML, display
from transformers import BertTokenizer

def bert_tokenize_and_color(text: list, tokenizer: BertTokenizer) -> None:
    """Visualize BERT tokenization by coloring tokens.
    args:
        text (list): List of strings to tokenize and visualize.
        tokenizer (BertTokenizer): BERT tokenizer instance.
    returns:
        None
    """
    colored_text = ""
    colors = ['#FF5733', '#33FF57', '#3357FF', '#FFD700', '#00CED1', '#FF00FF', '#FFFF00',
              '#FF0000', '#00FF00', '#0000FF', '#00FFFF', '#FF1493', '#8A2BE2',
              '#FF8C00', '#228B22', '#DC143C', '#32CD32', '#1E90FF', '#FFD700', '#FF69B4']

    for line in text:
        line_html = ""
        tokens = tokenizer.tokenize(line)
        for token in tokens:
            color = colors[hash(token) % len(colors)]
            token_html = f'<span style="background-color:{color}; color: white; margin-right: 5px;">{token}</span>'
            line_html += token_html
        colored_text += f'<div style="margin-bottom: 10px;">{line_html}</div>'

    display(HTML(colored_text))